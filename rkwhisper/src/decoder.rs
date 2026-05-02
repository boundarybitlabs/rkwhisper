use crate::spec::WhisperSpec;
use rknpu2::api::runtime::RuntimeAPI;
use rknpu2::io::buffer::{BufMutView, BufView};
use rknpu2::io::input::Input;
use rknpu2::io::output::{Output, OutputKind};
use rknpu2::tensor::{TensorFormat, TensorFormatKind};
use rknpu2::{RKNN, f16};

/// Per-step decoder for models with per-layer KV inputs/outputs and mask-based
/// cache management.
///
/// Expected model I/O layout (all indices 0-based, L = N_LAYERS):
///
///   Inputs
///     0           tokens          [1, 1]              Int64   UNDEFINED
///     1 ..= L     past_k_l*       [1, ceil(H/8), T, D, 8]        Float16 NC1HWC2
///     L+1..2L     past_v_l*       [1, ceil(H/8), T, D, 8]        Float16 NC1HWC2
///     2L+1..3L    enc_k_l*        [1, ceil(ENC_SEQ/8), H, D, 8]  Float16 NC1HWC2
///     3L+1..4L    enc_v_l*        [1, ceil(ENC_SEQ/8), H, D, 8]  Float16 NC1HWC2
///     4L+1        self_attn_mask  [1, 1, T, 1]        Float16 NHWC
///     4L+2        kv_insert_mask  [1, T, 1, 1]        Float16 NHWC
///     4L+3        kv_retain_mask  [1, T, 1, 1]        Float16 NHWC
///     4L+4        pos_idx         [1]                 Int64   UNDEFINED
///
///   Outputs
///     0           logits          [1, 1, VOCAB]       Float16 UNDEFINED
///     1 ..= L     present_k_l*    [1, H, 1, D]        Float16 NCHW from rknn_outputs_get
///     L+1..2L     present_v_l*    [1, H, 1, D]        Float16 NCHW from rknn_outputs_get
/// Per-step decoder state (KV cache and position).
/// Can be cloned to support beam search.
#[derive(Clone)]
pub struct WhisperDecoderState {
    /// Per-layer self-attention KV cache in native NC1HWC2
    /// [1, ceil(N_HEADS/8), T_CACHE, D_HEAD, 8].
    pub past_k: Vec<Vec<f16>>,
    pub past_v: Vec<Vec<f16>>,

    /// Absolute step counter — number of tokens fed to the decoder since
    /// the last `reset()`.
    pub pos: usize,
}

impl WhisperDecoderState {
    pub fn new<S: WhisperSpec>() -> Self {
        let zero = f16::from_f32(0.0);
        let l = S::N_LAYERS;
        let t = S::T_CACHE;
        let d = S::D_HEAD;
        let h = S::N_HEADS;
        let past_len = nc1hwc2_len(h, t, d);
        Self {
            past_k: vec![vec![zero; past_len]; l],
            past_v: vec![vec![zero; past_len]; l],
            pos: 0,
        }
    }

    pub fn reset(&mut self) {
        let zero = f16::from_f32(0.0);
        for k in &mut self.past_k {
            k.fill(zero);
        }
        for v in &mut self.past_v {
            v.fill(zero);
        }
        self.pos = 0;
    }
}

pub struct WhisperDecoder<'a, S: WhisperSpec> {
    rknn_dec: &'a RKNN<RuntimeAPI>,

    /// Per-layer encoder cross-attention K/V in native NC1HWC2
    /// [1, ceil(ENC_SEQ/8), N_HEADS, D_HEAD, 8].
    enc_k: Vec<Vec<f16>>,
    enc_v: Vec<Vec<f16>>,

    /// Attention control masks. All are contiguous [1, ..., 1].
    self_attn_mask: Vec<f16>,
    kv_insert_mask: Vec<f16>,
    kv_retain_mask: Vec<f16>,

    /// Scalar inputs.
    token_i64: [i64; 1],
    pos_i64: [i64; 1],

    /// Logits output buffer, flat len = VOCAB.
    logits_f16: Vec<f16>,

    phantom: core::marker::PhantomData<S>,
}

impl<'a, S: WhisperSpec> WhisperDecoder<'a, S> {
    pub fn new(rknn_dec: &'a RKNN<RuntimeAPI>) -> Self {
        let zero = f16::from_f32(0.0);
        let l = S::N_LAYERS;
        let t = S::T_CACHE;
        let d = S::D_HEAD;
        let h = S::N_HEADS;
        let s = S::ENC_SEQ;
        let enc_len = nc1hwc2_len(s, h, d);
        Self {
            rknn_dec,
            enc_k: vec![vec![zero; enc_len]; l],
            enc_v: vec![vec![zero; enc_len]; l],
            self_attn_mask: vec![zero; t],
            kv_insert_mask: vec![zero; t],
            kv_retain_mask: vec![zero; t],
            token_i64: [0],
            pos_i64: [0],
            logits_f16: vec![zero; S::VOCAB],
            phantom: core::marker::PhantomData,
        }
    }

    /// Provide pre-computed per-layer encoder K/V (output of the enc-KV model).
    /// Each input slice is logical [1, ENC_SEQ, N_HEADS, D_HEAD] order.
    /// Packs to native NC1HWC2 [1, ceil(ENC_SEQ/8), N_HEADS, D_HEAD, 8].
    pub fn set_enc_kv(&mut self, enc_k: Vec<Vec<f16>>, enc_v: Vec<Vec<f16>>) {
        debug_assert_eq!(enc_k.len(), S::N_LAYERS);
        debug_assert_eq!(enc_v.len(), S::N_LAYERS);
        let l = S::N_LAYERS;
        let logical_len = S::ENC_SEQ * S::N_HEADS * S::D_HEAD;
        let enc_len = nc1hwc2_len(S::ENC_SEQ, S::N_HEADS, S::D_HEAD);
        for layer in 0..l {
            debug_assert_eq!(enc_k[layer].len(), logical_len);
            debug_assert_eq!(enc_v[layer].len(), logical_len);
            debug_assert_eq!(self.enc_k[layer].len(), enc_len);
            debug_assert_eq!(self.enc_v[layer].len(), enc_len);
            pack_enc_kv_to_nc1hwc2(
                &enc_k[layer],
                &mut self.enc_k[layer],
                S::ENC_SEQ,
                S::N_HEADS,
                S::D_HEAD,
            );
            pack_enc_kv_to_nc1hwc2(
                &enc_v[layer],
                &mut self.enc_v[layer],
                S::ENC_SEQ,
                S::N_HEADS,
                S::D_HEAD,
            );
        }
    }

    /// Run one decoder step and return logits as f32.
    pub fn step(
        &mut self,
        state: &mut WhisperDecoderState,
        token_id: u32,
    ) -> Result<Vec<f32>, rknpu2::Error> {
        debug_assert!(
            !self.enc_k[0].iter().all(|x| x.to_f32() == 0.0),
            "set_enc_kv() must be called before step()"
        );

        let l = S::N_LAYERS;
        let t = S::T_CACHE;
        let d = S::D_HEAD;
        let h = S::N_HEADS;
        let pos = state.pos;
        let write_slot = pos % t;

        let zero = f16::from_f32(0.0);
        let one = f16::from_f32(1.0);
        let neg_inf = f16::from_f32(-65504.0);

        // All masks contiguous (Size(B)=896).
        for i in 0..t {
            self.self_attn_mask[i] = if pos >= t || i <= pos { zero } else { neg_inf };
            self.kv_insert_mask[i] = if i == write_slot { one } else { zero };
            // Retain all previously written tokens except the one we are about to overwrite.
            self.kv_retain_mask[i] = if i != write_slot && (pos >= t || i < pos) {
                one
            } else {
                zero
            };
        }

        self.token_i64[0] = token_id as i64;
        self.pos_i64[0] = pos as i64;

        // --- inputs ---
        let mut inputs = Vec::with_capacity(4 * l + 5);

        inputs.push(Input {
            index: 0,
            buffer: BufView::I64(&self.token_i64),
            pass_through: false,
            fmt: TensorFormatKind::UNDEFINED(TensorFormat::UNDEFINED),
        });
        for layer in 0..l {
            inputs.push(Input {
                index: (1 + layer) as u32,
                buffer: BufView::F16(&state.past_k[layer]),
                pass_through: true,
                fmt: TensorFormatKind::NC1HWC2(TensorFormat::NC1HWC2),
            });
        }
        for layer in 0..l {
            inputs.push(Input {
                index: (l + 1 + layer) as u32,
                buffer: BufView::F16(&state.past_v[layer]),
                pass_through: true,
                fmt: TensorFormatKind::NC1HWC2(TensorFormat::NC1HWC2),
            });
        }
        for layer in 0..l {
            inputs.push(Input {
                index: (2 * l + 1 + layer) as u32,
                buffer: BufView::F16(&self.enc_k[layer]),
                pass_through: true,
                fmt: TensorFormatKind::NC1HWC2(TensorFormat::NC1HWC2),
            });
        }
        for layer in 0..l {
            inputs.push(Input {
                index: (3 * l + 1 + layer) as u32,
                buffer: BufView::F16(&self.enc_v[layer]),
                pass_through: true,
                fmt: TensorFormatKind::NC1HWC2(TensorFormat::NC1HWC2),
            });
        }
        inputs.push(Input {
            index: (4 * l + 1) as u32,
            buffer: BufView::F16(&self.self_attn_mask),
            pass_through: false,
            fmt: TensorFormatKind::NHWC(TensorFormat::NHWC),
        });
        inputs.push(Input {
            index: (4 * l + 2) as u32,
            buffer: BufView::F16(&self.kv_insert_mask),
            pass_through: false,
            fmt: TensorFormatKind::NHWC(TensorFormat::NHWC),
        });
        inputs.push(Input {
            index: (4 * l + 3) as u32,
            buffer: BufView::F16(&self.kv_retain_mask),
            pass_through: false,
            fmt: TensorFormatKind::NHWC(TensorFormat::NHWC),
        });
        inputs.push(Input {
            index: (4 * l + 4) as u32,
            buffer: BufView::I64(&self.pos_i64),
            pass_through: false,
            fmt: TensorFormatKind::UNDEFINED(TensorFormat::UNDEFINED),
        });

        self.rknn_dec.set_inputs(inputs)?;
        self.rknn_dec.run()?;

        // --- outputs ---
        // Allocate per-layer present buffers locally; iter_mut gives the
        // borrow-checker the disjointness proof it needs.
        let present_len = h * d;
        let mut present_k: Vec<Vec<f16>> = vec![vec![zero; present_len]; l];
        let mut present_v: Vec<Vec<f16>> = vec![vec![zero; present_len]; l];

        let mut outputs: Vec<Output<'_>> = Vec::with_capacity(2 * l + 1);
        outputs.push(Output {
            index: 0,
            kind: OutputKind::Preallocated {
                buf: BufMutView::F16(&mut self.logits_f16),
                want_float: false,
            },
        });
        for (layer, buf) in present_k.iter_mut().enumerate() {
            outputs.push(Output {
                index: (1 + layer) as u32,
                kind: OutputKind::Preallocated {
                    buf: BufMutView::F16(buf),
                    want_float: false,
                },
            });
        }
        for (layer, buf) in present_v.iter_mut().enumerate() {
            outputs.push(Output {
                index: (l + 1 + layer) as u32,
                kind: OutputKind::Preallocated {
                    buf: BufMutView::F16(buf),
                    want_float: false,
                },
            });
        }

        self.rknn_dec.get_outputs(&mut outputs)?;

        // rknn_outputs_get returns present_k/v in logical NCHW [1, H, 1, D].
        // Pack that single step into the native NC1HWC2 cache
        // [1, ceil(H/8), T, D, 8] at write_slot.
        for layer in 0..l {
            write_present_nchw_to_nc1hwc2_cache_slot(
                &present_k[layer],
                &mut state.past_k[layer],
                write_slot,
                t,
                d,
                h,
            );
            write_present_nchw_to_nc1hwc2_cache_slot(
                &present_v[layer],
                &mut state.past_v[layer],
                write_slot,
                t,
                d,
                h,
            );
        }

        state.pos += 1;

        let logits: Vec<f32> = self.logits_f16.iter().map(|x| x.to_f32()).collect();
        Ok(logits)
    }

    /// Prime the KV cache by stepping through every token in `prompt`.
    pub fn prime(
        &mut self,
        state: &mut WhisperDecoderState,
        prompt: &[u32],
    ) -> Result<Vec<f32>, rknpu2::Error> {
        let mut logits = vec![0f32; S::VOCAB];
        for &id in prompt {
            logits = self.step(state, id)?;
        }
        Ok(logits)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

#[inline]
fn nc1hwc2_len(channels: usize, height: usize, width: usize) -> usize {
    channels.div_ceil(8) * height * width * 8
}

/// Pack logical encoder KV [1, ENC_SEQ, N_HEADS, D_HEAD] into native NC1HWC2
/// [1, ceil(ENC_SEQ/8), N_HEADS, D_HEAD, 8].
#[inline]
fn pack_enc_kv_to_nc1hwc2(src: &[f16], dst: &mut [f16], s: usize, h: usize, d: usize) {
    debug_assert_eq!(src.len(), s * h * d);
    debug_assert_eq!(dst.len(), nc1hwc2_len(s, h, d));
    dst.fill(f16::from_f32(0.0));
    for seq in 0..s {
        let block = seq / 8;
        let lane = seq % 8;
        for head in 0..h {
            for dim in 0..d {
                let src_idx = (seq * h + head) * d + dim;
                let dst_idx = ((block * h + head) * d + dim) * 8 + lane;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

/// Pack a single-token present slice NCHW [1, H, 1, D] into a cache buffer
/// NC1HWC2 [1, ceil(H/8), T, D, 8] at position `slot`.
#[inline]
fn write_present_nchw_to_nc1hwc2_cache_slot(
    present: &[f16],
    past: &mut [f16],
    slot: usize,
    t: usize,
    d: usize,
    h: usize,
) {
    let c1 = h.div_ceil(8);
    debug_assert_eq!(present.len(), h * d);
    debug_assert_eq!(past.len(), c1 * t * d * 8);
    for head in 0..h {
        let block = head / 8;
        let lane = head % 8;
        for dim in 0..d {
            let src = head * d + dim;
            let dst = ((block * t + slot) * d + dim) * 8 + lane;
            past[dst] = present[src];
        }
    }
}
