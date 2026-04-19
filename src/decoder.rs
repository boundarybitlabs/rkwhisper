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
///     1 ..= L     past_k_l*       [1, T, D, H]        Float16 NHWC
///     L+1..2L     past_v_l*       [1, T, D, H]        Float16 NHWC
///     2L+1..3L    enc_k_l*        [1, D, H, ENC_SEQ]  Float16 NHWC
///     3L+1..4L    enc_v_l*        [1, D, H, ENC_SEQ]  Float16 NHWC
///     4L+1        self_attn_mask  [1, 1, T, 1]        Float16 NHWC
///     4L+2        kv_insert_mask  [1, T, 1, 1]        Float16 NHWC
///     4L+3        kv_retain_mask  [1, T, 1, 1]        Float16 NHWC
///     4L+4        pos_idx         [1]                 Int64   UNDEFINED
///
///   Outputs
///     0           logits          [1, 1, VOCAB]       Float16 UNDEFINED
///     1 ..= L     present_k_l*    [1, H, 1, D]        Float16 NCHW
///     L+1..2L     present_v_l*    [1, H, 1, D]        Float16 NCHW
pub struct WhisperDecoder<'a, S: WhisperSpec> {
    rknn_dec: &'a RKNN<RuntimeAPI>,

    /// Per-layer encoder cross-attention K/V — set once per utterance via
    /// `set_enc_kv`.  Each entry: NHWC [1, D_HEAD, N_HEADS, ENC_SEQ],
    /// flat len = D * H * ENC_SEQ.
    enc_k: Vec<Vec<f16>>,
    enc_v: Vec<Vec<f16>>,

    /// Per-layer self-attention KV cache in NHWC [1, T_CACHE, D_HEAD, N_HEADS].
    /// Flat len per layer = T * D * H.
    past_k: Vec<Vec<f16>>,
    past_v: Vec<Vec<f16>>,

    /// Attention control masks. All are contiguous [1, ..., 1].
    self_attn_mask: Vec<f16>,
    kv_insert_mask: Vec<f16>,
    kv_retain_mask: Vec<f16>,

    /// Scalar inputs.
    token_i64: [i64; 1],
    pos_i64: [i64; 1],

    /// Logits output buffer, flat len = VOCAB.
    logits_f16: Vec<f16>,

    /// Absolute step counter — number of tokens fed to the decoder since
    /// the last `reset()`.
    pos: usize,

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
        Self {
            rknn_dec,
            enc_k: vec![vec![zero; d * h * s]; l],
            enc_v: vec![vec![zero; d * h * s]; l],
            past_k: vec![vec![zero; t * d * h]; l],
            past_v: vec![vec![zero; t * d * h]; l],
            self_attn_mask: vec![zero; t],
            kv_insert_mask: vec![zero; t],
            kv_retain_mask: vec![zero; t],
            token_i64: [0],
            pos_i64: [0],
            logits_f16: vec![zero; S::VOCAB],
            pos: 0,
            phantom: core::marker::PhantomData,
        }
    }

    /// Provide pre-computed per-layer encoder K/V (output of the enc-KV model).
    /// Each input slice is in NCHW [1, ENC_SEQ, N_HEADS, D_HEAD] order (kv-init
    /// output format). Transposes to NHWC [1, D_HEAD, N_HEADS, ENC_SEQ] for the
    /// decoder input.
    pub fn set_enc_kv(&mut self, enc_k: Vec<Vec<f16>>, enc_v: Vec<Vec<f16>>) {
        debug_assert_eq!(enc_k.len(), S::N_LAYERS);
        debug_assert_eq!(enc_v.len(), S::N_LAYERS);
        let l = S::N_LAYERS;
        let d = S::D_HEAD;
        let h = S::N_HEADS;
        let s = S::ENC_SEQ;
        for layer in 0..l {
            nchw_enc_to_nhwc(&enc_k[layer], &mut self.enc_k[layer], d, h, s);
            nchw_enc_to_nhwc(&enc_v[layer], &mut self.enc_v[layer], d, h, s);
        }
    }

    /// Clear the self-attention KV cache and reset the step counter.
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

    /// Run one decoder step and return logits as f32.
    pub fn step(&mut self, token_id: u32) -> Result<Vec<f32>, rknpu2::Error> {
        debug_assert!(
            !self.enc_k[0].iter().all(|x| x.to_f32() == 0.0),
            "set_enc_kv() must be called before step()"
        );

        let l = S::N_LAYERS;
        let t = S::T_CACHE;
        let d = S::D_HEAD;
        let h = S::N_HEADS;
        let pos = self.pos;
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
                buffer: BufView::F16(&self.past_k[layer]),
                pass_through: false,
                fmt: TensorFormatKind::NHWC(TensorFormat::NHWC),
            });
        }
        for layer in 0..l {
            inputs.push(Input {
                index: (l + 1 + layer) as u32,
                buffer: BufView::F16(&self.past_v[layer]),
                pass_through: false,
                fmt: TensorFormatKind::NHWC(TensorFormat::NHWC),
            });
        }
        for layer in 0..l {
            inputs.push(Input {
                index: (2 * l + 1 + layer) as u32,
                buffer: BufView::F16(&self.enc_k[layer]),
                pass_through: false,
                fmt: TensorFormatKind::NHWC(TensorFormat::NHWC),
            });
        }
        for layer in 0..l {
            inputs.push(Input {
                index: (3 * l + 1 + layer) as u32,
                buffer: BufView::F16(&self.enc_v[layer]),
                pass_through: false,
                fmt: TensorFormatKind::NHWC(TensorFormat::NHWC),
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
        let hd = h * d;
        let mut present_k: Vec<Vec<f16>> = vec![vec![zero; hd]; l];
        let mut present_v: Vec<Vec<f16>> = vec![vec![zero; hd]; l];

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

        // Write present_k/v (NCHW [1,H,1,D]) into the cache (NHWC [1,T,D,H])
        // at write_slot.
        for layer in 0..l {
            write_present_nchw_to_nhwc_slot(
                &present_k[layer],
                &mut self.past_k[layer],
                write_slot,
                d,
                h,
            );
            write_present_nchw_to_nhwc_slot(
                &present_v[layer],
                &mut self.past_v[layer],
                write_slot,
                d,
                h,
            );
        }

        self.pos += 1;

        let logits: Vec<f32> = self.logits_f16.iter().map(|x| x.to_f32()).collect();
        Ok(logits)
    }

    /// Prime the KV cache by stepping through every token in `prompt`.
    pub fn prime(&mut self, prompt: &[u32]) -> Result<Vec<f32>, rknpu2::Error> {
        let mut logits = vec![0f32; S::VOCAB];
        for &id in prompt {
            logits = self.step(id)?;
        }
        Ok(logits)
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Transpose kv-init NCHW [1, ENC_SEQ, N_HEADS, D_HEAD] → decoder NHWC [1, D_HEAD, N_HEADS, ENC_SEQ].
///
///   src index: seq * head * D + head * D + dim
///   dst index: head * dim * S + dim * S + seq
fn nchw_enc_to_nhwc(src: &[f16], dst: &mut [f16], d: usize, h: usize, s: usize) {
    debug_assert_eq!(src.len(), s * d * h);
    debug_assert_eq!(dst.len(), d * h * s);
    for seq in 0..s {
        for head in 0..h {
            for dim in 0..d {
                // src is [Seq, Head, Dim] -> seq*H*D + head*D + dim
                // dst is [Head, Dim, Seq] -> head*D*S + dim*S + seq
                dst[head * d * s + dim * s + seq] = src[seq * h * d + head * d + dim];
            }
        }
    }
}

/// Copy a single-token present slice (NCHW [1, H, 1, D], flat len = H*D)
/// into a cache buffer (NHWC [1, T, D, H]) at position `slot`.
///
///   NCHW index: head * D + dim
///   NHWC index: slot * D * H + dim * H + head
#[inline]
fn write_present_nchw_to_nhwc_slot(
    present: &[f16],  // flat NCHW  len = H * D
    past: &mut [f16], // flat NHWC  len = T * D * H
    slot: usize,
    d: usize,
    h: usize,
) {
    debug_assert_eq!(present.len(), h * d);
    for head in 0..h {
        for dim in 0..d {
            past[slot * d * h + dim * h + head] = present[head * d + dim];
        }
    }
}
