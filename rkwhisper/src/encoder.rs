use crate::spec::WhisperSpec;
use rknpu2::api::runtime::RuntimeAPI;
use rknpu2::io::buffer::{BufMutView, BufView};
use rknpu2::io::input::Input;
use rknpu2::io::output::{Output, OutputKind};
use rknpu2::tensor::{TensorFormat, TensorFormatKind};
use rknpu2::{RKNN, f16};

/// Raw encoder hidden states: shape [1, ENC_SEQ, HIDDEN], stored as f16.
pub struct Encoded {
    pub data: Vec<f16>,
}

// ---------------------------------------------------------------------------
// Mel-spectrogram → encoder hidden states
// ---------------------------------------------------------------------------

pub struct WhisperEncoder<S: WhisperSpec> {
    rknn: RKNN<RuntimeAPI>,
    phantom: std::marker::PhantomData<S>,
}

impl<S: WhisperSpec> WhisperEncoder<S> {
    pub fn new(rknn: RKNN<RuntimeAPI>) -> Self {
        Self {
            rknn,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn encode(&self, audio: &[f32]) -> Result<Encoded, rknpu2::Error> {
        let mel_len = S::MEL_BINS * S::FRAMES;
        let mut wave = vec![0.0f32; mel_len];
        wave[..audio.len().min(mel_len)].copy_from_slice(&audio[..audio.len().min(mel_len)]);

        self.rknn.set_inputs(Input {
            index: 0,
            buffer: BufView::F32(&wave),
            pass_through: false,
            fmt: TensorFormatKind::UNDEFINED(TensorFormat::UNDEFINED),
        })?;
        self.rknn.run()?;

        let mut enc_hidden_state = vec![f16::from_f32(0.0); S::HIDDEN * S::ENC_SEQ];
        let mut enc_out = vec![Output {
            index: 0,
            kind: OutputKind::Preallocated {
                buf: BufMutView::F16(&mut enc_hidden_state),
                want_float: false,
            },
        }];
        self.rknn.get_outputs(&mut enc_out)?;

        Ok(Encoded {
            data: enc_hidden_state,
        })
    }
}

// ---------------------------------------------------------------------------
// Encoder hidden states → per-layer cross-attention K/V
// ---------------------------------------------------------------------------

/// Runs the enc-KV RKNN model and returns per-layer encoder K and V tensors.
///
/// Expected model I/O (L = N_LAYERS):
///   Input  0          : xa  f16 [1, ENC_SEQ, HIDDEN]  UNDEFINED
///   Outputs 0 ..= L-1 : enc_k_l*  f16 [1, ENC_SEQ, N_HEADS, D_HEAD]
///   Outputs L ..= 2L-1: enc_v_l*  f16 [1, ENC_SEQ, N_HEADS, D_HEAD]
pub struct EncKvModel<S: WhisperSpec> {
    rknn: RKNN<RuntimeAPI>,
    phantom: std::marker::PhantomData<S>,
}

impl<S: WhisperSpec> EncKvModel<S> {
    pub fn new(rknn: RKNN<RuntimeAPI>) -> Self {
        Self {
            rknn,
            phantom: std::marker::PhantomData,
        }
    }

    /// Returns `(enc_k, enc_v)`, each a `Vec` of `N_LAYERS` flat f16 buffers.
    /// Each buffer is logical [1, ENC_SEQ, N_HEADS, D_HEAD] order.
    pub fn compute(&self, enc: &Encoded) -> Result<(Vec<Vec<f16>>, Vec<Vec<f16>>), rknpu2::Error> {
        self.rknn.set_inputs(vec![Input {
            index: 0,
            buffer: BufView::F16(&enc.data),
            pass_through: false,
            fmt: TensorFormatKind::UNDEFINED(TensorFormat::UNDEFINED),
        }])?;
        self.rknn.run()?;

        let l = S::N_LAYERS;
        let per_layer_len = S::ENC_SEQ * S::N_HEADS * S::D_HEAD;
        let zero = f16::from_f32(0.0);

        let mut enc_k: Vec<Vec<f16>> = vec![vec![zero; per_layer_len]; l];
        let mut enc_v: Vec<Vec<f16>> = vec![vec![zero; per_layer_len]; l];

        let mut outputs: Vec<Output<'_>> = Vec::with_capacity(2 * l);
        for (i, (k_buf, v_buf)) in enc_k.iter_mut().zip(enc_v.iter_mut()).enumerate() {
            outputs.push(Output {
                index: (2 * i) as u32,
                kind: OutputKind::Preallocated {
                    buf: BufMutView::F16(k_buf),
                    want_float: false,
                },
            });
            outputs.push(Output {
                index: (2 * i + 1) as u32,
                kind: OutputKind::Preallocated {
                    buf: BufMutView::F16(v_buf),
                    want_float: false,
                },
            });
        }

        self.rknn.get_outputs(&mut outputs)?;
        Ok((enc_k, enc_v))
    }
}
