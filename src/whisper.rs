use half::f16;
use rknpu2::{prelude::*, tensor::TensorFormatKind};
use std::error::Error;

// ---------- helpers ----------

fn f32_to_f16_vec(x: &[f32]) -> Vec<f16> {
    x.iter().map(|v| f16::from_f32(*v)).collect()
}

/// NCHW [1,C,H,W] -> NHWC [1,H,W,C]
fn nchw_to_nhwc_f16(src: &[f16], c: usize, h: usize, w: usize) -> Vec<f16> {
    let mut dst = vec![f16::ZERO; h * w * c];
    // idx_nchw = ((c * H) + h) * W + w
    // idx_nhwc = ((h * W) + w) * C + c
    for ci in 0..c {
        for hi in 0..h {
            for wi in 0..w {
                let src_idx = ((ci * h) + hi) * w + wi;
                let dst_idx = ((hi * w) + wi) * c + ci;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
    dst
}

/// Slice the last `keep` time steps on NHWC [1, S, W, C] along S (the first spatial axis).
fn nhwc_slice_last_f16(buf: &[f16], s: usize, w: usize, c: usize, keep: usize) -> Vec<f16> {
    let start = s.saturating_sub(keep);
    let mut out = vec![f16::ZERO; keep * w * c];
    let src = &buf[start * w * c..(start + keep) * w * c];
    out.copy_from_slice(src);
    out
}

/// Argmax over vocab dimension of logits [1, T, V]; returns the last-step token id.
fn argmax_last_step(logits_last_step: &[f32]) -> i64 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits_last_step.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as i64
}

// ---------- main pipeline sketch ----------

fn run_whisper_pipeline(
    encoder_rknn_path: &str,
    decoder_init_rknn_path: &str,
    decoder_step_rknn_path: &str,
    mel_f32: &[f32],           // [1,128,3000] flattened
    context_input_ids: &[i64], // e.g., BOS/lang/task/notimestamps... length <= 400
    max_new_tokens: usize,
    eos_token_id: i64,
) -> Result<Vec<i64>, Box<dyn Error>> {
    // 0) Load librknnrt and models
    let lib = find_rknn_library().next().ok_or("librknnrt.so not found")?;
    let mut enc_bytes = std::fs::read(encoder_rknn_path)?;
    let mut dec_init_bytes = std::fs::read(decoder_init_rknn_path)?;
    let mut dec_step_bytes = std::fs::read(decoder_step_rknn_path)?;

    let encoder = RKNN::new_with_library(lib.clone(), &mut enc_bytes, 0)?;
    let decoder_init = RKNN::new_with_library(lib.clone(), &mut dec_init_bytes, 0)?;
    let decoder_step = RKNN::new_with_library(lib, &mut dec_step_bytes, 0)?;

    // 1) ENCODER: input_features (F32) -> last_hidden_state (F32) -> cast to F16
    //    NOTE: If your encoder input expects INT8, you can still send F32 and set pass_through=true
    //    so RKNN quantizes according to model params. Alternatively, pre-quantize yourself.
    let enc_input = Input {
        index: 0,
        buffer: BufView::F32(mel_f32),    // [1,128,3000] flattened
        pass_through: true,               // let RKNN quantize; set false if you pre-quantize
        fmt: TensorFormatKind::UNDEFINED, // encoder input is UNDEFINED in your IO table
    };
    encoder.set_inputs([enc_input])?;
    encoder.run()?;

    // Fetch encoder output as float (want_float = true)
    let mut enc_out = Output::new(0, DataType::Float32, /*pre_alloc*/ None);
    encoder.get_outputs(
        &mut [enc_out],
        OutputsPolicy {
            want_float: true,
            ..Default::default()
        },
    )?;
    let enc_hs_f32: Vec<f32> = enc_out.try_into_vec_f32()?; // shape [1,1500,1280]
    let enc_hs_f16: Vec<f16> = f32_to_f16_vec(&enc_hs_f32); // feed to FP16 decoders

    // 2) DECODER INIT: input_ids [1,N], encoder_hidden_states [1,1500,1280](F16)
    let ids_init = Input {
        index: 0,
        buffer: BufView::I64(context_input_ids),
        pass_through: true,
        fmt: TensorFormatKind::UNDEFINED,
    };
    let enc_hs_init = Input {
        index: 1,
        buffer: BufView::F16(&enc_hs_f16),
        pass_through: true,
        fmt: TensorFormatKind::UNDEFINED,
    };
    decoder_init.set_inputs([ids_init, enc_hs_init])?;
    decoder_init.run()?;

    // Pull logits [1,N,V] as float for token selection; we only need the last step’s row.
    let mut init_logits = Output { index: 0 };
    // present.* outputs: we’ll fetch by name for 4 layers
    let mut p_dec_k = [
        Output::named("present.0.decoder.key", DataType::Float16, None),
        Output::named("present.1.decoder.key", DataType::Float16, None),
        Output::named("present.2.decoder.key", DataType::Float16, None),
        Output::named("present.3.decoder.key", DataType::Float16, None),
    ];
    let mut p_dec_v = [
        Output::named("present.0.decoder.value", DataType::Float16, None),
        Output::named("present.1.decoder.value", DataType::Float16, None),
        Output::named("present.2.decoder.value", DataType::Float16, None),
        Output::named("present.3.decoder.value", DataType::Float16, None),
    ];
    let mut p_enc_k = [
        Output::named("present.0.encoder.key", DataType::Float16, None),
        Output::named("present.1.encoder.key", DataType::Float16, None),
        Output::named("present.2.encoder.key", DataType::Float16, None),
        Output::named("present.3.encoder.key", DataType::Float16, None),
    ];
    let mut p_enc_v = [
        Output::named("present.0.encoder.value", DataType::Float16, None),
        Output::named("present.1.encoder.value", DataType::Float16, None),
        Output::named("present.2.encoder.value", DataType::Float16, None),
        Output::named("present.3.encoder.value", DataType::Float16, None),
    ];

    // Ask RKNN to dequantize to float where needed (logits). K/V are already FP16.
    let mut all_outputs: Vec<&mut dyn AnyOutput> = vec![&mut init_logits];
    for i in 0..4 {
        all_outputs.push(&mut p_dec_k[i]);
        all_outputs.push(&mut p_dec_v[i]);
    }
    for i in 0..4 {
        all_outputs.push(&mut p_enc_k[i]);
        all_outputs.push(&mut p_enc_v[i]);
    }

    decoder_init.get_outputs_dyn(
        &mut all_outputs,
        OutputsPolicy {
            want_float: true,
            ..Default::default()
        },
    )?;

    // Convert K/V to NHWC and stash. Decoder K/V seq=400; Encoder K/V seq=1500.
    // Shapes from rknn-inspect: present.* are NCHW [1,20,S,64].
    let heads = 20usize;
    let head_dim = 64usize;
    let dec_seq = 400usize;
    let enc_seq = 1500usize;

    let mut past_dec_k: [Vec<f16>; 4] = std::array::from_fn(|_| Vec::new());
    let mut past_dec_v: [Vec<f16>; 4] = std::array::from_fn(|_| Vec::new());
    let mut past_enc_k: [Vec<f16>; 4] = std::array::from_fn(|_| Vec::new());
    let mut past_enc_v: [Vec<f16>; 4] = std::array::from_fn(|_| Vec::new());

    for l in 0..4 {
        let dec_k_nchw = p_dec_k[l].try_into_vec_f16()?; // [1,20,400,64]
        let dec_v_nchw = p_dec_v[l].try_into_vec_f16()?; // [1,20,400,64]
        past_dec_k[l] = nchw_to_nhwc_f16(&dec_k_nchw, heads, dec_seq, head_dim); // [1,400,64,20]
        past_dec_v[l] = nchw_to_nhwc_f16(&dec_v_nchw, heads, dec_seq, head_dim);

        let enc_k_nchw = p_enc_k[l].try_into_vec_f16()?; // [1,20,1500,64]
        let enc_v_nchw = p_enc_v[l].try_into_vec_f16()?;
        past_enc_k[l] = nchw_to_nhwc_f16(&enc_k_nchw, heads, enc_seq, head_dim); // [1,1500,64,20]
        past_enc_v[l] = nchw_to_nhwc_f16(&enc_v_nchw, heads, enc_seq, head_dim);
    }

    // Seed generated tokens with your context; pick the next token from init logits
    // init logits: [1, N, V] — take the last row (N-1)
    let vocab = 51866usize;
    let n_ctx = context_input_ids.len();
    let last_row = &init_logits.try_into_vec_f32()?[(n_ctx - 1) * vocab..n_ctx * vocab];
    let mut next_id = argmax_last_step(last_row);
    let mut out_tokens = Vec::from(context_input_ids);

    // 3) DECODER STEP LOOP
    let keep_len = 400usize; // your fixed window
    for _step in 0..max_new_tokens {
        out_tokens.push(next_id);
        if next_id == eos_token_id {
            break;
        }

        // Prepare inputs for decoder-with-past
        let ids_step = [next_id];
        let mut inputs = Vec::with_capacity(2 + 4 * 4);

        inputs.push(Input {
            index: 0,
            buffer: BufView::I64(&ids_step),
            pass_through: true,
            fmt: TensorFormatKind::UNDEFINED,
        });
        inputs.push(Input {
            index: 1,
            buffer: BufView::F16(&enc_hs_f16),
            pass_through: true,
            fmt: TensorFormatKind::UNDEFINED,
        });

        // past_key_values for 4 layers: decoder.{key,value} [1,400,64,20], encoder.{key,value} [1,1500,64,20]
        // Assume input indices 2.. are ordered to match your TOML (or use Input::named if you have it)
        for l in 0..4 {
            inputs.push(Input {
                index: 2 + l * 4 + 0,
                buffer: BufView::F16(&past_dec_k[l]),
                pass_through: true,
                fmt: TensorFormatKind::NHWC,
            });
            inputs.push(Input {
                index: 2 + l * 4 + 1,
                buffer: BufView::F16(&past_dec_v[l]),
                pass_through: true,
                fmt: TensorFormatKind::NHWC,
            });
            inputs.push(Input {
                index: 2 + l * 4 + 2,
                buffer: BufView::F16(&past_enc_k[l]),
                pass_through: true,
                fmt: TensorFormatKind::NHWC,
            });
            inputs.push(Input {
                index: 2 + l * 4 + 3,
                buffer: BufView::F16(&past_enc_v[l]),
                pass_through: true,
                fmt: TensorFormatKind::NHWC,
            });
        }

        decoder_step.set_inputs(inputs)?;
        decoder_step.run()?;

        // Fetch outputs: logits [1,1,V] + present.*.decoder.* [1,20,401,64]
        let mut step_logits = Output::named("logits", DataType::Float32, None);
        let mut pres_dec_k = [
            Output::named("present.0.decoder.key", DataType::Float16, None),
            Output::named("present.1.decoder.key", DataType::Float16, None),
            Output::named("present.2.decoder.key", DataType::Float16, None),
            Output::named("present.3.decoder.key", DataType::Float16, None),
        ];
        let mut pres_dec_v = [
            Output::named("present.0.decoder.value", DataType::Float16, None),
            Output::named("present.1.decoder.value", DataType::Float16, None),
            Output::named("present.2.decoder.value", DataType::Float16, None),
            Output::named("present.3.decoder.value", DataType::Float16, None),
        ];

        let mut outs: Vec<&mut dyn AnyOutput> = vec![&mut step_logits];
        for l in 0..4 {
            outs.push(&mut pres_dec_k[l]);
            outs.push(&mut pres_dec_v[l]);
        }

        decoder_step.get_outputs_dyn(
            &mut outs,
            OutputsPolicy {
                want_float: true,
                ..Default::default()
            },
        )?;

        // Next token from logits[1,1,V] → slice [V]
        let step_logits_v = step_logits.try_into_vec_f32()?;
        let token_id = argmax_last_step(&step_logits_v);
        next_id = token_id;

        // Update past for decoder: NCHW [1,20,401,64] -> NHWC [1,401,64,20] -> slice last 400
        let new_seq = 401usize;
        for l in 0..4 {
            let k_nchw = pres_dec_k[l].try_into_vec_f16()?;
            let v_nchw = pres_dec_v[l].try_into_vec_f16()?;
            let k_nhwc = nchw_to_nhwc_f16(&k_nchw, heads, new_seq, head_dim);
            let v_nhwc = nchw_to_nhwc_f16(&v_nchw, heads, new_seq, head_dim);
            past_dec_k[l] = nhwc_slice_last_f16(&k_nhwc, new_seq, head_dim, heads, keep_len);
            past_dec_v[l] = nhwc_slice_last_f16(&v_nhwc, new_seq, head_dim, heads, keep_len);
        }
    }

    Ok(out_tokens)
}
