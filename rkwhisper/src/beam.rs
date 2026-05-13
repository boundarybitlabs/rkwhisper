use crate::decoder::{WhisperDecoder, WhisperDecoderState};
use crate::spec::WhisperSpec;
use std::cmp::Ordering;

pub struct Beam<S: WhisperSpec> {
    pub tokens: Vec<u32>,
    pub log_prob: f32,
    pub state: WhisperDecoderState,
    pub last_logits: Vec<f32>,
    pub finished: bool,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: WhisperSpec> Clone for Beam<S> {
    fn clone(&self) -> Self {
        Self {
            tokens: self.tokens.clone(),
            log_prob: self.log_prob,
            state: self.state.clone(),
            last_logits: self.last_logits.clone(),
            finished: self.finished,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S: WhisperSpec> Beam<S> {
    pub fn new(
        tokens: Vec<u32>,
        log_prob: f32,
        state: WhisperDecoderState,
        last_logits: Vec<f32>,
    ) -> Self {
        Self {
            tokens,
            log_prob,
            state,
            last_logits,
            finished: false,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn score(&self, alpha: f32) -> f32 {
        let len = self.tokens.len() as f32;
        if len == 0.0 {
            return self.log_prob;
        }
        self.log_prob / len.powf(alpha)
    }
}

pub struct BeamSearch<S: WhisperSpec> {
    pub size: usize,
    pub beams: Vec<Beam<S>>,
    pub finished_beams: Vec<Beam<S>>,
    pub alpha: f32,
}

impl<S: WhisperSpec> BeamSearch<S> {
    pub fn new(
        size: usize,
        initial_state: WhisperDecoderState,
        initial_logits: Vec<f32>,
        initial_tokens: Vec<u32>,
        alpha: f32,
    ) -> Self {
        Self {
            size,
            beams: vec![Beam::new(
                initial_tokens,
                0.0,
                initial_state,
                initial_logits,
            )],
            finished_beams: Vec::new(),
            alpha,
        }
    }

    pub fn step(
        &mut self,
        decoder: &mut WhisperDecoder<S>,
        suppress_tokens: &dyn Fn(&[u32], &mut [f32]),
    ) -> anyhow::Result<()> {
        // Drain beams into Options so each parent can be moved on its last use or cloned
        // for earlier uses, avoiding clones for candidates that won't survive selection.
        let mut beams: Vec<Option<Beam<S>>> = self.beams.drain(..).map(Some).collect();

        // Phase 1: collect lightweight candidates — no state/token cloning yet.
        // Tuple: (beam_idx, token_id, parent_log_prob, token_log_prob)
        let mut candidates: Vec<(usize, u32, f32, f32)> = Vec::new();

        for (beam_idx, beam_opt) in beams.iter().enumerate() {
            let beam = beam_opt.as_ref().unwrap();
            if beam.finished {
                continue;
            }

            let mut logits = beam.last_logits.clone();
            suppress_tokens(&beam.tokens, &mut logits);

            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
            let log_sum_exp = max_logit + sum_exp.ln();

            let mut beam_candidates: Vec<(f32, u32)> = logits
                .iter()
                .enumerate()
                .map(|(id, &l)| (l - log_sum_exp, id as u32))
                .collect();
            beam_candidates.select_nth_unstable_by(self.size, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
            });
            beam_candidates.truncate(self.size);

            for (log_p, token_id) in beam_candidates {
                candidates.push((beam_idx, token_id, beam.log_prob, log_p));
            }
        }

        // Select the global top-`size` candidates before touching any state.
        candidates.sort_by(|a, b| {
            let score_a = a.2 + a.3;
            let score_b = b.2 + b.3;
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });
        candidates.truncate(self.size);

        // Move any already-finished beams from the drained set into finished_beams.
        for beam_opt in beams.iter_mut() {
            if beam_opt.as_ref().map_or(false, |b| b.finished) {
                self.finished_beams.push(beam_opt.take().unwrap());
            }
        }

        // Phase 2: build new beams, counting down references so the last use of each
        // parent beam moves its state rather than cloning it.
        let mut ref_counts = vec![0usize; beams.len()];
        for &(beam_idx, _, _, _) in &candidates {
            ref_counts[beam_idx] += 1;
        }

        for (beam_idx, token_id, parent_log_prob, token_log_prob) in candidates {
            ref_counts[beam_idx] -= 1;
            let (parent_tokens, parent_state) = if ref_counts[beam_idx] == 0 {
                let beam = beams[beam_idx].take().unwrap();
                (beam.tokens, beam.state)
            } else {
                let beam = beams[beam_idx].as_ref().unwrap();
                (beam.tokens.clone(), beam.state.clone())
            };

            let mut tokens = parent_tokens;
            tokens.push(token_id);
            let combined_log_prob = parent_log_prob + token_log_prob;

            if token_id == S::TOKEN_EOT {
                self.finished_beams.push(Beam {
                    tokens,
                    log_prob: combined_log_prob,
                    state: parent_state,
                    last_logits: Vec::new(),
                    finished: true,
                    _phantom: std::marker::PhantomData,
                });
            } else {
                let mut state = parent_state;
                let next_logits = decoder.step(&mut state, token_id)?;
                self.beams.push(Beam {
                    tokens,
                    log_prob: combined_log_prob,
                    state,
                    last_logits: next_logits,
                    finished: false,
                    _phantom: std::marker::PhantomData,
                });
            }
        }

        Ok(())
    }

    pub fn best_result(&self) -> Option<Vec<u32>> {
        self.beams
            .iter()
            .chain(self.finished_beams.iter())
            .max_by(|a, b| {
                a.score(self.alpha)
                    .partial_cmp(&b.score(self.alpha))
                    .unwrap_or(Ordering::Equal)
            })
            .map(|b| b.tokens.clone())
    }
}
