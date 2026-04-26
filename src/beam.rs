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
        suppress_tokens: &dyn Fn(usize, &mut [f32]),
    ) -> anyhow::Result<()> {
        let mut candidates = Vec::new();

        for beam in self.beams.drain(..) {
            if beam.finished {
                self.finished_beams.push(beam);
                continue;
            }

            let mut logits = beam.last_logits.clone();

            // Apply suppression rules
            suppress_tokens(beam.tokens.len(), &mut logits);

            // Log-softmax
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
                candidates.push(Candidate {
                    parent_log_prob: beam.log_prob,
                    log_prob: log_p,
                    token_id,
                    parent_tokens: beam.tokens.clone(),
                    parent_state: beam.state.clone(),
                });
            }
        }

        candidates.sort_by(|a, b| {
            let score_a = a.parent_log_prob + a.log_prob;
            let score_b = b.parent_log_prob + b.log_prob;
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });

        for cand in candidates.into_iter().take(self.size) {
            let mut tokens = cand.parent_tokens;
            tokens.push(cand.token_id);

            let mut finished = cand.token_id == S::TOKEN_EOT;

            // Repetition check (4+4)
            if tokens.len() >= 8 {
                let tail = &tokens[tokens.len() - 8..];
                if tail[0..4] == tail[4..8] {
                    finished = true;
                }
            }

            if finished {
                self.finished_beams.push(Beam {
                    tokens,
                    log_prob: cand.parent_log_prob + cand.log_prob,
                    state: cand.parent_state,
                    last_logits: Vec::new(),
                    finished: true,
                    _phantom: std::marker::PhantomData,
                });
            } else {
                let mut state = cand.parent_state;
                let next_logits = decoder.step(&mut state, cand.token_id)?;
                self.beams.push(Beam {
                    tokens,
                    log_prob: cand.parent_log_prob + cand.log_prob,
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
        let mut all = self.beams.clone();
        all.extend(self.finished_beams.clone());

        all.sort_by(|a, b| {
            b.score(self.alpha)
                .partial_cmp(&a.score(self.alpha))
                .unwrap_or(Ordering::Equal)
        });

        all.first().map(|b| b.tokens.clone())
    }
}

struct Candidate {
    parent_log_prob: f32,
    log_prob: f32,
    token_id: u32,
    parent_tokens: Vec<u32>,
    parent_state: WhisperDecoderState,
}
