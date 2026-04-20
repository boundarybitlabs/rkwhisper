use crate::decoder::{WhisperDecoder, WhisperDecoderState};
use crate::spec::WhisperSpec;
use std::cmp::Ordering;

pub struct Beam<S: WhisperSpec> {
    pub tokens: Vec<u32>,
    pub log_prob: f32,
    pub state: WhisperDecoderState,
    pub finished: bool,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: WhisperSpec> Clone for Beam<S> {
    fn clone(&self) -> Self {
        Self {
            tokens: self.tokens.clone(),
            log_prob: self.log_prob,
            state: self.state.clone(),
            finished: self.finished,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<S: WhisperSpec> Beam<S> {
    pub fn new(tokens: Vec<u32>, log_prob: f32, state: WhisperDecoderState) -> Self {
        Self {
            tokens,
            log_prob,
            state,
            finished: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Calculate score with length penalty.
    /// score = log_prob / length^alpha
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
        initial_tokens: Vec<u32>,
        alpha: f32,
    ) -> Self {
        Self {
            size,
            beams: vec![Beam::new(initial_tokens, 0.0, initial_state)],
            finished_beams: Vec::new(),
            alpha,
        }
    }

    /// Run one step of beam search.
    pub fn step(&mut self, decoder: &mut WhisperDecoder<S>) -> anyhow::Result<()> {
        let mut candidates = Vec::new();

        // 1. Collect top candidates from each beam
        for beam in self.beams.drain(..) {
            if beam.finished {
                self.finished_beams.push(beam);
                continue;
            }

            let last_token = *beam.tokens.last().unwrap();
            let mut state = beam.state.clone();
            let logits = decoder.step(&mut state, last_token)?;

            // Log-softmax
            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
            let log_sum_exp = max_logit + sum_exp.ln();

            // Find top N tokens for THIS beam to avoid cloning state for everything
            let mut beam_candidates: Vec<(f32, u32)> = logits
                .iter()
                .enumerate()
                .map(|(id, &l)| (l - log_sum_exp, id as u32))
                .collect();

            // Partial sort to get top N
            beam_candidates.select_nth_unstable_by(self.size, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
            });
            beam_candidates.truncate(self.size);

            for (log_p, token_id) in beam_candidates {
                // We don't clone the state yet! Just store the parent beam index and token.
                // But since we drained `self.beams`, we need to store the data we need.
                candidates.push(Candidate {
                    parent_log_prob: beam.log_prob,
                    log_prob: log_p,
                    token_id,
                    parent_tokens: beam.tokens.clone(),
                    parent_state: beam.state.clone(), // This is still a bit heavy, but better than VOCAB clones
                });
            }
        }

        // 2. Sort all candidates (N * N) globally
        candidates.sort_by(|a, b| {
            let score_a = a.parent_log_prob + a.log_prob;
            let score_b = b.parent_log_prob + b.log_prob;
            score_b.partial_cmp(&score_a).unwrap_or(Ordering::Equal)
        });

        // 3. Keep top N
        for cand in candidates.into_iter().take(self.size) {
            let mut tokens = cand.parent_tokens;
            tokens.push(cand.token_id);

            let state = cand.parent_state;
            // We need to advance the state for the token we just added
            // Wait, the decoder was already run for the parent's last token.
            // The state we have in Candidate is the state AFTER decoder.step(parent_last_token).
            // So it's already updated.

            let finished = cand.token_id == S::TOKEN_EOT;
            let beam = Beam {
                tokens,
                log_prob: cand.parent_log_prob + cand.log_prob,
                state,
                finished,
                _phantom: std::marker::PhantomData,
            };

            if finished {
                self.finished_beams.push(beam);
            } else {
                self.beams.push(beam);
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
