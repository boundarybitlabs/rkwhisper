use crate::spec::WhisperSpec;
use anyhow::{Result, anyhow};
use std::collections::HashSet;
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SuppressTokens {
    Default,
    None,
    Ids(Vec<u32>),
}

impl SuppressTokens {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim() {
            "default" => Ok(Self::Default),
            "none" => Ok(Self::None),
            "" => Ok(Self::Ids(Vec::new())),
            ids => {
                let mut parsed = Vec::new();
                for part in ids.split(',') {
                    let token_id = part.trim().parse::<u32>().map_err(|e| {
                        anyhow!("invalid token id `{part}` in --suppress-tokens: {e}")
                    })?;
                    parsed.push(token_id);
                }
                Ok(Self::Ids(parsed))
            }
        }
    }
}

pub struct TokenSuppressor {
    prompt_len: usize,
    notimestamps: bool,
    eot: u32,
    sot: u32,
    notimestamps_id: u32,
    timestamp_begin: u32,
    ids: HashSet<u32>,
}

impl TokenSuppressor {
    pub fn new<S: WhisperSpec>(
        tokenizer: &Tokenizer,
        prompt_len: usize,
        notimestamps: bool,
        mode: &SuppressTokens,
    ) -> Result<Self> {
        let eot = S::TOKEN_EOT;
        let sot = S::TOKEN_SOT;
        let notimestamps_id = S::TOKEN_NOTIMESTAMPS;
        let timestamp_begin = S::TOKEN_TIMESTAMP_BEGIN;
        let mut ids = HashSet::new();

        match mode {
            SuppressTokens::Default => {
                ids.extend(DEFAULT_SUPPRESS_TOKENS.iter().copied());
                insert_if_present(tokenizer, &mut ids, "<|nospeech|>");
            }
            SuppressTokens::None => {}
            SuppressTokens::Ids(extra) => ids.extend(extra.iter().copied()),
        }

        Ok(Self {
            prompt_len,
            notimestamps,
            eot,
            sot,
            notimestamps_id,
            timestamp_begin,
            ids,
        })
    }

    pub fn apply(&self, tokens: &[u32], logits: &mut [f32]) {
        let gen_len = tokens.len();
        if gen_len <= self.prompt_len {
            suppress_id(logits, self.eot);
        }

        for token_id in self.sot..=self.notimestamps_id {
            suppress_id(logits, token_id);
        }

        for &token_id in &self.ids {
            suppress_id(logits, token_id);
        }

        if self.notimestamps {
            for i in (self.notimestamps_id as usize + 1)..logits.len() {
                logits[i] = -1e4;
            }
        }

        // Prevent consecutive timestamps
        if let Some(&last) = tokens.last() {
            if last >= self.timestamp_begin {
                for i in (self.timestamp_begin as usize)..logits.len() {
                    logits[i] = -1e4;
                }
            }
        }

        // Standard repetition penalty for text tokens (not timestamps, not special)
        // We track seen tokens to apply the penalty only once per unique token.
        let mut seen = HashSet::new();
        for &id in tokens.iter().skip(self.prompt_len) {
            if id < self.sot && seen.insert(id) {
                if let Some(logit) = logits.get_mut(id as usize) {
                    if *logit > 0.0 {
                        *logit /= 1.1; // Slightly milder penalty
                    } else {
                        *logit *= 1.1;
                    }
                }
            }
        }
    }
}

fn insert_if_present(tokenizer: &Tokenizer, ids: &mut HashSet<u32>, token: &str) {
    if let Some(id) = tokenizer.token_to_id(token) {
        ids.insert(id);
    }
}

fn suppress_id(logits: &mut [f32], token_id: u32) {
    if let Some(logit) = logits.get_mut(token_id as usize) {
        *logit = -1e4;
    }
}

const DEFAULT_SUPPRESS_TOKENS: &[u32] = &[
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 359,
    503, 522, 542, 873, 893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268,
    3536, 3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331,
    12562, 13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520,
    26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425, 49870, 50254,
    50258, 50358, 50359, 50360, 50361, 50362,
];

#[cfg(test)]
mod tests {
    use super::SuppressTokens;

    #[test]
    fn parses_suppression_modes() {
        assert_eq!(
            SuppressTokens::parse("default").unwrap(),
            SuppressTokens::Default
        );
        assert_eq!(SuppressTokens::parse("none").unwrap(), SuppressTokens::None);
        assert_eq!(
            SuppressTokens::parse("1, 2,3").unwrap(),
            SuppressTokens::Ids(vec![1, 2, 3])
        );
    }
}
