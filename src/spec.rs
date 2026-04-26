pub trait WhisperSpec {
    // Mel frontend
    const MEL_BINS: usize; // 80 for V2, 128 for V3
    const FRAMES: usize; // 3000 (30 sec)

    // Encoder output
    const ENC_SEQ: usize; // 1500 (downsample x2)
    const HIDDEN: usize; // encoder hidden dim

    // Decoder architecture
    const N_LAYERS: usize;
    const N_HEADS: usize;
    const D_HEAD: usize;
    const T_CACHE: usize; // self-attention KV cache length

    // Tokens
    const VOCAB: usize;
    const TOKEN_EOT: u32;
}

pub struct WhisperTiny;

impl WhisperSpec for WhisperTiny {
    const MEL_BINS: usize = 80;
    const FRAMES: usize = 3000;

    const ENC_SEQ: usize = 1500;
    const HIDDEN: usize = 384;

    const N_LAYERS: usize = 4;
    const N_HEADS: usize = 6;
    const D_HEAD: usize = 64;
    const T_CACHE: usize = 448;

    const VOCAB: usize = 51865;
    const TOKEN_EOT: u32 = 50257;
}

pub struct WhisperBase;

impl WhisperSpec for WhisperBase {
    const MEL_BINS: usize = 80;
    const FRAMES: usize = 3000;

    const ENC_SEQ: usize = 1500;
    const HIDDEN: usize = 512;

    const N_LAYERS: usize = 6;
    const N_HEADS: usize = 8;
    const D_HEAD: usize = 64;
    const T_CACHE: usize = 448;

    const VOCAB: usize = 51865;
    const TOKEN_EOT: u32 = 50257;
}

pub struct WhisperLargeV3Turbo;

impl WhisperSpec for WhisperLargeV3Turbo {
    const MEL_BINS: usize = 128;
    const FRAMES: usize = 3000;

    const ENC_SEQ: usize = 1500;
    const HIDDEN: usize = 1280;

    const N_LAYERS: usize = 4;
    const N_HEADS: usize = 20;
    const D_HEAD: usize = 64;
    const T_CACHE: usize = 400;

    const VOCAB: usize = 51866;
    const TOKEN_EOT: u32 = 50257;
}

pub struct WhisperMedium;

impl WhisperSpec for WhisperMedium {
    const MEL_BINS: usize = 80;
    const FRAMES: usize = 3000;

    const ENC_SEQ: usize = 1500;
    const HIDDEN: usize = 1024;

    const N_LAYERS: usize = 24;
    const N_HEADS: usize = 16;
    const D_HEAD: usize = 64;
    const T_CACHE: usize = 448;

    const VOCAB: usize = 51865;
    const TOKEN_EOT: u32 = 50257;
}

pub struct WhisperSmall;

impl WhisperSpec for WhisperSmall {
    const MEL_BINS: usize = 80;
    const FRAMES: usize = 3000;

    const ENC_SEQ: usize = 1500;
    const HIDDEN: usize = 768;

    const N_LAYERS: usize = 12;
    const N_HEADS: usize = 12;
    const D_HEAD: usize = 64;
    const T_CACHE: usize = 448;

    const VOCAB: usize = 51865;
    const TOKEN_EOT: u32 = 50257;
}
