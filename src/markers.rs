use std::marker::PhantomData;

// ----- Units & invariants -----
pub struct Hz<const N: u32>;
pub struct MelBands<const M: usize>;
pub struct Frames<const T: usize>;
pub struct Sec<const S: u32>;
pub struct Mono;

pub struct SampleRate<const N: u32>(PhantomData<Hz<N>>);
pub struct Channels<C>(PhantomData<C>);

pub type SR16K = SampleRate<16_000>;

pub struct Audio<const S: u32, C> {
    pub pcm: Vec<f32>,
    _sr: SampleRate<S>,
    _ch: Channels<C>,
}
impl<const S: u32, C> Audio<S, C> {
    pub fn new(pcm: Vec<f32>) -> Self {
        Self {
            pcm,
            _sr: SampleRate(PhantomData),
            _ch: Channels(PhantomData),
        }
    }
}

pub struct LogMel<const M: usize, const T: usize> {
    // shape [M, T], row-major (mel, frames)
    pub data: Vec<f32>,
    _mel: PhantomData<MelBands<M>>,
    _frames: PhantomData<Frames<T>>,
}

pub struct EncoderIn<const M: usize, const T: usize>(pub LogMel<M, T>); // enforce M=80, T=3000
pub struct EncoderOut<const S: usize, const H: usize> {
    // S=1500, H=1280 for large-v3
    // [S, H]
    pub data: Vec<f32>,
}

pub struct Logits<const S: usize, const VOCAB_SIZE: usize> {
    // [S, VOCAB_SIZE]
    pub data: Vec<f32>,
}
