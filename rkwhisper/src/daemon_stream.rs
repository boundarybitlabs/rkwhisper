use anyhow::{Context, Result, bail};
use std::io::Read;
use std::os::unix::net::UnixStream;
use tokio::sync::mpsc;

use crate::{
    daemon::pcm_s16le_to_f32,
    protocol::{SIGNAL_CANCEL, SIGNAL_DATA_READY, SIGNAL_END_OF_STREAM, SharedAudioRing},
};

pub struct LiveChunk {
    pub samples: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StreamReadStats {
    pub total_samples: usize,
    pub total_windows: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum ReadOutcome {
    Completed(StreamReadStats),
    Cancelled(StreamReadStats),
}

impl ReadOutcome {
    pub fn stats(self) -> StreamReadStats {
        match self {
            Self::Completed(stats) | Self::Cancelled(stats) => stats,
        }
    }
}

pub fn read_live_chunks(
    mut stream: UnixStream,
    ring: SharedAudioRing,
    chunk_tx: mpsc::Sender<Result<LiveChunk>>,
) -> Result<ReadOutcome> {
    let mut pcm = Vec::<u8>::new();
    let mut stats = StreamReadStats::default();

    loop {
        let mut signal = [0u8; 1];
        let n = stream
            .read(&mut signal)
            .context("failed to read shared-memory signal")?;
        if n == 0 {
            flush_pcm_chunks(&chunk_tx, &mut pcm, &mut stats, true)?;
            break;
        }

        match signal[0] {
            SIGNAL_DATA_READY => {
                ring.drain_available(&mut pcm)?;
                flush_pcm_chunks(&chunk_tx, &mut pcm, &mut stats, false)?;
            }
            SIGNAL_END_OF_STREAM => {
                ring.drain_available(&mut pcm)?;
                flush_pcm_chunks(&chunk_tx, &mut pcm, &mut stats, true)?;
                break;
            }
            SIGNAL_CANCEL => return Ok(ReadOutcome::Cancelled(stats)),
            other => {
                let message = format!("unsupported shared-memory signal {other}");
                let _ = chunk_tx.blocking_send(Err(anyhow::anyhow!(message.clone())));
                bail!("{message}");
            }
        }
    }

    Ok(ReadOutcome::Completed(stats))
}

fn flush_pcm_chunks(
    chunk_tx: &mpsc::Sender<Result<LiveChunk>>,
    pcm: &mut Vec<u8>,
    stats: &mut StreamReadStats,
    final_flush: bool,
) -> Result<()> {
    // Match the default VAD window: 512 samples, 2 bytes per sample.
    let chunk_bytes = 1024;
    while pcm.len() >= chunk_bytes {
        let chunk = pcm.drain(..chunk_bytes).collect::<Vec<_>>();
        let samples = pcm_s16le_to_f32(&chunk)?;
        stats.total_samples += samples.len();
        chunk_tx
            .blocking_send(Ok(LiveChunk { samples }))
            .map_err(|_| anyhow::anyhow!("live stream scheduler stopped"))?;
        stats.total_windows += 1;
    }

    if final_flush && !pcm.is_empty() {
        let chunk = std::mem::take(pcm);
        let samples = pcm_s16le_to_f32(&chunk)?;
        stats.total_samples += samples.len();
        chunk_tx
            .blocking_send(Ok(LiveChunk { samples }))
            .map_err(|_| anyhow::anyhow!("live stream scheduler stopped"))?;
        stats.total_windows += 1;
    }
    Ok(())
}
