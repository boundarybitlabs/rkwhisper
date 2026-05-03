use pyo3::prelude::*;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyRuntimeError;
use rkwhisper_protocol::{ClientHello, SAMPLE_RATE};

#[pyclass(name = "AudioFormat")]
#[derive(Clone)]
pub struct PyAudioFormat {
    #[pyo3(get, set)]
    pub sample_rate: u32,
    #[pyo3(get, set)]
    pub channels: u32,
    #[pyo3(get, set)]
    pub sample_format: i32,
}

#[pymethods]
impl PyAudioFormat {
    #[new]
    #[pyo3(signature = (sample_rate=SAMPLE_RATE, channels=1, sample_format=1))]
    fn new(sample_rate: u32, channels: u32, sample_format: i32) -> Self {
        Self {
            sample_rate,
            channels,
            sample_format,
        }
    }
}

#[pyclass(name = "VadOptions")]
#[derive(Clone)]
pub struct PyVadOptions {
    #[pyo3(get, set)]
    pub threshold: Option<f32>,
    #[pyo3(get, set)]
    pub min_speech_ms: Option<u32>,
    #[pyo3(get, set)]
    pub min_silence_ms: Option<u32>,
    #[pyo3(get, set)]
    pub speech_pad_ms: Option<u32>,
    #[pyo3(get, set)]
    pub window_samples: Option<usize>,
}

#[pymethods]
impl PyVadOptions {
    #[new]
    #[pyo3(signature = (threshold=None, min_speech_ms=None, min_silence_ms=None, speech_pad_ms=None, window_samples=None))]
    fn new(
        threshold: Option<f32>,
        min_speech_ms: Option<u32>,
        min_silence_ms: Option<u32>,
        speech_pad_ms: Option<u32>,
        window_samples: Option<usize>,
    ) -> Self {
        Self {
            threshold,
            min_speech_ms,
            min_silence_ms,
            speech_pad_ms,
            window_samples,
        }
    }
}

#[pyclass(name = "ClientHello")]
pub struct PyClientHello {
    #[pyo3(get, set)]
    pub model: String,
    #[pyo3(get, set)]
    pub mode: String,
    #[pyo3(get, set)]
    pub lang: String,
    #[pyo3(get, set)]
    pub task: String,
    #[pyo3(get, set)]
    pub max_new_tokens: usize,
    #[pyo3(get, set)]
    pub beam_size: usize,
    #[pyo3(get, set)]
    pub notimestamps: bool,
    #[pyo3(get, set)]
    pub suppress_tokens: String,
    #[pyo3(get, set)]
    pub audio_format: PyAudioFormat,
    #[pyo3(get, set)]
    pub vad: PyVadOptions,
    #[pyo3(get, set)]
    pub client_id: String,
}

#[pymethods]
impl PyClientHello {
    #[new]
    #[pyo3(signature = (model, mode="batch".to_string(), lang="en".to_string(), task="transcribe".to_string(), max_new_tokens=128, beam_size=5, notimestamps=false, suppress_tokens="default".to_string(), audio_format=None, vad=None, client_id="".to_string()))]
    fn new(
        model: String,
        mode: String,
        lang: String,
        task: String,
        max_new_tokens: usize,
        beam_size: usize,
        notimestamps: bool,
        suppress_tokens: String,
        audio_format: Option<PyAudioFormat>,
        vad: Option<PyVadOptions>,
        client_id: String,
    ) -> Self {
        Self {
            model,
            mode,
            lang,
            task,
            max_new_tokens,
            beam_size,
            notimestamps,
            suppress_tokens,
            audio_format: audio_format.unwrap_or_else(|| PyAudioFormat::new(SAMPLE_RATE, 1, 1)),
            vad: vad.unwrap_or_else(|| PyVadOptions::new(None, None, None, None, None)),
            client_id,
        }
    }
}

#[pyclass(name = "Segment")]
pub struct PySegment {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub begin: f32,
    #[pyo3(get)]
    pub end: f32,
}

#[pyclass(name = "SpeechStarted")]
pub struct PySpeechStarted {
    #[pyo3(get)]
    pub begin: f32,
}

#[pyclass(name = "SpeechEnded")]
pub struct PySpeechEnded {
    #[pyo3(get)]
    pub end: f32,
}

#[pyclass(name = "Done")]
pub struct PyDone {
    #[pyo3(get)]
    pub audio_s: f32,
    #[pyo3(get)]
    pub rtf: f32,
}

#[pyclass]
pub struct SyncSession {
    inner: client::sync::Session,
}

#[pymethods]
impl SyncSession {
    #[staticmethod]
    fn connect(socket_path: String, hello: &PyClientHello) -> PyResult<Self> {
        let hello_internal = ClientHello {
            model: hello.model.clone(),
            mode: hello.mode.clone(),
            lang: hello.lang.clone(),
            task: hello.task.clone(),
            max_new_tokens: hello.max_new_tokens,
            beam_size: hello.beam_size,
            notimestamps: hello.notimestamps,
            suppress_tokens: hello.suppress_tokens.clone(),
            audio_format: rkwhisper_protocol::AudioFormat {
                sample_rate: hello.audio_format.sample_rate,
                channels: hello.audio_format.channels,
                sample_format: hello.audio_format.sample_format,
            },
            vad: rkwhisper_protocol::VadOptions {
                threshold: hello.vad.threshold,
                min_speech_ms: hello.vad.min_speech_ms,
                min_silence_ms: hello.vad.min_silence_ms,
                speech_pad_ms: hello.vad.speech_pad_ms,
                window_samples: hello.vad.window_samples,
            },
            client_id: hello.client_id.clone(),
        };

        let session = client::sync::Session::connect(socket_path, hello_internal)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self { inner: session })
    }

    fn send_audio(&mut self, pcm: Vec<u8>) -> PyResult<()> {
        self.inner.send_audio(&pcm).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn finish(&mut self) -> PyResult<()> {
        self.inner.finish().map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn cancel(&mut self) -> PyResult<()> {
        self.inner.cancel().map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: PyObject,
        _exc_value: PyObject,
        _traceback: PyObject,
    ) -> PyResult<()> {
        Ok(())
    }

    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyObject>> {
        match self.recv_response() {
            Ok(Some(obj)) => {
                let py = unsafe { Python::assume_gil_acquired() };
                // We stop iteration when we get PyDone
                if obj.bind(py).is_instance_of::<PyDone>() {
                    return Ok(None);
                }
                Ok(Some(obj))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn recv_response(&mut self) -> PyResult<Option<PyObject>> {
        let py = unsafe { Python::assume_gil_acquired() };
        match self.inner.recv_response() {
            Ok(rkwhisper_protocol::Response::Segment { text, begin, end }) => {
                Ok(Some(PySegment { text, begin, end }.into_py_any(py)?))
            }
            Ok(rkwhisper_protocol::Response::SpeechStarted { begin }) => {
                Ok(Some(PySpeechStarted { begin }.into_py_any(py)?))
            }
            Ok(rkwhisper_protocol::Response::SpeechEnded { end }) => {
                Ok(Some(PySpeechEnded { end }.into_py_any(py)?))
            }
            Ok(rkwhisper_protocol::Response::Done { audio_s, rtf }) => {
                Ok(Some(PyDone { audio_s, rtf }.into_py_any(py)?))
            }
            Ok(rkwhisper_protocol::Response::Error { error }) => Err(PyRuntimeError::new_err(error)),
            Ok(rkwhisper_protocol::Response::ServerHello(_)) => {
                Err(PyRuntimeError::new_err("unexpected server hello"))
            }
            Ok(rkwhisper_protocol::Response::Cancelled { .. }) => {
                Err(PyRuntimeError::new_err("session cancelled"))
            }
            Ok(rkwhisper_protocol::Response::BackOff { reason, .. }) => {
                Err(PyRuntimeError::new_err(format!("server backoff: {reason}")))
            }
            Err(e) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }
}

#[pymodule]
fn rkwhisper_client(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAudioFormat>()?;
    m.add_class::<PyVadOptions>()?;
    m.add_class::<PyClientHello>()?;
    m.add_class::<PySegment>()?;
    m.add_class::<PySpeechStarted>()?;
    m.add_class::<PySpeechEnded>()?;
    m.add_class::<PyDone>()?;
    m.add_class::<SyncSession>()?;
    Ok(())
}
