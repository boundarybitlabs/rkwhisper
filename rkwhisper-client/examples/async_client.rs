use rkwhisper_client::{ClientHello, asynchronous::Session};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let socket_path =
        env::var("RKWHISPER_SOCKET").unwrap_or_else(|_| "/run/rkwhisper/asr.sock".to_string());
    let model =
        env::var("RKWHISPER_TEST_MODEL").unwrap_or_else(|_| "whisper-small-30s".to_string());

    println!("Connecting to {} with model {}...", socket_path, model);

    let hello = ClientHello {
        model,
        client_id: "rust-async-example".to_string(),
        ..ClientHello::default()
    };

    let mut session = Session::connect(socket_path, hello).await?;
    println!("Connected! Handshake successful.");

    // Just receive one response to verify it works
    println!("Waiting for any response (send some audio to the socket if needed)...");

    // In a real example we would send audio, but here we just prove the plumbing.
    // If we send nothing, it might just block or the daemon might time out.

    // Let's send a tiny bit of silence to get a response
    let silence = vec![0.0f32; 16000]; // 1 second
    let pcm = rkwhisper_client::samples_to_pcm(&silence);
    session.send_audio(&pcm).await?;
    session.finish().await?;

    while let Ok(response) = session.recv_response().await {
        println!("Received: {:?}", response);
        if matches!(response, rkwhisper_protocol::Response::Done { .. }) {
            break;
        }
    }

    Ok(())
}
