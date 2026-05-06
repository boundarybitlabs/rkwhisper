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

    let (mut sender, mut receiver) = session.split();

    // 1. Start audio sender task
    let sender_handle = tokio::spawn(async move {
        println!("Sender: Sending 1 second of silence...");
        let silence = vec![0.0f32; 16000]; // 1 second
        let pcm = rkwhisper_client::samples_to_pcm(&silence);
        sender.send_audio(&pcm).await?;
        sender.finish().await?;
        println!("Sender: Finished.");
        Ok::<(), anyhow::Error>(())
    });

    // 2. Process responses in the main task
    println!("Receiver: Waiting for responses...");
    while let Ok(response) = receiver.recv_response().await {
        println!("Received: {:?}", response);
        if matches!(response, rkwhisper_protocol::Response::Done { .. }) {
            break;
        }
    }

    sender_handle.await??;

    Ok(())
}
