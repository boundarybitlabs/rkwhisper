# Daemon Concurrency Plan

`rkwhisperd` currently accepts multiple Unix socket clients, but it processes
them serially. The main accept loop calls `handle_connection` directly with a
mutable reference to the shared model pools, so a long live stream prevents the
daemon from accepting and serving the next client until the stream finishes.

This document describes the plan for supporting multiple clients safely while
keeping RKNN/NPU contention explicit.

## Goals

- Accept multiple client connections concurrently.
- Keep each client protocol session independent: Protobuf control messages,
  one shared-memory PCM ring per client, and Protobuf responses back to that
  client.
- Prevent unbounded work from piling up in memory.
- Preserve ordered segment responses within each client stream.
- Make NPU sharing deliberate instead of allowing many clients to run model
  inference at once accidentally.
- Return clear busy or queue-full errors when the daemon is saturated.

## Non-Goals

- Do not attempt fully parallel inference for unlimited clients.
- Do not share one memfd ring between clients.
- Do not make the Python client responsible for daemon-side scheduling.
- Do not optimize for multi-process daemon deployments until single-process
  scheduling is explicit and measured.

## Current Limitation

The daemon owns a single `DaemonPools` value:

- `main` loads all enabled model pools once at startup.
- The accept loop receives a `UnixStream`.
- `handle_connection` borrows `DaemonPools` mutably.
- The selected `ModelPool` is borrowed mutably for the lifetime of the
  transcription.

That shape guarantees exclusive access to model pools, but it also means only
one client can make progress at a time.

## Recommended Architecture

Use one accept thread plus one scheduler per model.

The accept thread should:

1. Accept each Unix socket connection promptly.
2. Spawn a lightweight session thread for protocol I/O.
3. Read and validate `ClientHello`.
4. Create the per-client memfd ring and send it with `SCM_RIGHTS`.
5. Convert socket data-ready signals into `LiveWindow` messages.
6. Submit a transcription job to the selected model scheduler.
7. Stream scheduler output back to that client as Protobuf responses.

Each model scheduler should own its model pool:

1. A bounded job channel accepts client transcription jobs for that model.
2. The scheduler processes jobs one at a time using its `ParallelTranscriberPool`.
3. Each job carries a `LiveWindow` receiver and a segment response sender.
4. When the job finishes, the scheduler sends a final `done` or `error`.

This gives the daemon concurrent clients without concurrent access to the same
RKNN model pool. It also preserves the current safety property that only one
transcription uses a given model pool at a time.

## Data Flow

Per client:

1. Client connects to the daemon socket.
2. Client sends `ClientHello`.
3. Session thread validates the model and audio format.
4. Session thread creates a memfd-backed shared-memory ring.
5. Daemon sends the memfd plus `ServerHello`.
6. Client writes PCM into its ring and sends one-byte data-ready signals.
7. Session reader drains PCM into 30-second `LiveWindow` items.
8. Session submits a job to the selected model scheduler.
9. Scheduler runs inference and sends segment/done/error messages.
10. Session writer serializes those messages back to the client.

The shared-memory ring remains per-client. The scheduler only sees normalized
window messages and does not need to know about sockets or file descriptors.

## Backpressure

Use bounded channels at every queue boundary:

- Accept backlog: controlled by the Unix listener and OS socket backlog.
- Per-model job queue: small, configurable, for example 1 to 4 pending jobs.
- Per-client window queue: small, currently 4 `LiveWindow` messages.
- Per-client response queue: bounded to prevent a slow reader from consuming
  unbounded memory.

When the selected model queue is full, the session should return a Protobuf
`error` response such as `model queue full` and close the connection.

When the client writes audio faster than the daemon can consume it, the ring
buffer naturally overwrites old audio if the writer gets too far ahead. The
protocol should either document this as drop-oldest behavior or add explicit
overrun detection by comparing write and read offsets before draining.

## Cancellation

Cancellation should remain client-local:

- `SIGNAL_CANCEL` closes the client's `LiveWindow` stream.
- The scheduler should stop the active job after the current inference boundary.
- The session should return either `done` with partial timing stats or `error`
  with a cancellation message. The exact response should be standardized before
  exposing this as stable protocol behavior.

If the client disconnects, the session should drop the job's window sender and
response receiver. The scheduler should treat that as cancellation.

## Error Handling

Errors should be Protobuf responses, not stderr-only events.

Recommended errors:

- unsupported protocol version
- unsupported audio format
- unknown model
- model queue full
- shared-memory setup failed
- client disconnected
- stream cancelled
- inference failed

The daemon should still log errors with connection/model context, but clients
need structured responses for recovery.

## Implementation Steps

1. Introduce `ModelSchedulers`.
   - Replace `DaemonPools` in the accept path with a map from model ID to
     bounded job sender.
   - Move each loaded `ModelPool` into its own scheduler thread.

2. Define a job type.
   - Include `RequestHeader`.
   - Include `mpsc::Receiver<Result<LiveWindow>>`.
   - Include a bounded response sender for `Response`.

3. Split session I/O from inference.
   - Keep socket negotiation and shared-memory ring handling in the session
     thread.
   - Keep model inference only inside scheduler threads.

4. Add queue-full behavior.
   - Use nonblocking `try_send` or a short timeout when submitting jobs.
   - Return a Protobuf `error` if the selected model is saturated.

5. Add client disconnect handling.
   - Treat read failure, write failure, and dropped channels as cancellation.
   - Ensure the reader thread exits and the memfd mapping is released.

6. Add tests.
   - Unit-test scheduler queue-full behavior.
   - Unit-test unknown model errors.
   - Add an integration-style daemon test with two clients where the second
     connection receives either queued service or a structured busy error.

7. Document operational limits.
   - Add config for per-model queue depth.
   - Document that each model is processed serially unless multiple model pool
     replicas are configured in the future.

## Future Extension: Pool Replicas

After the single-scheduler design is stable, the daemon can support multiple
replicas per model:

- `model_workers = N` in config.
- Start `N` scheduler workers for that model.
- Dispatch jobs round-robin or by least queued work.

This should only be enabled after measuring RKNN runtime behavior under
concurrent model contexts. On RK3588-class devices, uncontrolled parallel
inference may reduce throughput or increase tail latency.

## Open Questions

- Should a queued live stream wait for scheduler capacity, or should live
  streams fail fast when the model is busy?
- Should cancellation return `done` with partial stats or a distinct
  cancellation response?
- Should ring overruns be detected and reported as protocol errors?
- Should the daemon support separate limits for batch-style finite streams and
  long-running live streams?
