//! Rinch debug protocol client — captures full-window screenshots via TCP.
//!
//! When rinch runs with `features = ["debug"]`, it starts a TCP server on
//! localhost. We discover the port from `~/.rinch/debug/{pid}.json` and
//! issue a `screenshot` command to get the composited window pixels.

use base64::Engine;
use rkf_core::automation::{AutomationError, AutomationResult};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;

/// Discover the rinch debug TCP port for this process.
fn discover_debug_port() -> Option<u16> {
    let home = std::env::var("HOME").ok()?;
    let path = PathBuf::from(home)
        .join(".rinch")
        .join("debug")
        .join(format!("{}.json", std::process::id()));
    let data = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
    json.get("port")?.as_u64().map(|p| p as u16)
}

/// Write a length-prefixed frame (4-byte big-endian length + payload).
fn write_frame(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
    let len = data.len() as u32;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(data)?;
    stream.flush()
}

/// Read a length-prefixed frame.
fn read_frame(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > 64 * 1024 * 1024 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame too large",
        ));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

/// Capture the full composited window via the rinch debug protocol.
///
/// Returns PNG-encoded bytes.
pub fn capture_window_screenshot() -> AutomationResult<Vec<u8>> {
    let port = discover_debug_port().ok_or_else(|| {
        AutomationError::EngineError(
            "rinch debug server not found — is the 'debug' feature enabled?".into(),
        )
    })?;

    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .map_err(|e| AutomationError::EngineError(format!("connect to rinch debug: {e}")))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .ok();

    // Handshake
    let handshake = serde_json::json!({
        "protocol": "rinch-debug",
        "version": 1
    });
    write_frame(
        &mut stream,
        serde_json::to_vec(&handshake).unwrap().as_slice(),
    )
    .map_err(|e| AutomationError::EngineError(format!("handshake send: {e}")))?;

    let resp_data = read_frame(&mut stream)
        .map_err(|e| AutomationError::EngineError(format!("handshake recv: {e}")))?;
    let _resp: serde_json::Value = serde_json::from_slice(&resp_data)
        .map_err(|e| AutomationError::EngineError(format!("handshake parse: {e}")))?;

    // Send screenshot request
    let request = serde_json::json!({
        "id": 1,
        "method": "screenshot"
    });
    write_frame(
        &mut stream,
        serde_json::to_vec(&request).unwrap().as_slice(),
    )
    .map_err(|e| AutomationError::EngineError(format!("screenshot send: {e}")))?;

    // Read response
    let result_data = read_frame(&mut stream)
        .map_err(|e| AutomationError::EngineError(format!("screenshot recv: {e}")))?;
    let result: serde_json::Value = serde_json::from_slice(&result_data)
        .map_err(|e| AutomationError::EngineError(format!("screenshot parse: {e}")))?;

    // Extract base64 PNG data
    let result_type = result
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("error");
    match result_type {
        "bytes" => {
            let b64 = result
                .get("data")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    AutomationError::EngineError("missing data in screenshot response".into())
                })?;
            base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| AutomationError::EngineError(format!("base64 decode: {e}")))
        }
        "error" => {
            let msg = result
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            Err(AutomationError::EngineError(format!(
                "rinch screenshot failed: {msg}"
            )))
        }
        _ => Err(AutomationError::EngineError(format!(
            "unexpected response type: {result_type}"
        ))),
    }
}
