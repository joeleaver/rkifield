//! IPC communication — Unix socket and TCP fallback.
//!
//! The engine opens a Unix socket listener at `/tmp/rkifield-{pid}.sock`.
//! `rkf-mcp` can connect to this socket (or fall back to localhost TCP).
//! Messages are newline-delimited JSON-RPC 2.0.

use crate::protocol::{
    self, InitializeResult, JsonRpcRequest, JsonRpcResponse, McpToolDef, ServerCapabilities,
    ServerInfo, ToolsCallParams, ToolsCapability, ToolsListResult, INVALID_PARAMS,
    METHOD_NOT_FOUND,
};
use crate::registry::ToolMode;
use rkf_core::automation::AutomationApi;
use serde_json::Value;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, UnixListener};

/// Configuration for the IPC server.
#[derive(Debug, Clone)]
pub struct IpcConfig {
    /// Unix socket path (e.g., /tmp/rkifield-{pid}.sock)
    pub socket_path: Option<String>,
    /// TCP port for fallback
    pub tcp_port: u16,
    /// Tool access mode
    pub mode: ToolMode,
}

impl IpcConfig {
    /// Generate default socket path based on current PID.
    pub fn default_socket_path() -> String {
        format!("/tmp/rkifield-{}.sock", std::process::id())
    }
}

/// Run the IPC server, accepting connections on Unix socket or TCP.
///
/// This is the main server loop. Each connection is handled in a separate task.
/// The server runs until the tokio runtime is shut down.
pub async fn run_server(
    config: IpcConfig,
    api: Arc<dyn AutomationApi>,
) -> anyhow::Result<()> {
    if let Some(ref path) = config.socket_path {
        // Clean up stale socket file
        let _ = std::fs::remove_file(path);

        let listener = UnixListener::bind(path)?;
        log::info!("MCP server listening on Unix socket: {path}");

        loop {
            let (stream, _addr) = listener.accept().await?;
            let api = Arc::clone(&api);

            tokio::spawn(async move {
                let (reader, writer) = stream.into_split();
                if let Err(e) = handle_connection(reader, writer, &*api).await {
                    log::error!("Connection error: {e}");
                }
            });
        }
    } else {
        let addr = format!("127.0.0.1:{}", config.tcp_port);
        let listener = TcpListener::bind(&addr).await?;
        log::info!("MCP server listening on TCP: {addr}");

        loop {
            let (stream, peer) = listener.accept().await?;
            log::info!("TCP connection from {peer}");
            let api = Arc::clone(&api);

            tokio::spawn(async move {
                let (reader, writer) = stream.into_split();
                if let Err(e) = handle_connection(reader, writer, &*api).await {
                    log::error!("Connection error from {peer}: {e}");
                }
            });
        }
    }
}

/// Handle a single connection — read newline-delimited JSON-RPC messages and respond.
///
/// Dispatches `initialize`, `tools/list`, and `tools/call` directly through
/// `AutomationApi::list_tools_json` / `call_tool_json`, with no `ToolRegistry`
/// dependency.
async fn handle_connection<R, W>(
    reader: R,
    mut writer: W,
    api: &dyn AutomationApi,
) -> anyhow::Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    let buf_reader = BufReader::new(reader);
    let mut lines = buf_reader.lines();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let response = match protocol::parse_request(&line) {
            Ok(request) => dispatch_ipc_request(&request, api),
            Err(err_response) => Some(err_response),
        };

        if let Some(resp) = response {
            let json = serde_json::to_string(&resp)?;
            writer.write_all(json.as_bytes()).await?;
            writer.write_all(b"\n").await?;
            writer.flush().await?;
        }
    }

    Ok(())
}

/// Dispatch a JSON-RPC request for the IPC server.
///
/// Uses `AutomationApi::list_tools_json` and `call_tool_json` instead of a
/// `ToolRegistry`, so tools are self-describing from the engine implementation.
fn dispatch_ipc_request(
    request: &JsonRpcRequest,
    api: &dyn AutomationApi,
) -> Option<JsonRpcResponse> {
    match request.method.as_str() {
        "initialize" => {
            let result = InitializeResult {
                protocol_version: "2024-11-05".to_string(),
                capabilities: ServerCapabilities {
                    tools: ToolsCapability {
                        list_changed: false,
                    },
                },
                server_info: ServerInfo {
                    name: "rkf-mcp".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                },
            };
            Some(JsonRpcResponse::success(
                request.id.clone(),
                serde_json::to_value(result).unwrap(),
            ))
        }

        "notifications/initialized" => None,

        "tools/list" => {
            let tool_defs: Vec<McpToolDef> = api
                .list_tools_json()
                .ok()
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default();

            let result = ToolsListResult { tools: tool_defs };
            Some(JsonRpcResponse::success(
                request.id.clone(),
                serde_json::to_value(result).unwrap(),
            ))
        }

        "tools/call" => {
            let params = match &request.params {
                Some(p) => p.clone(),
                None => {
                    return Some(JsonRpcResponse::error(
                        request.id.clone(),
                        INVALID_PARAMS,
                        "missing params for tools/call",
                    ));
                }
            };

            let call_params: ToolsCallParams = match serde_json::from_value(params) {
                Ok(p) => p,
                Err(e) => {
                    return Some(JsonRpcResponse::error(
                        request.id.clone(),
                        INVALID_PARAMS,
                        format!("invalid tools/call params: {e}"),
                    ));
                }
            };

            let tool_args = call_params.arguments.unwrap_or(Value::Null);

            match api.call_tool_json(&call_params.name, tool_args) {
                Ok(result) => Some(JsonRpcResponse::success(request.id.clone(), result)),
                Err(e) => {
                    let result = serde_json::json!({
                        "content": [{ "type": "text", "text": format!("Error: {e}") }],
                        "isError": true
                    });
                    Some(JsonRpcResponse::success(request.id.clone(), result))
                }
            }
        }

        _ => Some(JsonRpcResponse::error(
            request.id.clone(),
            METHOD_NOT_FOUND,
            format!("method not found: {}", request.method),
        )),
    }
}

/// Connect to an engine's IPC socket as a client.
///
/// Returns a client that can send JSON-RPC requests and receive responses.
pub struct IpcClient {
    reader: BufReader<tokio::io::ReadHalf<tokio::net::UnixStream>>,
    writer: tokio::io::WriteHalf<tokio::net::UnixStream>,
}

impl IpcClient {
    /// Connect to a Unix socket.
    pub async fn connect_unix(path: &str) -> anyhow::Result<Self> {
        let stream = tokio::net::UnixStream::connect(path).await?;
        let (reader, writer) = tokio::io::split(stream);
        Ok(Self {
            reader: BufReader::new(reader),
            writer,
        })
    }

    /// Send a JSON-RPC request and receive the response.
    pub async fn request(&mut self, req: &JsonRpcRequest) -> anyhow::Result<JsonRpcResponse> {
        let json = serde_json::to_string(req)?;
        self.writer.write_all(json.as_bytes()).await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;

        let mut line = String::new();
        self.reader.read_line(&mut line).await?;
        let resp: JsonRpcResponse = serde_json::from_str(line.trim())?;
        Ok(resp)
    }

    /// Send a notification (no response expected).
    pub async fn notify(&mut self, req: &JsonRpcRequest) -> anyhow::Result<()> {
        let json = serde_json::to_string(req)?;
        self.writer.write_all(json.as_bytes()).await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;
        Ok(())
    }
}

/// TCP client variant.
pub struct TcpClient {
    reader: BufReader<tokio::io::ReadHalf<tokio::net::TcpStream>>,
    writer: tokio::io::WriteHalf<tokio::net::TcpStream>,
}

impl TcpClient {
    /// Connect to a TCP address.
    pub async fn connect(addr: &str) -> anyhow::Result<Self> {
        let stream = tokio::net::TcpStream::connect(addr).await?;
        let (reader, writer) = tokio::io::split(stream);
        Ok(Self {
            reader: BufReader::new(reader),
            writer,
        })
    }

    /// Send a JSON-RPC request and receive the response.
    pub async fn request(&mut self, req: &JsonRpcRequest) -> anyhow::Result<JsonRpcResponse> {
        let json = serde_json::to_string(req)?;
        self.writer.write_all(json.as_bytes()).await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;

        let mut line = String::new();
        self.reader.read_line(&mut line).await?;
        let resp: JsonRpcResponse = serde_json::from_str(line.trim())?;
        Ok(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ToolMode;
    use rkf_core::automation::StubAutomationApi;

    #[tokio::test]
    async fn unix_socket_roundtrip() {
        let socket_path = format!("/tmp/rkf-test-{}.sock", std::process::id());
        let _ = std::fs::remove_file(&socket_path);

        let api: Arc<dyn AutomationApi> = Arc::new(StubAutomationApi);

        // Start server in background
        let server_api = Arc::clone(&api);
        let server_path = socket_path.clone();
        let server_handle = tokio::spawn(async move {
            let _ = run_server(
                IpcConfig {
                    socket_path: Some(server_path),
                    tcp_port: 0,
                    mode: ToolMode::Editor,
                },
                server_api,
            )
            .await;
        });

        // Give server time to bind
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Connect client
        let mut client = IpcClient::connect_unix(&socket_path).await.unwrap();

        // Send initialize
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(1.into())),
            method: "initialize".to_string(),
            params: Some(serde_json::json!({})),
        };
        let resp = client.request(&req).await.unwrap();
        assert!(resp.error.is_none());
        assert!(resp.result.is_some());

        // Send tools/list — StubAutomationApi returns empty array (default)
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(2.into())),
            method: "tools/list".to_string(),
            params: None,
        };
        let resp = client.request(&req).await.unwrap();
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 0); // StubAutomationApi has no tools

        // Send tools/call — should return isError since stub doesn't implement tools
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(3.into())),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({"name": "screenshot", "arguments": {}})),
        };
        let resp = client.request(&req).await.unwrap();
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);

        // Cleanup
        server_handle.abort();
        let _ = std::fs::remove_file(&socket_path);
    }

    #[tokio::test]
    async fn tcp_fallback_roundtrip() {
        let api: Arc<dyn AutomationApi> = Arc::new(StubAutomationApi);

        // Use port 0 to get OS-assigned port, but we need to pick one
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let server_api = Arc::clone(&api);
        let server_handle = tokio::spawn(async move {
            let _ = run_server(
                IpcConfig {
                    socket_path: None,
                    tcp_port: port,
                    mode: ToolMode::Debug,
                },
                server_api,
            )
            .await;
        });

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let mut client = TcpClient::connect(&format!("127.0.0.1:{port}")).await.unwrap();

        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(Value::Number(1.into())),
            method: "tools/list".to_string(),
            params: None,
        };
        let resp = client.request(&req).await.unwrap();
        assert!(resp.error.is_none());

        server_handle.abort();
    }
}
