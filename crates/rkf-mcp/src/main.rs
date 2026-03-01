//! rkf-mcp — MCP server for agent-driven interaction with the RKIField SDF engine.
//!
//! Runs MCP protocol over stdio (stdin/stdout). Starts with stub implementations
//! that return "not implemented" for all engine tools. Use the `connect` tool to
//! attach to a running engine process via IPC.

use clap::Parser;
use rkf_core::automation::StubAutomationApi;
use rkf_mcp::protocol::ApiSlot;
use rkf_mcp::registry::ToolMode;
use rkf_mcp::tools;
use std::sync::{Arc, RwLock};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// MCP server mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    /// Full access — all observation and mutation tools
    Editor,
    /// Read-only — observation tools only
    Debug,
}

/// RKIField MCP Server — runs over stdio for Claude Code integration
#[derive(Parser, Debug)]
#[command(name = "rkf-mcp", about = "MCP server for the RKIField SDF engine")]
struct Cli {
    /// Server mode: editor (full access) or debug (read-only)
    #[arg(long, value_enum, default_value = "editor")]
    mode: Mode,
}

/// Auto-discover a running engine and connect to it.
///
/// Scans `/tmp/rkifield-*.sock` for available sockets. If exactly one is found
/// (and it's connectable), swaps the API slot to a live bridge. This lets agents
/// use tools immediately without a manual `connect` call.
fn auto_connect(api_slot: &ApiSlot) {
    let mut sockets = Vec::new();
    if let Ok(entries) = std::fs::read_dir("/tmp") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("rkifield-") && name.ends_with(".sock") {
                sockets.push(entry.path().to_string_lossy().to_string());
            }
        }
    }

    match sockets.len() {
        0 => {
            log::info!("No running engines found — starting with stub API");
        }
        1 => {
            let socket_path = &sockets[0];
            // Verify the socket is connectable (not stale)
            match std::os::unix::net::UnixStream::connect(socket_path) {
                Ok(_) => {
                    let bridge =
                        rkf_mcp::bridge::BridgeAutomationApi::new(socket_path.to_string());
                    if let Ok(mut slot) = api_slot.write() {
                        *slot = Arc::new(bridge);
                        // Read discovery metadata if available
                        let meta_path = socket_path.replace(".sock", ".json");
                        let engine_name = std::fs::read_to_string(&meta_path)
                            .ok()
                            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                            .and_then(|v| v["name"].as_str().map(String::from))
                            .unwrap_or_else(|| "unknown".into());
                        log::info!(
                            "Auto-connected to {engine_name} via {socket_path}"
                        );
                    }
                }
                Err(e) => {
                    log::warn!(
                        "Found socket {socket_path} but connection failed: {e} — starting with stub API"
                    );
                }
            }
        }
        n => {
            log::info!(
                "Found {n} engine sockets — use `connect` tool to choose one: {sockets:?}"
            );
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logging to stderr (stdout is the MCP protocol channel)
    env_logger::Builder::from_default_env()
        .target(env_logger::Target::Stderr)
        .init();

    let cli = Cli::parse();
    let tool_mode = match cli.mode {
        Mode::Editor => ToolMode::Editor,
        Mode::Debug => ToolMode::Debug,
    };

    log::info!("rkf-mcp starting in {:?} mode (stdio transport)", cli.mode);

    // Swappable API — starts with stubs, connect tool can swap to real engine.
    // Uses Arc<RwLock<Arc<dyn AutomationApi>>> so the inner Arc can be cloned
    // and the lock released before tool dispatch (prevents deadlocks with meta tools).
    let api_slot: ApiSlot = Arc::new(RwLock::new(Arc::new(StubAutomationApi)));

    // Build tool registry — meta tools + generic search/use tools live locally.
    // Engine-specific tools are discovered dynamically via search_tools and
    // called via use_tool, so we never need to restart the MCP server when
    // adding new engine functionality.
    let mut registry = rkf_mcp::registry::ToolRegistry::new();
    tools::meta::register_meta_tools(&mut registry, Arc::clone(&api_slot));
    tools::dynamic::register_dynamic_tools(&mut registry);
    let registry = Arc::new(registry);

    // Auto-discover and connect to a running engine
    auto_connect(&api_slot);

    // Run stdio MCP loop
    let stdin = BufReader::new(tokio::io::stdin());
    let mut stdout = tokio::io::stdout();
    let mut lines = stdin.lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let response = match rkf_mcp::protocol::parse_request(&line) {
            Ok(request) => {
                rkf_mcp::protocol::handle_request_locked(&request, &registry, tool_mode, &api_slot)
            }
            Err(err_response) => Some(err_response),
        };

        if let Some(resp) = response {
            let json = serde_json::to_string(&resp)?;
            stdout.write_all(json.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }

    Ok(())
}
