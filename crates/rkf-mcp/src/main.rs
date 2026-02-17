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

    // Build tool registry
    let mut registry = rkf_mcp::registry::ToolRegistry::new();
    tools::observation::register_observation_tools(&mut registry);
    tools::meta::register_meta_tools(&mut registry, Arc::clone(&api_slot));
    let registry = Arc::new(registry);

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
