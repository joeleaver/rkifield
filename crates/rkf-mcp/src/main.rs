//! # rkf-mcp
//!
//! MCP server binary for agent-driven interaction with the RKIField SDF engine.
//!
//! Connects to a running engine process (editor, game, or testbed) via IPC
//! and exposes engine functionality through MCP tools over JSON-RPC 2.0.

use clap::Parser;

/// MCP server mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
pub enum Mode {
    /// Full access — all observation and mutation tools
    Editor,
    /// Read-only — observation tools only
    Debug,
}

/// RKIField MCP Server
#[derive(Parser, Debug)]
#[command(name = "rkf-mcp", about = "MCP server for the RKIField SDF engine")]
struct Cli {
    /// Server mode: editor (full access) or debug (read-only)
    #[arg(long, value_enum, default_value = "editor")]
    mode: Mode,

    /// Unix socket path to connect to the engine
    #[arg(long)]
    connect: Option<String>,

    /// TCP port fallback (used when Unix socket is unavailable)
    #[arg(long, default_value = "9100")]
    port: u16,
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();
    log::info!("rkf-mcp starting in {:?} mode", cli.mode);

    if let Some(ref path) = cli.connect {
        log::info!("Connecting to engine at: {}", path);
    } else {
        log::info!("No --connect path specified, will use TCP port {}", cli.port);
    }

    // TODO: Phase 1 tasks 1.3-1.8 will wire up:
    // - Tool registry
    // - MCP protocol (JSON-RPC 2.0)
    // - IPC connection to engine
    // - Built-in tool registration

    println!("rkf-mcp: server ready (mode={:?})", cli.mode);
}
