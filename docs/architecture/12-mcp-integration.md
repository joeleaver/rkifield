# MCP Integration — Agent-Native Engine

> **Status: DECIDED**

RKIField is designed to be operated by LLMs and AI agents as a first-class use case. The engine exposes its full functionality through the Model Context Protocol (MCP), enabling autonomous development, testing, content creation, and debugging.

### Decision: Dedicated `rkf-mcp` Crate — Generic MCP Server with Tool Discovery

**Chosen over:** Embedding MCP directly in the editor (limits agent access to editor-only mode), per-tool hardcoded servers (doesn't scale, no discovery).

A standalone binary (`rkf-mcp`) implements an MCP server that connects to any running RKIField process (editor, game, or testbed). Tools are registered dynamically via a discovery system — adding a new tool requires implementing a trait and registering it, not modifying the MCP server itself.

```
┌─────────────────┐       IPC (Unix socket / named pipe)       ┌──────────────────┐
│  AI Agent        │◄─────────────────────────────────────────►│  rkf-mcp server   │
│  (Claude, etc.)  │          MCP Protocol (JSON-RPC)          │                    │
└─────────────────┘                                            │  ┌──────────────┐ │
                                                               │  │ Tool Registry │ │
                                                               │  └──────┬───────┘ │
                                                               │         │         │
                                                               └─────────┼─────────┘
                                                                         │ Automation API
                                                                         ▼
                                                               ┌──────────────────┐
                                                               │  RKIField Engine  │
                                                               │  (editor / game / │
                                                               │   testbed)        │
                                                               └──────────────────┘
```

**Two deployment modes:**

| Mode | Binary | Tools Available | Use Case |
|------|--------|----------------|----------|
| **Editor MCP** | `rkf-mcp --mode editor` | All tools (full scene authoring) | Agent-driven content creation, automated editing |
| **Debug MCP** | `rkf-mcp --mode debug` | Read-only + screenshots + logs | Testing, CI validation, autonomous QA |

Both modes connect to the same engine process. The mode determines which tool subset is exposed.

### Decision: Automation API — Engine-Side Trait System

The engine exposes an `AutomationApi` that the MCP server calls. This is a Rust trait implemented by `rkf-runtime`, providing a clean boundary between engine internals and external tooling.

```rust
/// The engine's automation surface — called by rkf-mcp
trait AutomationApi: Send + Sync {
    // --- Observation ---
    fn screenshot(&self, width: u32, height: u32) -> Result<RgbaImage>;
    fn scene_graph(&self) -> Result<SceneGraphSnapshot>;
    fn entity_inspect(&self, entity_id: u64) -> Result<EntitySnapshot>;
    fn render_stats(&self) -> Result<RenderStats>;
    fn asset_status(&self) -> Result<AssetStatusReport>;
    fn read_log(&self, lines: usize) -> Result<Vec<LogEntry>>;
    fn camera_state(&self) -> Result<CameraSnapshot>;

    // --- Mutation (editor mode only) ---
    fn entity_set_component(&mut self, entity_id: u64, component: ComponentDef) -> Result<()>;
    fn entity_spawn(&mut self, def: EntityDef) -> Result<u64>;
    fn entity_despawn(&mut self, entity_id: u64) -> Result<()>;
    fn material_set(&mut self, id: u16, material: MaterialDef) -> Result<()>;
    fn brush_apply(&mut self, op: CompactEditOp) -> Result<()>;
    fn scene_load(&mut self, path: &str) -> Result<()>;
    fn scene_save(&mut self, path: &str) -> Result<()>;
    fn camera_set(&mut self, position: WorldPosition, rotation: Quat) -> Result<()>;
    fn quality_preset(&mut self, preset: QualityPreset) -> Result<()>;
    fn execute_command(&mut self, command: &str) -> Result<String>;
}
```

The `AutomationApi` is implemented by `rkf-runtime` and passed to `rkf-mcp` at startup. Editor mode exposes all methods. Debug mode exposes only the observation methods.

### Decision: Tool Discovery — Self-Describing Tool Registry

Tools register themselves with full metadata. The MCP server generates tool definitions dynamically from the registry — no hardcoded tool list.

```rust
/// A registered MCP tool
struct ToolDefinition {
    name: String,                           // e.g., "screenshot"
    description: String,                    // human/agent-readable
    category: ToolCategory,                 // Observation, Mutation, Debug
    parameters: Vec<ParameterDef>,          // JSON Schema-compatible
    return_type: ReturnTypeDef,             // what the tool returns
    mode: ToolMode,                         // Editor, Debug, Both
}

struct ParameterDef {
    name: String,
    description: String,
    param_type: ParamType,                  // String, Number, Boolean, Enum, Object
    required: bool,
    default: Option<serde_json::Value>,
}

/// Central registry — tools self-register at startup
struct ToolRegistry {
    tools: HashMap<String, RegisteredTool>,
}

impl ToolRegistry {
    fn register(&mut self, def: ToolDefinition, handler: Box<dyn ToolHandler>);
    fn list_tools(&self, mode: ToolMode) -> Vec<&ToolDefinition>;
    fn call(&self, name: &str, params: serde_json::Value) -> Result<serde_json::Value>;
}

trait ToolHandler: Send + Sync {
    fn call(&self, api: &dyn AutomationApi, params: serde_json::Value) -> Result<serde_json::Value>;
}
```

**Adding a new tool** is three steps:
1. Implement `ToolHandler`
2. Create a `ToolDefinition` with parameters and description
3. Call `registry.register(def, handler)` at startup

The MCP server's `tools/list` response is generated from the registry. No MCP server code changes needed for new tools.

### Decision: Built-In Tool Catalog

**Observation tools (both modes):**

| Tool | Description | Returns |
|------|-------------|---------|
| `screenshot` | Capture current viewport as PNG | Base64 PNG image |
| `screenshot_buffer` | Capture a specific G-buffer channel (depth, normals, material IDs, motion) | Base64 PNG image |
| `scene_graph` | List all entities with hierarchy, types, and transforms | JSON entity tree |
| `entity_inspect` | Read all components of a specific entity | JSON component data |
| `entity_search` | Find entities by name, tag, type, or spatial query | JSON entity list |
| `render_stats` | Frame time, pass timings, brick pool usage, memory | JSON stats |
| `asset_status` | Loading progress, loaded chunks, pending uploads | JSON status report |
| `camera_get` | Current camera position, orientation, FOV | JSON camera state |
| `quality_get` | Current quality preset and per-system settings | JSON config |
| `log_read` | Read recent engine log entries, filterable by level | JSON log entries |
| `brick_pool_stats` | Brick pool occupancy, free list size, LRU state | JSON pool stats |
| `spatial_query` | Query SDF distance/material at a world position | JSON sample data |

**Mutation tools (editor mode only):**

| Tool | Description | Parameters |
|------|-------------|------------|
| `entity_spawn` | Create a new entity | EntityDef JSON |
| `entity_despawn` | Remove an entity | entity_id |
| `entity_set_transform` | Move/rotate/scale an entity | entity_id, transform |
| `entity_set_component` | Modify any component on an entity | entity_id, component_type, data |
| `material_edit` | Modify a material in the table | material_id, fields to update |
| `brush_apply` | Execute a CSG brush operation | CompactEditOp fields |
| `camera_set` | Move the camera to a position/orientation | position, rotation |
| `quality_set` | Change quality preset or individual settings | preset or settings JSON |
| `scene_load` | Load a .rkscene file | file path |
| `scene_save` | Save current scene to .rkscene | file path |
| `undo` | Undo last editor action | — |
| `redo` | Redo last undone action | — |
| `command` | Execute an engine console command | command string |

### Decision: Communication — Local IPC with JSON-RPC

```
Connection lifecycle:
  1. Engine starts → opens IPC listener (Unix socket: /tmp/rkifield-{pid}.sock)
  2. rkf-mcp connects to the socket
  3. MCP protocol over JSON-RPC 2.0
  4. Engine exposes the AutomationApi through the IPC channel
  5. rkf-mcp translates MCP tool calls → AutomationApi calls → MCP responses
```

**Why IPC, not HTTP:**
- Lower latency (no TCP overhead for local communication)
- No port conflicts
- Natural process lifecycle (socket disappears when engine exits)
- Still supports remote access via SSH tunnel if needed

**Fallback:** If IPC is unavailable (platform limitation), fall back to localhost TCP on a configurable port.

### Decision: Engine-Side Integration — Minimal Overhead

The automation API runs on a dedicated thread in the engine process. It communicates with the main engine loop via a channel:

```rust
struct AutomationBridge {
    // Main thread → automation thread: state snapshots
    state_tx: Sender<EngineStateSnapshot>,
    // Automation thread → main thread: mutation requests
    command_rx: Receiver<AutomationCommand>,
}
```

**Observation calls** read from the latest state snapshot (non-blocking, no frame stall).

**Mutation calls** enqueue commands that execute on the main thread at the start of the next frame (same pattern as input events).

**Screenshot capture** uses an async GPU readback — the screenshot is available 1-2 frames after the request. The MCP tool blocks until the readback completes.

### Decision: Testbed and CI Integration

The `rkf-testbed` binary starts with MCP enabled by default. This allows agents to:
1. Launch testbed with a specific scene/test configuration
2. Connect via MCP
3. Take screenshots for visual regression
4. Query render stats for performance regression
5. Inspect scene graph for correctness validation

```bash
# Agent-driven testing workflow
rkf-testbed --scene test_scene.rkscene --mcp &
sleep 1
rkf-mcp --mode debug --connect /tmp/rkifield-$(pgrep rkf-testbed).sock
```

**CI pipeline integration:**
```
1. Build all crates
2. Launch rkf-testbed in headless mode (wgpu software rasterizer)
3. Connect rkf-mcp
4. Execute test script via MCP (load scene, screenshot, compare)
5. Report pass/fail
```

### Decision: MCP Server Grows with the Engine

The MCP tool catalog is not fixed. As new engine features are implemented, corresponding MCP tools are added in the same phase:

| Engine Feature | MCP Tools Added |
|---------------|-----------------|
| Brick pool (Phase 2) | `brick_pool_stats`, `spatial_query` |
| Ray march (Phase 3) | `screenshot`, `screenshot_buffer` |
| Shading (Phase 5) | `render_stats` |
| ECS (Phase 11) | `scene_graph`, `entity_*` tools |
| Materials (Phase 14) | `material_edit` |
| Editing (Phase 15) | `brush_apply`, `undo`, `redo` |
| Streaming (Phase 13) | `asset_status` |
| Editor (Phase 21) | `scene_load`, `scene_save`, full mutation set |

**Rule:** Every new user-facing feature gets an MCP tool in the same PR. No feature ships without agent accessibility.

### Decision: Resource Format for Screenshots

Screenshots are returned as base64-encoded PNG in the MCP response. For large/frequent screenshots, an optional `file` parameter saves to disk and returns the path instead.

```json
{
    "tool": "screenshot",
    "params": {
        "width": 1920,
        "height": 1080,
        "buffer": "color",
        "format": "png",
        "save_to": null
    }
}
```

Buffer options: `color` (final composited), `depth`, `normal`, `material_id`, `motion`, `gi_radiance`, `volumetric`.

### Session Summary: MCP Integration Decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| Architecture | Standalone `rkf-mcp` binary + engine `AutomationApi` trait | Clean separation |
| Communication | Local IPC (Unix socket), JSON-RPC 2.0 | TCP fallback |
| Tool discovery | Self-describing registry, dynamic tool list | New tools = implement trait + register |
| Modes | Editor (full access) + Debug (read-only) | Same binary, mode flag |
| Observation tools | 12 built-in (screenshot, scene graph, entity, stats, logs, etc.) | Grow with engine |
| Mutation tools | 12 built-in (spawn, edit, brush, camera, scene, etc.) | Editor mode only |
| Screenshot | Async GPU readback, base64 PNG or file, multiple buffer types | 1-2 frame latency |
| Engine integration | Dedicated thread, channel-based, non-blocking observations | No frame stalls |
| CI/testing | Testbed + headless wgpu + MCP for visual/perf regression | Agent-driven QA |
| Growth rule | Every new feature gets MCP tools in the same phase | No feature without agent access |
