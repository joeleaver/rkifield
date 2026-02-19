//! Shader hot-reload — registry, change detection, and compilation state.
//!
//! [`ShaderRegistry`] tracks all registered WGSL shader sources by name and
//! path. [`ShaderReloadState`] queues pending reload events and records
//! compilation failures so the editor/runtime can display diagnostics.

#![allow(dead_code)]

// ── Shader Source ───────────────────────────────────────────────────────────

/// A tracked shader source file.
#[derive(Debug, Clone)]
pub struct ShaderSource {
    /// Filesystem path to the `.wgsl` file.
    pub path: String,
    /// Human-readable name used as a lookup key.
    pub name: String,
    /// Last-known modification timestamp (epoch seconds or frame number).
    pub last_modified: u64,
    /// FNV-1a hash of the most recently loaded source text.
    pub source_hash: u64,
    /// Whether the shader has been successfully compiled at least once.
    pub compiled: bool,
}

// ── Registry ────────────────────────────────────────────────────────────────

/// Central registry of all shader sources known to the engine.
#[derive(Debug, Clone)]
pub struct ShaderRegistry {
    /// All registered shaders.
    shaders: Vec<ShaderSource>,
    /// Directories being watched for changes.
    pub watch_paths: Vec<String>,
}

impl ShaderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            shaders: Vec::new(),
            watch_paths: Vec::new(),
        }
    }

    /// Register a new shader. Duplicate names are allowed (latest wins on lookup).
    pub fn register_shader(&mut self, path: &str, name: &str) {
        self.shaders.push(ShaderSource {
            path: path.to_string(),
            name: name.to_string(),
            last_modified: 0,
            source_hash: 0,
            compiled: false,
        });
    }

    /// Find a shader by name. Returns the first match.
    pub fn find_shader(&self, name: &str) -> Option<&ShaderSource> {
        self.shaders.iter().find(|s| s.name == name)
    }

    /// Total number of registered shaders.
    pub fn shader_count(&self) -> usize {
        self.shaders.len()
    }

    /// Slice of all registered shaders.
    pub fn all_shaders(&self) -> &[ShaderSource] {
        &self.shaders
    }
}

impl Default for ShaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Change Events ───────────────────────────────────────────────────────────

/// What kind of filesystem event triggered a reload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// File contents were modified.
    Modified,
    /// A new shader file appeared.
    Created,
    /// A shader file was removed.
    Deleted,
}

/// An event indicating a shader source changed on disk.
#[derive(Debug, Clone)]
pub struct ShaderChangeEvent {
    /// Name of the affected shader.
    pub shader_name: String,
    /// Filesystem path that changed.
    pub path: String,
    /// Kind of change.
    pub change_type: ChangeType,
}

// ── Compile Errors ──────────────────────────────────────────────────────────

/// A compilation failure recorded by the reload system.
#[derive(Debug, Clone)]
pub struct CompileError {
    /// Name of the shader that failed.
    pub shader_name: String,
    /// Error message from the compiler (naga / wgpu).
    pub message: String,
    /// Source line of the error, if available.
    pub line: Option<u32>,
    /// Source column of the error, if available.
    pub column: Option<u32>,
}

// ── Reload State ────────────────────────────────────────────────────────────

/// Mutable state for the shader hot-reload pipeline.
#[derive(Debug, Clone)]
pub struct ShaderReloadState {
    /// Events waiting to be processed.
    pending_reloads: Vec<ShaderChangeEvent>,
    /// Errors from the most recent compilation attempts.
    failed_compiles: Vec<CompileError>,
    /// Frame number of the last successful reload.
    pub last_reload_frame: u64,
    /// Whether file-watcher auto-reload is enabled.
    auto_reload_enabled: bool,
}

impl ShaderReloadState {
    /// Create a new reload state with auto-reload enabled by default.
    pub fn new() -> Self {
        Self {
            pending_reloads: Vec::new(),
            failed_compiles: Vec::new(),
            last_reload_frame: 0,
            auto_reload_enabled: true,
        }
    }

    /// Queue a change event for processing.
    pub fn queue_reload(&mut self, event: ShaderChangeEvent) {
        self.pending_reloads.push(event);
    }

    /// Drain all pending reload events, returning them.
    pub fn drain_pending(&mut self) -> Vec<ShaderChangeEvent> {
        std::mem::take(&mut self.pending_reloads)
    }

    /// Record a compilation error.
    pub fn record_compile_error(&mut self, error: CompileError) {
        self.failed_compiles.push(error);
    }

    /// Clear all recorded compilation errors.
    pub fn clear_errors(&mut self) {
        self.failed_compiles.clear();
    }

    /// Whether there are any outstanding compilation errors.
    pub fn has_errors(&self) -> bool {
        !self.failed_compiles.is_empty()
    }

    /// Number of outstanding compilation errors.
    pub fn error_count(&self) -> usize {
        self.failed_compiles.len()
    }

    /// The most recently recorded error, if any.
    pub fn last_error(&self) -> Option<&CompileError> {
        self.failed_compiles.last()
    }

    /// Slice of all compilation errors.
    pub fn errors(&self) -> &[CompileError] {
        &self.failed_compiles
    }

    /// Enable or disable auto-reload.
    pub fn set_auto_reload(&mut self, enabled: bool) {
        self.auto_reload_enabled = enabled;
    }

    /// Whether auto-reload is currently enabled.
    pub fn is_auto_reload(&self) -> bool {
        self.auto_reload_enabled
    }
}

impl Default for ShaderReloadState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Hash Utilities ──────────────────────────────────────────────────────────

/// Compute a 64-bit FNV-1a hash of the given shader source text.
pub fn compute_source_hash(source: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;

    let mut hash = FNV_OFFSET;
    for byte in source.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Check whether a shader's source has changed based on hash comparison.
pub fn check_shader_changed(shader: &ShaderSource, new_hash: u64) -> bool {
    shader.source_hash != new_hash
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // -- ShaderRegistry -------------------------------------------------------

    #[test]
    fn registry_starts_empty() {
        let reg = ShaderRegistry::new();
        assert_eq!(reg.shader_count(), 0);
        assert!(reg.all_shaders().is_empty());
    }

    #[test]
    fn register_and_find_shader() {
        let mut reg = ShaderRegistry::new();
        reg.register_shader("shaders/ray_march.wgsl", "ray_march");
        reg.register_shader("shaders/shade.wgsl", "shade");

        assert_eq!(reg.shader_count(), 2);
        let found = reg.find_shader("ray_march").unwrap();
        assert_eq!(found.path, "shaders/ray_march.wgsl");
        assert!(!found.compiled);
    }

    #[test]
    fn find_shader_not_found() {
        let reg = ShaderRegistry::new();
        assert!(reg.find_shader("nonexistent").is_none());
    }

    #[test]
    fn all_shaders_returns_slice() {
        let mut reg = ShaderRegistry::new();
        reg.register_shader("a.wgsl", "a");
        reg.register_shader("b.wgsl", "b");
        reg.register_shader("c.wgsl", "c");
        assert_eq!(reg.all_shaders().len(), 3);
    }

    // -- ShaderReloadState ----------------------------------------------------

    #[test]
    fn reload_state_starts_clean() {
        let state = ShaderReloadState::new();
        assert!(!state.has_errors());
        assert_eq!(state.error_count(), 0);
        assert!(state.is_auto_reload());
    }

    #[test]
    fn queue_and_drain_events() {
        let mut state = ShaderReloadState::new();
        state.queue_reload(ShaderChangeEvent {
            shader_name: "ray_march".into(),
            path: "shaders/ray_march.wgsl".into(),
            change_type: ChangeType::Modified,
        });
        state.queue_reload(ShaderChangeEvent {
            shader_name: "shade".into(),
            path: "shaders/shade.wgsl".into(),
            change_type: ChangeType::Created,
        });

        let drained = state.drain_pending();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].shader_name, "ray_march");
        assert_eq!(drained[1].change_type, ChangeType::Created);

        // After drain, pending is empty.
        assert!(state.drain_pending().is_empty());
    }

    #[test]
    fn compile_errors_lifecycle() {
        let mut state = ShaderReloadState::new();
        assert!(!state.has_errors());

        state.record_compile_error(CompileError {
            shader_name: "shade".into(),
            message: "unexpected token".into(),
            line: Some(42),
            column: Some(10),
        });
        assert!(state.has_errors());
        assert_eq!(state.error_count(), 1);

        let last = state.last_error().unwrap();
        assert_eq!(last.shader_name, "shade");
        assert_eq!(last.line, Some(42));

        state.clear_errors();
        assert!(!state.has_errors());
        assert!(state.last_error().is_none());
    }

    #[test]
    fn errors_returns_full_slice() {
        let mut state = ShaderReloadState::new();
        state.record_compile_error(CompileError {
            shader_name: "a".into(),
            message: "err1".into(),
            line: None,
            column: None,
        });
        state.record_compile_error(CompileError {
            shader_name: "b".into(),
            message: "err2".into(),
            line: Some(1),
            column: None,
        });
        assert_eq!(state.errors().len(), 2);
    }

    #[test]
    fn auto_reload_toggle() {
        let mut state = ShaderReloadState::new();
        assert!(state.is_auto_reload());
        state.set_auto_reload(false);
        assert!(!state.is_auto_reload());
        state.set_auto_reload(true);
        assert!(state.is_auto_reload());
    }

    // -- Hash Utilities -------------------------------------------------------

    #[test]
    fn source_hash_consistent() {
        let src = "@compute fn main() {}";
        let h1 = compute_source_hash(src);
        let h2 = compute_source_hash(src);
        assert_eq!(h1, h2);
    }

    #[test]
    fn source_hash_different_for_different_input() {
        let h1 = compute_source_hash("fn a() {}");
        let h2 = compute_source_hash("fn b() {}");
        assert_ne!(h1, h2);
    }

    #[test]
    fn source_hash_empty_string() {
        let h = compute_source_hash("");
        // FNV-1a of empty input is the offset basis.
        assert_eq!(h, 0xcbf29ce484222325);
    }

    #[test]
    fn check_shader_changed_detects_change() {
        let shader = ShaderSource {
            path: "test.wgsl".into(),
            name: "test".into(),
            last_modified: 0,
            source_hash: 12345,
            compiled: true,
        };
        assert!(check_shader_changed(&shader, 99999));
        assert!(!check_shader_changed(&shader, 12345));
    }
}
