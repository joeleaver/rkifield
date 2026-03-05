//! Shader composer — CPU-side WGSL composition for the uber-shader pipeline.
//!
//! WGSL has no `#include` directive. This module concatenates separate shader
//! files into a single uber-shader:
//!
//! ```text
//! shade_common.wgsl   →  shared infra (structs, bindings, SDF, shadows, AO, GI, PBR math, noise)
//! shade_pbr.wgsl      →  fn shade_pbr(ctx: ShadingContext) -> vec3<f32>
//! [other models]      →  fn shade_<name>(ctx: ShadingContext) -> vec3<f32>
//! [dispatch function] →  fn dispatch_shade(id: u32, ctx: ShadingContext) -> vec3<f32>
//! shade_main.wgsl     →  @compute entry point
//! ```
//!
//! The `dispatch_shade` switch statement is generated dynamically based on
//! registered shading models.

use std::collections::HashMap;

/// A registered shading model.
#[derive(Debug, Clone)]
struct ShaderEntry {
    /// Unique numeric ID (used in GPU switch statement).
    id: u32,
    /// Human-readable name (e.g. "pbr", "toon", "hologram").
    name: String,
    /// WGSL source containing `fn shade_<name>(ctx: ShadingContext) -> vec3<f32>`.
    source: String,
    /// Whether this is a built-in shader (vs user-provided).
    built_in: bool,
    /// File path to the WGSL source file (for editor "open in editor" feature).
    file_path: Option<String>,
}

/// Summary data for a registered shader (for UI display).
#[derive(Debug, Clone)]
pub struct ShaderSummaryData {
    pub name: String,
    pub id: u32,
    pub built_in: bool,
    pub file_path: String,
}

/// Composes separate WGSL shader files into a single uber-shader and manages
/// a registry of shading model name↔id mappings.
pub struct ShaderComposer {
    /// Shared infrastructure WGSL (structs, bindings, utility functions).
    common: String,
    /// Entry point WGSL (reads G-buffer, builds ShadingContext, debug modes).
    main_template: String,
    /// Registered shading models.
    shaders: Vec<ShaderEntry>,
    /// Name → id lookup.
    name_to_id: HashMap<String, u32>,
    /// Next auto-assigned ID for user shaders.
    next_id: u32,
    /// Cached composed WGSL source.
    composed: String,
    /// Whether the composed source needs regeneration.
    dirty: bool,
}

impl ShaderComposer {
    /// Create a new composer with the built-in PBR shader.
    pub fn new() -> Self {
        let common = include_str!("../shaders/shade_common.wgsl").to_string();
        let main_template = include_str!("../shaders/shade_main.wgsl").to_string();
        let pbr_source = include_str!("../shaders/shade_pbr.wgsl").to_string();

        let mut composer = Self {
            common,
            main_template,
            shaders: Vec::new(),
            name_to_id: HashMap::new(),
            next_id: 0,
            composed: String::new(),
            dirty: true,
        };

        // Register built-in shaders.
        composer.register_built_in("pbr", pbr_source);        // id=0

        let unlit_source = include_str!("../shaders/shade_unlit.wgsl").to_string();
        let toon_source = include_str!("../shaders/shade_toon.wgsl").to_string();
        let emissive_source = include_str!("../shaders/shade_emissive.wgsl").to_string();
        composer.register_built_in("unlit", unlit_source);     // id=1
        composer.register_built_in("toon", toon_source);       // id=2
        composer.register_built_in("emissive", emissive_source); // id=3

        composer
    }

    /// Register a built-in shading model. Returns the assigned ID.
    fn register_built_in(&mut self, name: &str, source: String) -> u32 {
        let file_path = format!("crates/rkf-render/shaders/shade_{name}.wgsl");
        let id = self.next_id;
        self.next_id += 1;
        self.shaders.push(ShaderEntry {
            id,
            name: name.to_string(),
            source,
            built_in: true,
            file_path: Some(file_path),
        });
        self.name_to_id.insert(name.to_string(), id);
        self.dirty = true;
        id
    }

    /// Register a user-provided shading model. Returns the assigned ID.
    ///
    /// If a shader with the same name already exists, its source is updated
    /// and the existing ID is returned.
    pub fn register(&mut self, name: &str, source: String) -> u32 {
        self.register_with_path(name, source, None)
    }

    /// Register a user-provided shading model with an optional file path.
    /// Returns the assigned ID.
    ///
    /// If a shader with the same name already exists, its source (and path) are
    /// updated and the existing ID is returned.
    pub fn register_with_path(&mut self, name: &str, source: String, file_path: Option<String>) -> u32 {
        if let Some(&existing_id) = self.name_to_id.get(name) {
            // Update existing shader source.
            if let Some(entry) = self.shaders.iter_mut().find(|e| e.id == existing_id) {
                entry.source = source;
                if file_path.is_some() {
                    entry.file_path = file_path;
                }
                self.dirty = true;
            }
            return existing_id;
        }

        let id = self.next_id;
        self.next_id += 1;
        self.shaders.push(ShaderEntry {
            id,
            name: name.to_string(),
            source,
            built_in: false,
            file_path,
        });
        self.name_to_id.insert(name.to_string(), id);
        self.dirty = true;
        id
    }

    /// Look up the numeric ID for a shader name. Returns 0 (PBR) for unknown names.
    pub fn shader_id(&self, name: &str) -> u32 {
        self.name_to_id.get(name).copied().unwrap_or(0)
    }

    /// Look up the name for a shader ID.
    pub fn shader_name(&self, id: u32) -> Option<&str> {
        self.shaders.iter().find(|e| e.id == id).map(|e| e.name.as_str())
    }

    /// List all registered shader names in ID order.
    pub fn shader_names(&self) -> Vec<String> {
        let mut entries: Vec<_> = self.shaders.iter().collect();
        entries.sort_by_key(|e| e.id);
        entries.iter().map(|e| e.name.clone()).collect()
    }

    /// List all shaders with their info (name, id, built_in).
    pub fn shader_info(&self) -> Vec<(String, u32, bool)> {
        let mut entries: Vec<_> = self.shaders.iter().collect();
        entries.sort_by_key(|e| e.id);
        entries.iter().map(|e| (e.name.clone(), e.id, e.built_in)).collect()
    }

    /// List all shaders with full summary data (for UI display).
    pub fn shader_summaries(&self) -> Vec<ShaderSummaryData> {
        let mut entries: Vec<_> = self.shaders.iter().collect();
        entries.sort_by_key(|e| e.id);
        entries.iter().map(|e| ShaderSummaryData {
            name: e.name.clone(),
            id: e.id,
            built_in: e.built_in,
            file_path: e.file_path.clone().unwrap_or_default(),
        }).collect()
    }

    /// Whether the composed source needs regeneration.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Generate the dispatch_shade function as a WGSL switch statement.
    fn generate_dispatch(&self) -> String {
        let mut s = String::new();
        s.push_str("fn dispatch_shade(shader_id: u32, ctx: ShadingContext) -> vec3<f32> {\n");
        s.push_str("    switch shader_id {\n");

        for entry in &self.shaders {
            s.push_str(&format!(
                "        case {}u: {{ return shade_{}(ctx); }}\n",
                entry.id, entry.name
            ));
        }

        // Default: fall back to PBR.
        s.push_str("        default: { return shade_pbr(ctx); }\n");
        s.push_str("    }\n");
        s.push_str("}\n");
        s
    }

    /// Compose all parts into a single WGSL string.
    ///
    /// The result is: common + all shader sources + dispatch function + main.
    /// The dispatch placeholder in shade_main.wgsl is replaced with the
    /// generated switch statement.
    pub fn compose(&mut self) -> &str {
        if !self.dirty {
            return &self.composed;
        }

        let mut result = String::with_capacity(
            self.common.len()
                + self.shaders.iter().map(|s| s.source.len()).sum::<usize>()
                + self.main_template.len()
                + 512,
        );

        // 1. Common infrastructure
        result.push_str(&self.common);
        result.push('\n');

        // 2. All shading model functions
        for entry in &self.shaders {
            result.push_str(&format!("// --- Shading model: {} (id={}) ---\n", entry.name, entry.id));
            result.push_str(&entry.source);
            result.push('\n');
        }

        // 3. Generate dispatch function
        let dispatch = self.generate_dispatch();
        result.push_str(&dispatch);
        result.push('\n');

        // 4. Main entry point (with placeholder replaced)
        let main = self.main_template.replace(
            "// SHADER_DISPATCH_PLACEHOLDER",
            "// Dispatch function generated by ShaderComposer (see above).",
        );
        result.push_str(&main);

        self.composed = result;
        self.dirty = false;
        &self.composed
    }

    /// Update the source of a shader by name. Marks dirty.
    pub fn update_source(&mut self, name: &str, source: String) -> bool {
        if let Some(&id) = self.name_to_id.get(name) {
            if let Some(entry) = self.shaders.iter_mut().find(|e| e.id == id) {
                entry.source = source;
                self.dirty = true;
                return true;
            }
        }
        false
    }
}

impl Default for ShaderComposer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registers_pbr() {
        let composer = ShaderComposer::new();
        assert_eq!(composer.shader_id("pbr"), 0);
        assert_eq!(composer.shader_name(0), Some("pbr"));
        assert_eq!(composer.shader_id("unlit"), 1);
        assert_eq!(composer.shader_name(1), Some("unlit"));
        assert_eq!(composer.shader_id("toon"), 2);
        assert_eq!(composer.shader_name(2), Some("toon"));
        assert_eq!(composer.shader_id("emissive"), 3);
        assert_eq!(composer.shader_name(3), Some("emissive"));
    }

    #[test]
    fn unknown_name_returns_zero() {
        let composer = ShaderComposer::new();
        assert_eq!(composer.shader_id("nonexistent"), 0);
    }

    #[test]
    fn register_user_shader() {
        let mut composer = ShaderComposer::new();
        let id = composer.register("hologram", "fn shade_hologram(ctx: ShadingContext) -> vec3<f32> { return vec3<f32>(0.0); }".into());
        assert_eq!(id, 4); // after pbr=0, unlit=1, toon=2, emissive=3
        assert_eq!(composer.shader_id("hologram"), id);
        assert_eq!(composer.shader_name(id), Some("hologram"));
    }

    #[test]
    fn register_duplicate_updates_source() {
        let mut composer = ShaderComposer::new();
        let id1 = composer.register("test", "source1".into());
        let id2 = composer.register("test", "source2".into());
        assert_eq!(id1, id2);
    }

    #[test]
    fn shader_names_in_id_order() {
        let mut composer = ShaderComposer::new();
        composer.register("zebra", "z".into());
        composer.register("alpha", "a".into());
        let names = composer.shader_names();
        assert_eq!(names[0], "pbr");
        assert_eq!(names[1], "unlit");
        assert_eq!(names[2], "toon");
        assert_eq!(names[3], "emissive");
        // After built-ins, zebra was registered first, then alpha.
        assert_eq!(names[4], "zebra");
        assert_eq!(names[5], "alpha");
    }

    #[test]
    fn compose_contains_all_parts() {
        let mut composer = ShaderComposer::new();
        let source = composer.compose().to_string();

        // Contains common (has ShadingContext struct)
        assert!(source.contains("struct ShadingContext"));
        // Contains all built-in shading model functions
        assert!(source.contains("fn shade_pbr"));
        assert!(source.contains("fn shade_unlit"));
        assert!(source.contains("fn shade_toon"));
        assert!(source.contains("fn shade_emissive"));
        // Contains dispatch
        assert!(source.contains("fn dispatch_shade"));
        // Contains entry point
        assert!(source.contains("fn main("));
        // Marker line was replaced (the comment describing the mechanism may remain)
        assert!(!source.contains("// SHADER_DISPATCH_PLACEHOLDER\n"));
    }

    #[test]
    fn compose_dispatch_includes_registered_shaders() {
        let mut composer = ShaderComposer::new();
        composer.register("test_model", "fn shade_test_model(ctx: ShadingContext) -> vec3<f32> { return vec3<f32>(1.0); }".into());
        let source = composer.compose().to_string();

        assert!(source.contains("case 0u: { return shade_pbr(ctx); }"));
        assert!(source.contains("case 1u: { return shade_unlit(ctx); }"));
        assert!(source.contains("case 2u: { return shade_toon(ctx); }"));
        assert!(source.contains("case 3u: { return shade_emissive(ctx); }"));
        assert!(source.contains("case 4u: { return shade_test_model(ctx); }"));
        assert!(source.contains("default: { return shade_pbr(ctx); }"));
    }

    #[test]
    fn dirty_tracking() {
        let mut composer = ShaderComposer::new();
        assert!(composer.is_dirty());
        composer.compose();
        assert!(!composer.is_dirty());
        composer.register("new_shader", "source".into());
        assert!(composer.is_dirty());
        composer.compose();
        assert!(!composer.is_dirty());
    }

    #[test]
    fn update_source_marks_dirty() {
        let mut composer = ShaderComposer::new();
        composer.compose();
        assert!(!composer.is_dirty());
        assert!(composer.update_source("pbr", "new source".into()));
        assert!(composer.is_dirty());
    }

    #[test]
    fn update_nonexistent_returns_false() {
        let mut composer = ShaderComposer::new();
        assert!(!composer.update_source("nonexistent", "source".into()));
    }

    #[test]
    fn shader_info_includes_built_in_flag() {
        let mut composer = ShaderComposer::new();
        composer.register("custom", "source".into());
        let info = composer.shader_info();
        assert!(info.iter().any(|(name, _, built_in)| name == "pbr" && *built_in));
        assert!(info.iter().any(|(name, _, built_in)| name == "custom" && !*built_in));
    }
}
