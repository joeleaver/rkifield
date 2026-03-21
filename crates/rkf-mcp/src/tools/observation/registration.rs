//! Tool registration for all observation tools.
//!
//! The [`register_observation_tools`] function registers handler structs from
//! [`super::handlers`] with a [`ToolRegistry`].

use super::handlers::*;
use crate::registry::*;
use std::sync::Arc;

/// Register all built-in observation tools with the registry.
pub fn register_observation_tools(registry: &mut ToolRegistry) {
    registry.register(
        ToolDefinition {
            name: "screenshot".to_string(),
            description: "Capture current viewport as PNG image".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "width".to_string(),
                    description: "Image width in pixels".to_string(),
                    param_type: ParamType::Integer,
                    required: false,
                    default: Some(serde_json::json!(1920)),
                },
                ParameterDef {
                    name: "height".to_string(),
                    description: "Image height in pixels".to_string(),
                    param_type: ParamType::Integer,
                    required: false,
                    default: Some(serde_json::json!(1080)),
                },
            ],
            return_type: ReturnTypeDef {
                description: "Base64-encoded PNG image data".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(ScreenshotHandler),
    );

    registry.register(
        ToolDefinition {
            name: "screenshot_window".to_string(),
            description: "Capture full editor window (UI + viewport composited) as PNG image".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "Base64-encoded PNG image of the full editor window".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(ScreenshotWindowHandler),
    );

    registry.register(
        ToolDefinition {
            name: "scene_graph".to_string(),
            description: "List all entities with hierarchy, types, and transforms".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "JSON entity tree".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(SceneGraphHandler),
    );

    registry.register(
        ToolDefinition {
            name: "entity_inspect".to_string(),
            description: "Read all components of a specific entity".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![ParameterDef {
                name: "entity_id".to_string(),
                description: "Entity ID to inspect".to_string(),
                param_type: ParamType::Integer,
                required: true,
                default: None,
            }],
            return_type: ReturnTypeDef {
                description: "JSON component data".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(EntityInspectHandler),
    );

    registry.register(
        ToolDefinition {
            name: "render_stats".to_string(),
            description: "Frame time, pass timings, brick pool usage, memory stats".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "JSON stats object".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(RenderStatsHandler),
    );

    registry.register(
        ToolDefinition {
            name: "log_read".to_string(),
            description: "Read recent engine log entries".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![ParameterDef {
                name: "lines".to_string(),
                description: "Number of log lines to return".to_string(),
                param_type: ParamType::Integer,
                required: false,
                default: Some(serde_json::json!(50)),
            }],
            return_type: ReturnTypeDef {
                description: "JSON array of log entries".to_string(),
                return_type: ParamType::Array,
            },
            mode: ToolMode::Both,
        },
        Arc::new(LogReadHandler),
    );

    registry.register(
        ToolDefinition {
            name: "camera_get".to_string(),
            description: "Current camera position, orientation, and FOV".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "JSON camera state".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(CameraGetHandler),
    );

    registry.register(
        ToolDefinition {
            name: "brick_pool_stats".to_string(),
            description: "Brick pool occupancy, free list size, LRU state".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "JSON pool stats".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(BrickPoolStatsHandler),
    );

    registry.register(
        ToolDefinition {
            name: "spatial_query".to_string(),
            description: "Query SDF distance and material at a world position".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "chunk_x".to_string(),
                    description: "Chunk X coordinate".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "chunk_y".to_string(),
                    description: "Chunk Y coordinate".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "chunk_z".to_string(),
                    description: "Chunk Z coordinate".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "local_x".to_string(),
                    description: "Local X position within chunk".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "local_y".to_string(),
                    description: "Local Y position within chunk".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "local_z".to_string(),
                    description: "Local Z position within chunk".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "SDF distance, material ID, inside flag".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(SpatialQueryHandler),
    );

    registry.register(
        ToolDefinition {
            name: "asset_status".to_string(),
            description: "Loading progress, loaded chunks, pending uploads".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "JSON status report".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(AssetStatusHandler),
    );

    registry.register(
        ToolDefinition {
            name: "debug_mode".to_string(),
            description: "Set shading debug visualization mode. 0=normal, 1=normals, 2=positions, 3=material IDs, 4=diffuse, 5=specular, 6=GI, 7=SDF distance, 8=brick boundaries".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![ParameterDef {
                name: "mode".to_string(),
                description: "Debug mode (0=normal, 1=normals, 2=positions, 3=material IDs, 4=diffuse, 5=specular, 6=GI, 7=SDF distance, 8=brick boundaries)".to_string(),
                param_type: ParamType::Integer,
                required: true,
                default: None,
            }],
            return_type: ReturnTypeDef {
                description: "Confirmation of mode change".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(DebugModeHandler),
    );

    registry.register(
        ToolDefinition {
            name: "camera_set".to_string(),
            description: "Set camera position and orientation".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "x".to_string(),
                    description: "X position".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "y".to_string(),
                    description: "Y position".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "z".to_string(),
                    description: "Z position".to_string(),
                    param_type: ParamType::Number,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "yaw".to_string(),
                    description: "Yaw in degrees".to_string(),
                    param_type: ParamType::Number,
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                },
                ParameterDef {
                    name: "pitch".to_string(),
                    description: "Pitch in degrees".to_string(),
                    param_type: ParamType::Number,
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of camera position change".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(CameraSetHandler),
    );

    registry.register(
        ToolDefinition {
            name: "env_set".to_string(),
            description: "Set an environment property (atmosphere, fog, clouds, post-processing). Use env_get with property='all' to see available properties.".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "property".to_string(),
                    description: "Property path, e.g. 'clouds.enabled', 'atmosphere.sun_intensity', 'post_process.exposure', 'fog.density'".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "value".to_string(),
                    description: "Value to set (number, boolean, or string)".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of property change".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(EnvSetHandler),
    );

    registry.register(
        ToolDefinition {
            name: "env_get".to_string(),
            description: "Get an environment property value. Use property='all' for a summary of all settings.".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![ParameterDef {
                name: "property".to_string(),
                description: "Property path (e.g. 'clouds.enabled', 'atmosphere.sun_intensity') or 'all' for summary".to_string(),
                param_type: ParamType::String,
                required: false,
                default: Some(serde_json::json!("all")),
            }],
            return_type: ReturnTypeDef {
                description: "Property value".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(EnvGetHandler),
    );

    // --- Node tree tools ---

    registry.register(
        ToolDefinition {
            name: "node_find".to_string(),
            description: "Find a node by name within an object's SDF tree and return its properties".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef { name: "object_id".to_string(), description: "SDF object ID".to_string(), param_type: ParamType::Integer, required: true, default: None },
                ParameterDef { name: "node_name".to_string(), description: "Node name to find".to_string(), param_type: ParamType::String, required: true, default: None },
            ],
            return_type: ReturnTypeDef { description: "Node properties as JSON".to_string(), return_type: ParamType::Object },
            mode: ToolMode::Both,
        },
        Arc::new(NodeFindHandler),
    );

    registry.register(
        ToolDefinition {
            name: "node_add_child".to_string(),
            description: "Add a child SDF primitive node to a named parent node within an object".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef { name: "object_id".to_string(), description: "SDF object ID".to_string(), param_type: ParamType::Integer, required: true, default: None },
                ParameterDef { name: "parent_node".to_string(), description: "Parent node name".to_string(), param_type: ParamType::String, required: true, default: None },
                ParameterDef { name: "child_primitive".to_string(), description: "Primitive type: sphere, box, capsule, torus, cylinder, plane".to_string(), param_type: ParamType::String, required: true, default: None },
                ParameterDef { name: "params".to_string(), description: "Shape parameters (e.g. [0.5] for sphere radius)".to_string(), param_type: ParamType::Array, required: false, default: Some(serde_json::json!([])) },
                ParameterDef { name: "name".to_string(), description: "Name for the new child node".to_string(), param_type: ParamType::String, required: true, default: None },
                ParameterDef { name: "material_id".to_string(), description: "Material table index".to_string(), param_type: ParamType::Integer, required: false, default: Some(serde_json::json!(0)) },
            ],
            return_type: ReturnTypeDef { description: "Confirmation".to_string(), return_type: ParamType::Object },
            mode: ToolMode::Editor,
        },
        Arc::new(NodeAddChildHandler),
    );

    registry.register(
        ToolDefinition {
            name: "node_remove".to_string(),
            description: "Remove a named node from an object's SDF tree".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef { name: "object_id".to_string(), description: "SDF object ID".to_string(), param_type: ParamType::Integer, required: true, default: None },
                ParameterDef { name: "node_name".to_string(), description: "Node name to remove".to_string(), param_type: ParamType::String, required: true, default: None },
            ],
            return_type: ReturnTypeDef { description: "Confirmation".to_string(), return_type: ParamType::Object },
            mode: ToolMode::Editor,
        },
        Arc::new(NodeRemoveHandler),
    );

    // --- Multi-scene tools ---

    registry.register(
        ToolDefinition {
            name: "scene_create".to_string(),
            description: "Create a new empty scene and return its index".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef { name: "name".to_string(), description: "Scene name".to_string(), param_type: ParamType::String, required: true, default: None },
            ],
            return_type: ReturnTypeDef { description: "New scene index".to_string(), return_type: ParamType::Object },
            mode: ToolMode::Editor,
        },
        Arc::new(SceneCreateHandler),
    );

    registry.register(
        ToolDefinition {
            name: "scene_list".to_string(),
            description: "List all scenes with names, active state, and persistence flags".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef { description: "JSON array of scene info".to_string(), return_type: ParamType::Array },
            mode: ToolMode::Both,
        },
        Arc::new(SceneListHandler),
    );

    registry.register(
        ToolDefinition {
            name: "scene_set_active".to_string(),
            description: "Set the active scene by index".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef { name: "index".to_string(), description: "Scene index".to_string(), param_type: ParamType::Integer, required: true, default: None },
            ],
            return_type: ReturnTypeDef { description: "Confirmation".to_string(), return_type: ParamType::Object },
            mode: ToolMode::Editor,
        },
        Arc::new(SceneSetActiveHandler),
    );

    registry.register(
        ToolDefinition {
            name: "scene_set_persistent".to_string(),
            description: "Mark a scene as persistent (survives scene swaps) or not".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef { name: "index".to_string(), description: "Scene index".to_string(), param_type: ParamType::Integer, required: true, default: None },
                ParameterDef { name: "persistent".to_string(), description: "Whether the scene is persistent".to_string(), param_type: ParamType::Boolean, required: true, default: None },
            ],
            return_type: ReturnTypeDef { description: "Confirmation".to_string(), return_type: ParamType::Object },
            mode: ToolMode::Editor,
        },
        Arc::new(SceneSetPersistentHandler),
    );

    registry.register(
        ToolDefinition {
            name: "scene_swap".to_string(),
            description: "Unload all non-persistent scenes, keeping persistent ones".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![],
            return_type: ReturnTypeDef { description: "Removed scene names and remaining count".to_string(), return_type: ParamType::Object },
            mode: ToolMode::Editor,
        },
        Arc::new(SceneSwapHandler),
    );

    // --- Camera entity tools ---

    registry.register(
        ToolDefinition {
            name: "camera_spawn".to_string(),
            description: "Spawn a camera entity at a position with orientation and FOV".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef { name: "label".to_string(), description: "Camera display name".to_string(), param_type: ParamType::String, required: true, default: None },
                ParameterDef { name: "x".to_string(), description: "X position".to_string(), param_type: ParamType::Number, required: false, default: Some(serde_json::json!(0.0)) },
                ParameterDef { name: "y".to_string(), description: "Y position".to_string(), param_type: ParamType::Number, required: false, default: Some(serde_json::json!(0.0)) },
                ParameterDef { name: "z".to_string(), description: "Z position".to_string(), param_type: ParamType::Number, required: false, default: Some(serde_json::json!(0.0)) },
                ParameterDef { name: "yaw".to_string(), description: "Yaw in degrees".to_string(), param_type: ParamType::Number, required: false, default: Some(serde_json::json!(0.0)) },
                ParameterDef { name: "pitch".to_string(), description: "Pitch in degrees".to_string(), param_type: ParamType::Number, required: false, default: Some(serde_json::json!(0.0)) },
                ParameterDef { name: "fov".to_string(), description: "Vertical FOV in degrees".to_string(), param_type: ParamType::Number, required: false, default: Some(serde_json::json!(60.0)) },
            ],
            return_type: ReturnTypeDef { description: "Entity ID of the spawned camera".to_string(), return_type: ParamType::Object },
            mode: ToolMode::Editor,
        },
        Arc::new(CameraSpawnHandler),
    );

    registry.register(
        ToolDefinition {
            name: "camera_list".to_string(),
            description: "List all camera entities with positions, orientations, and FOV".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef { description: "JSON array of camera entities".to_string(), return_type: ParamType::Array },
            mode: ToolMode::Both,
        },
        Arc::new(CameraListHandler),
    );

    registry.register(
        ToolDefinition {
            name: "camera_snap_to".to_string(),
            description: "Snap the rendering viewport camera to a camera entity's position and orientation".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef { name: "entity_id".to_string(), description: "Camera entity ID".to_string(), param_type: ParamType::Integer, required: true, default: None },
            ],
            return_type: ReturnTypeDef { description: "Confirmation".to_string(), return_type: ParamType::Object },
            mode: ToolMode::Editor,
        },
        Arc::new(CameraSnapToHandler),
    );

    // --- Light entity tools ---

    registry.register(
        ToolDefinition {
            name: "light_spawn".to_string(),
            description: "Spawn a point or spot light at a position".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef { name: "light_type".to_string(), description: "Light type: 'point' or 'spot'".to_string(), param_type: ParamType::String, required: true, default: None },
                ParameterDef { name: "x".to_string(), description: "X position".to_string(), param_type: ParamType::Number, required: false, default: Some(serde_json::json!(0.0)) },
                ParameterDef { name: "y".to_string(), description: "Y position".to_string(), param_type: ParamType::Number, required: false, default: Some(serde_json::json!(0.0)) },
                ParameterDef { name: "z".to_string(), description: "Z position".to_string(), param_type: ParamType::Number, required: false, default: Some(serde_json::json!(0.0)) },
            ],
            return_type: ReturnTypeDef { description: "Light ID of the spawned light".to_string(), return_type: ParamType::Object },
            mode: ToolMode::Editor,
        },
        Arc::new(LightSpawnHandler),
    );

    // --- Diagnostic tools ---

    registry.register(
        ToolDefinition {
            name: "voxel_slice".to_string(),
            description: "Sample a 2D XZ slice of raw SDF distances from an object's voxel data. Returns distance grid and slot status for diagnosing sculpt artifacts.".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "object_id".to_string(),
                    description: "Scene object ID (from scene_graph)".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "y_coord".to_string(),
                    description: "Object-local Y coordinate for the slice (default 0.0)".to_string(),
                    param_type: ParamType::Number,
                    required: false,
                    default: Some(serde_json::json!(0.0)),
                },
            ],
            return_type: ReturnTypeDef {
                description: "Grid of SDF distances with origin, spacing, dimensions, and slot status".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(VoxelSliceHandler),
    );

    registry.register(
        ToolDefinition {
            name: "sculpt_apply".to_string(),
            description: "Apply a single sculpt brush hit to an object. Creates an undo entry. Use mode 'add' to build geometry, 'subtract' to carve, 'smooth' to blur.".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef {
                    name: "object_id".to_string(),
                    description: "Scene object ID (from scene_graph)".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "position".to_string(),
                    description: "World-space position [x, y, z] for the brush center".to_string(),
                    param_type: ParamType::Array,
                    required: true,
                    default: None,
                },
                ParameterDef {
                    name: "mode".to_string(),
                    description: "Sculpt mode: 'add', 'subtract', or 'smooth'".to_string(),
                    param_type: ParamType::String,
                    required: false,
                    default: Some(serde_json::json!("add")),
                },
                ParameterDef {
                    name: "radius".to_string(),
                    description: "Brush radius in world units".to_string(),
                    param_type: ParamType::Number,
                    required: false,
                    default: Some(serde_json::json!(0.5)),
                },
                ParameterDef {
                    name: "strength".to_string(),
                    description: "Brush strength (0.0 to 1.0)".to_string(),
                    param_type: ParamType::Number,
                    required: false,
                    default: Some(serde_json::json!(0.5)),
                },
                ParameterDef {
                    name: "material_id".to_string(),
                    description: "Material table index for added geometry".to_string(),
                    param_type: ParamType::Integer,
                    required: false,
                    default: Some(serde_json::json!(1)),
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of sculpt operation".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        },
        Arc::new(SculptApplyHandler),
    );

    registry.register(
        ToolDefinition {
            name: "object_shape".to_string(),
            description: "Return a compact brick-level 3D shape overview. Each brick is categorized as '.' (empty), '#' (interior), or '+' (surface). Per-Y-level ASCII slices provide spatial understanding in one call.".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "object_id".to_string(),
                    description: "Scene object ID (from scene_graph)".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "Object shape with brick grid dims, AABB, counts, and ASCII y-slices".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(ObjectShapeHandler),
    );

    // --- Material tools ---

    registry.register(
        ToolDefinition {
            name: "material_list".to_string(),
            description: "List all materials in the library with slot, name, category, albedo, roughness, metallic, is_emissive".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "Array of material info objects".to_string(),
                return_type: ParamType::Array,
            },
            mode: ToolMode::Both,
        },
        Arc::new(MaterialListHandler),
    );

    registry.register(
        ToolDefinition {
            name: "material_get".to_string(),
            description: "Get full properties of a material at a given slot index".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![
                ParameterDef {
                    name: "slot".to_string(),
                    description: "Material table slot index (0-65535)".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "Full material properties snapshot".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Both,
        },
        Arc::new(MaterialGetHandler),
    );

    // --- Shader tools ---

    registry.register(
        ToolDefinition {
            name: "shader_list".to_string(),
            description: "List available shading models (name, id)".to_string(),
            category: ToolCategory::Observation,
            parameters: vec![],
            return_type: ReturnTypeDef {
                description: "Array of shader info objects".to_string(),
                return_type: ParamType::Array,
            },
            mode: ToolMode::Both,
        },
        Arc::new(ShaderListHandler),
    );

    // --- Scene management tools ---

    registry.register(
        ToolDefinition {
            name: "voxelize".to_string(),
            description: "Convert an analytical primitive object to voxelized geometry-first form. Object must currently be analytical (sphere, box, etc).".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef {
                    name: "object_id".to_string(),
                    description: "Scene object ID (from scene_graph)".to_string(),
                    param_type: ParamType::Integer,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of voxelization".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        },
        Arc::new(VoxelizeHandler),
    );

    registry.register(
        ToolDefinition {
            name: "scene_save".to_string(),
            description: "Save the current scene to disk (.rkscene + .rkf files). If no path given, saves to the last-used path.".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef {
                    name: "path".to_string(),
                    description: "Optional path to save the .rkscene file".to_string(),
                    param_type: ParamType::String,
                    required: false,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of save".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        },
        Arc::new(SceneSaveHandler),
    );

    registry.register(
        ToolDefinition {
            name: "scene_open".to_string(),
            description: "Open a .rkscene file, replacing the current scene.".to_string(),
            category: ToolCategory::Mutation,
            parameters: vec![
                ParameterDef {
                    name: "path".to_string(),
                    description: "Path to the .rkscene file to open".to_string(),
                    param_type: ParamType::String,
                    required: true,
                    default: None,
                },
            ],
            return_type: ReturnTypeDef {
                description: "Confirmation of scene load".to_string(),
                return_type: ParamType::Object,
            },
            mode: ToolMode::Editor,
        },
        Arc::new(SceneOpenHandler),
    );
}
