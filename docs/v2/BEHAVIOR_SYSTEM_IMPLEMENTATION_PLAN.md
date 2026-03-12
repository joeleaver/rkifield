# Behavior System Implementation Plan

## Context

The engine has no user-facing system for gameplay logic. All behavior is hardcoded in Rust within engine binaries. The behavior system architecture (`BEHAVIOR_SYSTEM_ARCHITECTURE.md`, 14 review passes) specifies: ECS components and systems in a hot-reloadable dylib, a game state store, sequences, blueprints, Play/Stop with two-world isolation, and full editor integration.

**This is additive work.** The engine renders and edits throughout. Phases 0–5 are pure library code (no editor/renderer changes). Phases 6–8 rework existing systems for the unified hecs model. Phases 9+ build the dylib, play/stop, and editor UI.

## Codebase Scope

- **Current state:** 13 crates, all v2 phases + geometry-first complete, 1,680+ tests
- **New crate:** `rkf-macros` (proc macros for `#[component]` and `#[system]`)
- **Major changes:** rkf-runtime (behavior runtime), rkf-render (hecs migration), rkf-editor (edit pipeline, play/stop, inspector)
- **Critical path:** rkf-macros → rkf-runtime → rkf-render migration → rkf-editor integration

## Commit Convention

Commit messages use `behavior-N.M:` prefix (e.g., `behavior-3.2: bidirectional StableId index maps`).

## Dependency Graph

```
Phase 0  (Proc Macro Foundation + Core Types)
    │
Phase 1  (Game State Store + Events)
    │
Phase 2  (SystemContext + CommandQueue + Scheduler)
    │
Phase 3  (StableId + Entity Identity)
    │
Phase 4  (Entity Hierarchy + Naming + Lookup)
    │
Phase 5  (Sequences + Scene Ownership)
    │
Phase 6  (Renderer Migration to hecs)
    │
Phase 7  (.rkscene v3 + Scene I/O)
    │
Phase 8  (Edit Pipeline + Undo/Redo Rework)
    │
Phase 9  (Engine Component Registration)
    │
Phase 10 (Dylib Infrastructure + Hot-Reload)
    │
Phase 11 (Persistence + Save/Load + Blueprints)
    │
Phase 12 (Play/Stop Mode)
    │
Phase 13 (Editor UI Integration)
    │
Phase 14 (MCP Tools + Integration Testing)
```

Phases 0–5 are pure rkf-runtime + rkf-macros work with no editor/renderer impact. The engine remains fully functional throughout — these phases add new code alongside existing systems.

---

## Phase 0: Proc Macro Foundation + Core Types

**Goal:** Create the `rkf-macros` proc macro crate with `#[component]` and `#[system]` attribute macros, and define the core types they generate into. Everything in the behavior system depends on these macros.

### behavior-0.1 — Create rkf-macros proc macro crate
- New crate: `crates/rkf-macros/`
- `Cargo.toml`: `proc-macro = true`, deps on `syn` (full features), `quote`, `proc-macro2`
- Add to workspace `Cargo.toml`
- Stub `lib.rs` with empty macro definitions
- Verify workspace compiles

### behavior-0.2 — Core trait and type definitions in rkf-runtime
- `GameValue` enum: Bool(bool), Int(i64), Float(f64), String(String), Vec3(Vec3), WorldPosition(WorldPosition), Quat(Quat), Color([f32; 4]), List(Vec\<GameValue\>), Ron(String)
- `GameValue` conversion traits: `From<bool>`, `From<i64>`, `From<f64>`, `From<f32>` (→ Float), `From<String>`, `From<&str>`, `From<Vec3>`, `From<WorldPosition>`, `From<Quat>`, `TryFrom<GameValue>` for each concrete type
- `FieldType` enum: Float, Int, Bool, Vec3, WorldPosition, Quat, String, Entity, Enum, List, Color
- `FieldMeta` struct: name, field_type, transient, range: Option<(f64, f64)>, default: Option\<GameValue\>
- `ComponentMeta` trait: `type_name() -> &'static str`, `fields() -> &'static [FieldMeta]`
- `ComponentEntry` struct: name, serialize fn, deserialize_insert fn, remove fn, has fn, get_field fn, set_field fn, meta &[FieldMeta]
- `QueryError` enum: NotFound, Multiple
- Tests: GameValue conversions (all types, round-trip), FieldType coverage, ComponentEntry struct construction

### behavior-0.3 — GameplayRegistry
- `GameplayRegistry` struct: component catalog (`HashMap<String, ComponentEntry>`), system metadata list (`Vec<SystemMeta>`), blueprint catalog (`HashMap<String, Blueprint>`)
- `SystemMeta` struct: name, module_path, phase, after deps, before deps, fn pointer
- `collect_all()`: gathers `inventory`-registered component entries and system metadata
- `register_component(entry: ComponentEntry)` — manual registration (used by engine_register)
- Query methods: `component_names()`, `component_entry(name)`, `system_list()`, `has_component(name)`
- Duplicate component name → error
- Tests: register + query round-trip, duplicate name detection, collect_all with inventory items

### behavior-0.4 — #[component] attribute macro (structs)
- Parse struct definition with `syn`
- Emit derives: `Serialize`, `Deserialize`, `Clone`
- Add `#[serde(default)]` at struct level
- Derive `Default` (unless the struct already has a manual impl — detect via `#[component(no_default)]` opt-out)
- Generate `ComponentMeta` impl: map field types to `FieldType`, detect `#[serde(skip)]` → `transient: true`
- Generate `ComponentEntry` creation function registered via `inventory::submit!`:
  - `serialize`: `world.get::<T>(entity)` → serialize to RON string. For `Entity` fields: look up StableId from world, substitute UUID. Handle `Option<Entity>` (None → null) and `Vec<Entity>` (map each)
  - `deserialize_insert`: RON string → deserialize T, remap StableId UUIDs back to `Entity` handles via index map, `world.insert_one(entity, component)`
  - `remove`: `world.remove_one::<T>(entity)`
  - `has`: `world.satisfies::<&T>(entity).unwrap_or(false)`
  - `get_field(world, entity, field_name)` → `Option<GameValue>`: match on field name, read field, convert to GameValue
  - `set_field(world, entity, field_name, value)` → `Result`: match on field name, convert GameValue to field type, write
  - `meta`: static slice of `FieldMeta` for each non-skip field
- Entity field detection: scan field types for `Entity`, `Option<Entity>`, `Vec<Entity>` — generate remapping code in serialize/deserialize
- Tests: basic struct (fields, meta, round-trip), transient fields (skip detected, not in meta), Entity fields (remapping works), Option\<Entity\> and Vec\<Entity\>, get_field/set_field round-trip for all FieldTypes

### behavior-0.5 — #[component] attribute macro (enums)
- Parse enum definition
- Same derives as struct (Serialize, Deserialize, Clone, Default — default is first variant)
- `ComponentMeta`: `fields()` returns empty slice
- `ComponentEntry`: serialize/deserialize work on whole component as unit. `get_field`/`set_field` return `Err` (not applicable to enums)
- Standard serde enum encoding: unit variants as strings, data variants as tagged structs
- Tests: unit variants, data variants, round-trip serialization, get_field returns error

### behavior-0.6 — #[system] attribute macro
- Parse function with signature `fn(&mut SystemContext)` — error if signature doesn't match
- Required attribute: `phase = Update` or `phase = LateUpdate`
- Optional repeatable attributes: `after = "system_name"`, `before = "system_name"`
- Generate `inventory::submit!` with `SystemMeta` containing: function name, module path, phase, dependency lists, function pointer
- Tests: basic registration, phase parsing, multiple after/before deps, wrong signature → compile error

---

## Phase 1: Game State Store + Events

**Goal:** Implement `GameManager` — the global key-value state store with typed access and fire-and-forget events. This is the cross-entity, cross-scene communication backbone.

### behavior-1.1 — GameManager core store
- `GameManager` struct in rkf-runtime
- Internal: `HashMap<String, GameValue>`
- `set<T: Into<GameValue>>(&mut self, key: &str, value: T)` — immediate write
- `get<T: TryFrom<GameValue>>(&self, key: &str) -> Option<T>` — returns None if key missing or type mismatch
- `remove(&mut self, key: &str) -> Option<GameValue>`
- `remove_prefix(&mut self, prefix: &str)` — remove all keys starting with prefix
- `list(&self, prefix: &str) -> impl Iterator<Item = (&str, &GameValue)>`
- Tests: set/get all GameValue types, remove, remove_prefix (multiple keys), list with prefix filtering, type mismatch returns None, missing key returns None

### behavior-1.2 — RON serialization escape hatch
- `set_ron<T: Serialize>(&mut self, key: &str, value: &T)` — serialize T to RON, store as `GameValue::Ron`
- `get_ron<T: DeserializeOwned>(&self, key: &str) -> Option<T>` — deserialize from stored RON string
- Tests: arbitrary struct round-trip, nested types, error on malformed RON

### behavior-1.3 — Event system
- `Event { name: String, source: Option<Entity>, data: Option<GameValue> }`
- Internal: `Vec<Event>` (frame event buffer)
- `emit(&mut self, name: &str, source: Option<Entity>, data: Option<GameValue>)` — append to buffer, immediately visible
- `events(&self, name: &str) -> impl Iterator<Item = &Event>` — filter by name
- `drain_events(&mut self)` — clear buffer (called by engine at frame end)
- Tests: emit + consume same frame, multiple events same name, filter by name, drain clears all, events from different phases visible

### behavior-1.4 — Save/load infrastructure
- `save_to_ron(&self) -> String` — serialize entire store (HashMap) to RON
- `load_from_ron(&mut self, data: &str) -> Result<()>` — deserialize, overwrite entire store, emit `"state_loaded"` event (source: None, data: None)
- Tests: save/load round-trip preserves all value types, state_loaded event emitted after load, old keys removed on load

---

## Phase 2: SystemContext + CommandQueue + Scheduler

**Goal:** Build the system execution infrastructure — the CommandQueue for deferred structural mutations, SystemContext for the system interface, and the SystemScheduler for phase-based execution with dependency ordering.

### behavior-2.1 — CommandQueue core
- `CommandQueue` struct: pending spawns, despawns, inserts, removes
- `TempEntity` handle: opaque index into pending spawn list
- `spawn(EntityBuilder) -> TempEntity` — queue entity creation
- `despawn(Entity)` — queue entity destruction
- `insert<C: Component>(Entity, C)` — queue component addition on existing entity
- `insert<C: Component>(TempEntity, C)` — batch component onto pending spawn
- `remove<C: Component>(Entity)` — queue component removal
- `flush(&mut self, world: &mut World, stable_id_index: &mut StableIdIndex)` — apply all pending ops:
  - Spawns: create entity, auto-inject Transform (origin) + StableId (new UUID) + EditorMetadata (auto-name) if not provided by user
  - TempEntity inserts applied atomically with spawn
  - Despawns: collect all descendants first (cascading), then despawn all
  - Inserts/removes on existing entities applied
- Tests: spawn + flush creates entity with auto-components, TempEntity batching (multiple inserts), despawn cascading (parent + children), insert/remove on existing, flush order independence

### behavior-2.2 — SystemContext
- `SystemContext` struct: `world: &mut World`, `game: &mut GameManager`, `dt: f32`, `commands: CommandQueue`, `registry: &GameplayRegistry`
- Convenience method stubs for future: `find_one`, `find_path`, `find_relative`, `find_tagged` (delegate to lookup module, implemented in Phase 4)
- Tests: struct construction, field access, commands accessible

### behavior-2.3 — SystemScheduler
- `SystemScheduler` struct
- `build(registry: &GameplayRegistry) -> Result<Self>` — construct from registered system metadata:
  - Group systems by phase
  - Build dependency DAG within each phase from after/before declarations
  - Topological sort — produce execution order
  - Detect cycles → return error naming the involved systems
  - Name resolution: short name if unique across crate, module-qualified if ambiguous (both flagged as requiring qualification)
- `tick(&mut self, world: &mut World, game: &mut GameManager, dt: f32, registry: &GameplayRegistry)`:
  1. Run Phase::Update systems in resolved order, each wrapped in `catch_unwind`
  2. Flush CommandQueue
  3. Run Phase::LateUpdate systems in resolved order, each wrapped in `catch_unwind`
  4. Flush CommandQueue
  5. Drain events
- Panic handling: on catch, log error, mark system as **faulted**, skip on subsequent frames
- `clear_faults(&mut self)` — reset all faulted flags (called on hot-reload)
- Tests: two phases execute in order, dependency ordering respected, cycle detection (error with names), panic recovery (system faulted, others continue), faulted skip + clear, ambiguous name detection

### behavior-2.4 — Frame execution integration
- Integration point in engine frame loop: call `scheduler.tick()` at the correct position
- Pre-tick bridge stub: update `dt` from frame timing
- Post-tick bridge stub: placeholder for store → rinch signals
- Tests: scheduler integrates into frame loop, dt propagates correctly

---

## Phase 3: StableId + Entity Identity

**Goal:** Implement the persistent identity system — every entity gets a UUID that survives save/load, hot-reload, and Play/Stop cloning. This is the foundation for entity reference serialization.

### behavior-3.1 — StableId component
- `StableId(Uuid)` component in rkf-runtime
- Uses `uuid` crate with v4 feature
- Implements `Serialize`, `Deserialize` (UUID string format), `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash`
- NOT registered via `#[component]` macro — engine-managed, no inspector editing, no ComponentEntry
- Tests: creation, serialization round-trip (UUID ↔ string), equality, hashing

### behavior-3.2 — Bidirectional index maps
- `StableIdIndex` struct: `stable_to_entity: HashMap<Uuid, Entity>`, `entity_to_stable: HashMap<Entity, Uuid>`
- `insert(uuid: Uuid, entity: Entity)` — add mapping both directions
- `remove_by_entity(entity: Entity) -> Option<Uuid>` — remove both directions
- `remove_by_stable(uuid: Uuid) -> Option<Entity>` — remove both directions
- `get_entity(uuid: Uuid) -> Option<Entity>`
- `get_stable(entity: Entity) -> Option<Uuid>`
- Integrated into CommandQueue flush: new entities get StableId assigned + indexed
- Despawn removes from index
- Tests: insert + lookup both directions, remove cleans both maps, CommandQueue flush assigns and indexes, despawn removes

### behavior-3.3 — Entity reference serialization helpers
- `serialize_entity(entity: Entity, world: &World) -> Option<Uuid>` — look up StableId
- `deserialize_entity(uuid: Uuid, index: &StableIdIndex) -> Option<Entity>` — resolve
- Used by `#[component]` macro-generated serialize/deserialize code
- Handle missing references: serialize returns None (logged warning), deserialize returns None (logged warning)
- Integration test: component with Entity field → serialize → deserialize into different world → reference resolves correctly via StableId
- Tests: round-trip through serialization, missing entity → None + warning, missing StableId → None + warning

---

## Phase 4: Entity Hierarchy + Naming + Lookup

**Goal:** Rework entity hierarchy (Parent, WorldTransform, naming) and implement the entity lookup API. These are used by gameplay systems to find and navigate entities.

### behavior-4.1 — EditorMetadata rework
- `EditorMetadata { name: String, tags: Vec<String>, locked: bool }`
- Register via `#[component]` macro (engine component, but uses same macro)
- Auto-generated name at CommandQueue flush: "Entity_1", "Entity_2", ... (global counter)
- Tests: creation with defaults, serialization round-trip, auto-naming

### behavior-4.2 — Parent component + WorldTransform
- `Parent(Entity)` component — serialized as StableId in scene files
- `WorldTransform` component — derived each frame, not serialized
- `compute_world_transforms(world: &mut World)`: walk hierarchy, compose `parent.world_transform * child.local_transform`
- Root entities (no Parent): WorldTransform = Transform
- Handles deep hierarchies (iterative, not recursive — avoids stack overflow)
- Integration with existing transform bake pass (replace or extend)
- Tests: single parent, chain of 3+ deep, reparenting updates, root entity identity

### behavior-4.3 — Entity name uniqueness
- At CommandQueue flush: check new entity's name against siblings (entities sharing same Parent, or all root entities if no Parent)
- Auto-suffix on collision: "Guard" → "Guard_2" → "Guard_3"
- Also enforced on rename (editor action, edit pipeline)
- Tests: spawn two with same name → second gets suffix, non-siblings can share names, rename triggers dedup

### behavior-4.4 — Tag indexing
- `TagIndex` struct: `HashMap<String, HashSet<Entity>>`
- Updated on: entity spawn (from EditorMetadata.tags), entity despawn (remove from all tags), tag modification
- `find_tagged(tag: &str) -> impl Iterator<Item = Entity>`
- Tests: index built from spawned entities, despawn removes, tag modification updates, query returns correct set

### behavior-4.5 — Entity lookup API on SystemContext
- `find_one::<Q>(world) -> Result<(Entity, Q), QueryError>` — runs hecs query, returns error if 0 or 2+ matches
- `find_path(world, path: &str) -> Result<Entity, LookupError>` — absolute path, "/" separated, walks hierarchy by name
- `find_relative(world, from: Entity, path: &str) -> Result<Entity, LookupError>` — relative path, "../" goes to parent, otherwise walks children by name
- `find_tagged(tag_index, tag: &str) -> impl Iterator<Item = Entity>` — delegates to TagIndex
- `LookupError` enum: NotFound, Ambiguous, InvalidPath
- Tests: find_one success/NotFound/Multiple, find_path root entity, find_path deep path, find_relative child/sibling (../), find_tagged, error cases

### behavior-4.6 — Position ergonomics
- `WorldPosition: From<Vec3>` — automatic chunk normalization (divide by chunk size, remainder is local)
- `impl Into<WorldPosition>` on all position-accepting APIs (CommandQueue spawn_blueprint, Sequence move_to, etc.)
- `WorldPosition + Vec3 -> WorldPosition` (Add impl)
- `WorldPosition - WorldPosition -> Vec3` (Sub impl, via f64 intermediary for precision)
- Ensure all existing WorldPosition arithmetic is consistent
- Tests: From\<Vec3\> normalizes large values, arithmetic across chunk boundaries, precision at large distances

---

## Phase 5: Sequences + Scene Ownership

**Goal:** Implement the Sequence component (timed multi-step actions) and the SceneOwnership system (entity lifecycle across scene loads). Complete the CommandQueue spawn variants.

### behavior-5.1 — SceneOwnership component
- `SceneOwnership { scene: Option<String> }` — None = persistent, survives scene transitions
- Register via `#[component]` macro
- Tests: creation, serialization, None vs Some

### behavior-5.2 — CommandQueue spawn variants + scene tracking
- `spawn(builder) -> TempEntity` — tags with current scene
- `spawn_persistent(builder) -> TempEntity` — tags with `scene: None`
- `spawn_in_scene(builder, scene: &str) -> TempEntity` — tags with explicit scene
- `spawn_blueprint(name, position) -> TempEntity` — look up blueprint in registry, spawn with current scene
- `spawn_blueprint_persistent(name, position) -> TempEntity` — blueprint, no scene
- `spawn_blueprint_in_scene(name, position, scene) -> TempEntity` — blueprint, explicit scene
- Current scene tracking: `CommandQueue` holds `loaded_scenes: Vec<String>`, set by engine
  - One scene → `spawn()` uses it
  - Multiple scenes → `spawn()` panics with clear error ("multiple scenes loaded, use spawn_in_scene()")
  - Zero scenes → `spawn()` uses None (persistent) with warning
- SceneOwnership auto-tagged at flush
- Tests: all 6 variants, multi-scene error, persistent tagging, zero-scene fallback

### behavior-5.3 — Sequence component + SequenceStep
- `Sequence { steps: Vec<SequenceStep>, current: usize, timer: f32 }`
- `SequenceStep` enum:
  - `Wait { duration: f32 }`
  - `MoveTo { target: WorldPosition, duration: f32, ease: Ease }`
  - `MoveBy { offset: Vec3, duration: f32, ease: Ease }`
  - `RotateTo { target: Quat, duration: f32, ease: Ease }`
  - `RotateBy { rotation: Quat, duration: f32, ease: Ease }`
  - `ScaleTo { target: Vec3, duration: f32, ease: Ease }`
  - `Emit { name: String }`
  - `EmitWith { name: String, data: GameValue }`
  - `EmitFrom { name: String, source: Option<Entity>, data: Option<GameValue> }`
  - `SetState { key: String, value: GameValue }`
  - `SpawnBlueprint { name: String }`
  - `Despawn`
  - `Repeat { count: usize, steps: Vec<SequenceStep> }`
- `Ease` enum: Linear, InQuad, OutQuad, InOutQuad, InCubic, OutCubic, InOutCubic, InExpo, OutExpo, OutBounce, OutElastic, OutBack
- Implements Serialize, Deserialize (Entity in EmitFrom needs remap)
- Tests: all step variants construct, serialization round-trip, Ease enum coverage

### behavior-5.4 — Sequence builder API
- `Sequence::new() -> SequenceBuilder`
- `.wait(seconds)`, `.move_to(impl Into<WorldPosition>, duration)`, `.move_by(Vec3, duration)`, `.rotate_to(Quat, duration)`, `.rotate_by(Quat, duration)`, `.scale_to(Vec3, duration)`
- `.emit(name)`, `.emit_with(name, GameValue)`, `.emit_from(name, Option<Entity>, Option<GameValue>)`
- `.set_state(key, impl Into<GameValue>)`
- `.spawn_blueprint(name)`
- `.despawn()`
- `.repeat(n, |s| ...)` — closure receives sub-builder
- `.ease(Ease)` — modifies the last added step
- `.build() -> Sequence` (or implicit via Into\<Sequence\>)
- Tests: builder produces correct step list, ease modifies last step, repeat nests correctly, chaining works

### behavior-5.5 — sequence_system + execution
- `sequence_system` registered as `#[system(phase = Update)]` — engine-provided, in rkf-runtime
- For each entity with Sequence component:
  - Advance timer by dt
  - When timer exceeds current step's duration: execute step, advance to next
  - Lerp steps: capture start value when step begins (not when Sequence built), interpolate using easing
  - `MoveTo`/`MoveBy`: modify entity's Transform.position
  - `RotateTo`/`RotateBy`: modify Transform.rotation (slerp)
  - `ScaleTo`: modify Transform.scale
  - `Emit`/`EmitWith`/`EmitFrom`: call `game.emit()` with source = this entity (or specified source). Check source entity liveness — substitute None if despawned
  - `SetState`: call `game.set()`
  - `SpawnBlueprint`: call `commands.spawn_blueprint()` — inherits this entity's SceneOwnership
  - `Despawn`: call `commands.despawn(this_entity)`
  - `Repeat`: expand into flat step sequence on first entry (or track repeat counter)
- **Completion:** when all steps finished, remove Sequence component from entity
- Tests: Wait timing, MoveTo interpolation, easing curves (compare a few sample values), emit with source, completion removes component, stale entity handle → None, SpawnBlueprint inherits scene, Despawn queues correctly, Repeat executes N times

---

## Phase 6: Renderer Migration to hecs

**Goal:** Replace `Scene`/`SceneObject` with hecs component queries in the renderer. After this phase, the renderer reads entity data directly from the hecs World. The old data structures are removed.

### behavior-6.1 — SdfTree component
- `SdfTree { root: SceneNode }` component in rkf-runtime (wraps existing SceneNode tree)
- Serialize: voxelized nodes store asset path (e.g., `"guard.rkf"`), not runtime BrickMapHandle
- Deserialize: asset path stored, handle resolved by streaming system on load
- Register via `#[component]` macro
- Tests: construction from existing SceneNode, serialization round-trip, asset path preserved

### behavior-6.2 — Migrate object rendering to hecs
- Replace `Scene::objects` / `SceneObject` iteration with `world.query::<(&Transform, &SdfTree)>()`
- GpuObject metadata buffer built from Transform + SdfTree query results
- Object count, transform upload, brick map references — all from hecs
- BVH construction: iterate hecs query for AABBs instead of Scene vector
- Tile culling: same hecs-sourced object list
- **Visual regression:** render output must be pixel-identical before/after for the same scene data
- Tests: same GpuObject data produced, BVH identical, visual regression (testbed screenshot comparison)

### behavior-6.3 — Migrate light rendering to hecs
- Replace `Scene::lights` with `world.query::<(&Transform, &Light)>()`
- Light buffer upload from hecs query
- Tile light culling from hecs query results
- Tests: same light data produced, shading identical

### behavior-6.4 — Migrate camera to hecs
- Replace direct camera struct access with `world.query::<(&Transform, &CameraComponent)>()`
- Camera controls (editor orbit, fly mode) update the camera entity's Transform
- Main camera selection: entity with `CameraComponent { active: true }` or find_one
- Tests: camera behavior unchanged, controls work

### behavior-6.5 — Remove Scene/SceneObject
- Delete `Scene`, `SceneObject`, and all related data structures
- Remove all code that builds/maintains Scene from entities (currently the editor builds a Scene each frame)
- Clean up imports throughout the workspace
- Verify: `cargo test --workspace` passes, `cargo clippy --workspace` clean
- Tests: workspace compiles, all existing tests pass, editor runs and renders

---

## Phase 7: .rkscene v3 + Scene I/O

**Goal:** Implement the new scene file format — a uniform list of entities, each a StableId with a bag of named components serialized as RON. Replace the existing v2 format.

### behavior-7.1 — Engine component Serialize/Deserialize
- Enable `serde` feature on `glam` dependency
- Implement or derive Serialize/Deserialize for all engine components: Transform, Light, CameraComponent, FogVolumeComponent, EditorMetadata, SdfTree, Parent (serialized as StableId UUID)
- `WorldTransform` explicitly excluded — derived data, never serialized
- Tests: round-trip serialization for each engine component type

### behavior-7.2 — .rkscene v3 format definition
- `SceneFile { version: u32, entities: Vec<EntityRecord> }`
- `EntityRecord { stable_id: Uuid, parent: Option<Uuid>, components: HashMap<String, String> }`
- Component values stored as RON strings (both engine and gameplay)
- Engine components serialized by engine code (knows concrete types)
- Gameplay components serialized via `ComponentEntry.serialize` (type-erased through dylib)
- RON encoding for the outer structure
- Tests: format parses correctly, sample scene validates

### behavior-7.3 — Scene save
- `save_scene(world: &World, registry: &GameplayRegistry, stable_index: &StableIdIndex) -> SceneFile`
- Iterate all entities in world
- For each: read StableId, read Parent (convert to StableId), serialize each component
- Engine components: serialize directly (match on known type names)
- Gameplay components: iterate registry entries, call `ComponentEntry.serialize` for each that `has` returns true
- Write to `.rkscene` file as RON
- Tests: save produces valid RON, all component types included, parent references are UUIDs

### behavior-7.4 — Scene load
- `load_scene(scene: &SceneFile, world: &mut World, registry: &GameplayRegistry, stable_index: &mut StableIdIndex)`
- Clear world (or load additively — controlled by caller)
- For each EntityRecord: create entity, assign StableId, index it
- Deserialize engine components directly
- Deserialize gameplay components via `ComponentEntry.deserialize_insert`
- Resolve Parent references: StableId UUID → Entity via stable_index
- Tests: load produces correct hecs state, entity references resolved, all component types restored

### behavior-7.5 — Unknown component preservation
- If a component name has no matching `ComponentEntry` in registry: store raw RON string in `UnknownComponents` component on the entity
- `UnknownComponents { data: HashMap<String, String> }` — maps component name → RON
- On save: unknown components re-emitted as-is (data not lost)
- On reload: if component type becomes available, attempt deserialization from stored RON
- Tests: unknown component survives save/load round-trip, re-added type restores from preserved data

### behavior-7.6 — Backward compatibility with .rkscene v2
- Detect version field on load
- v2 loader: convert old format (separate objects/lights/cameras sections) into v3 EntityRecord format
- One-way migration — save always writes v3
- Tests: existing v2 .rkscene files load correctly into new system

---

## Phase 8: Edit Pipeline + Undo/Redo Rework

**Goal:** Unify all edit operations through a single pipeline that writes to the edit world, captures undo, and pokes rinch signals. Replace the current ad-hoc undo system.

### behavior-8.1 — Edit pipeline
- `EditPipeline` struct: receives edit operations, applies to edit world, triggers rinch signal updates
- Edit operation types:
  - `SetProperty { entity, component_name, field_name, value: GameValue }`
  - `AddComponent { entity, component_name, initial_data: String (RON) }`
  - `RemoveComponent { entity, component_name }`
  - `SpawnEntity { builder, parent: Option<Entity> }`
  - `DespawnEntity { entity }`
  - `GeometryEdit { entity, edit_data }` (wraps existing sculpt/paint edit types)
- `apply(&mut self, op: EditOp, world: &mut World, registry: &GameplayRegistry)` — captures undo, then applies
- Tests: each operation type modifies world correctly, rinch signals poked

### behavior-8.2 — Undo/redo stack
- `UndoStack` struct: `actions: Vec<UndoAction>`, `redo: Vec<UndoAction>`
- `UndoAction` enum:
  - `PropertyChange { entity, component_name, field_name, old_value: GameValue, new_value: GameValue }`
  - `ComponentAdd { entity, component_name, data: String (RON) }`
  - `ComponentRemove { entity, component_name, data: String (RON) }`
  - `EntitySpawn { entity, all_components: Vec<(String, String)>, stable_id: Uuid, parent: Option<Uuid> }`
  - `EntityDespawn { entity, all_components: Vec<(String, String)>, stable_id: Uuid, parent: Option<Uuid> }`
  - `GeometryEdit { entity, before_snapshot, after_snapshot }`
- `undo(world, registry)` — pop action, apply reverse, push to redo
- `redo(world, registry)` — pop from redo, apply forward, push to undo
- New edit clears redo stack
- `clear()` — empty both stacks (called on scene load)
- Tests: undo/redo for each action type, redo clearing on new edit, clear empties both

### behavior-8.3 — Coalescing
- `begin_coalesce(&mut self)` — start accumulating (called on mouse-down/drag-start)
- `end_coalesce(&mut self)` — finalize: merge accumulated edits into single UndoAction (called on mouse-up)
- Between begin/end: multiple SetProperty ops on same entity+field merge (keep first old_value, last new_value)
- Tests: gizmo drag (many SetProperty) produces single undo action, undo restores to pre-drag state

### behavior-8.4 — Route existing tools through pipeline
- Gizmo transforms → `SetProperty` through pipeline
- Inspector field edits → `SetProperty` through pipeline
- Sculpt operations → `GeometryEdit` through pipeline
- Paint operations → `GeometryEdit` through pipeline
- MCP mutation tools → appropriate EditOp through pipeline
- Remove direct hecs writes from all tools
- Clear undo stack on scene load
- Tests: each tool produces undo-able actions, full round-trip (edit → undo → redo)

---

## Phase 9: Engine Component Registration

**Goal:** Engine components (Transform, Light, etc.) register into the same `GameplayRegistry` as gameplay components, making inspector, MCP, and scene I/O uniform.

### behavior-9.1 — Engine components use #[component]
- Apply `#[component]` attribute macro to: Transform, Light, CameraComponent, FogVolumeComponent, EditorMetadata
- SdfTree uses #[component] but may need custom serialize/deserialize (asset path resolution)
- Parent is special-cased (Entity reference → StableId, engine handles directly)
- StableId remains unregistered (engine-managed internal)
- Verify: ComponentMeta generated correctly, get_field/set_field work for each
- Tests: each engine component has correct FieldMeta, get_field/set_field round-trip

### behavior-9.2 — engine_register() startup step
- `engine_register(registry: &mut GameplayRegistry)` function in rkf-runtime
- Runs at editor startup, before any dylib loads
- Collects engine component registrations via `inventory` from the static binary
- Populates GameplayRegistry with engine ComponentEntry items
- Tests: registry contains all engine components after call

### behavior-9.3 — Uniform treatment verification
- Inspector renders engine components via ComponentMeta (same as gameplay would be)
- Scene save/load uses ComponentEntry for engine components (same code path as gameplay)
- MCP component_get/component_set works on engine components
- Tests: integration tests confirming uniform behavior for engine components

---

## Phase 10: Dylib Infrastructure + Hot-Reload

**Goal:** Implement the dynamic library loading system — compile game code as a cdylib, load it at runtime, and hot-swap it when code changes. This is the core iteration loop for gameplay development.

### behavior-10.1 — DylibLoader
- `DylibLoader` struct in rkf-runtime, wraps `libloading::Library`
- `load(path: &Path) -> Result<Self>` — open cdylib
- `call_register(&self, registry: &mut GameplayRegistry) -> Result<()>` — resolve `register` symbol, call it
- `unload(self)` — drop library (dlclose)
- Tests: load test dylib, call register, verify registry populated, unload cleanly

### behavior-10.2 — ABI version checking
- Build script (`build.rs`) in rkf-runtime: embed `rustc --version` hash as const
- Dylib exports matching const (generated by rkf-macros or template)
- On load: compare engine hash vs dylib hash
- Mismatch → reject with error: "Game code compiled with rustc X.Y.Z but editor requires A.B.C"
- Tests: matching versions accepted, intentional mismatch rejected with clear error

### behavior-10.3 — File watcher + background build
- Watch game crate `src/` directory via `notify` crate (debounced, 200ms)
- On `.rs` file change: spawn `cargo build -p <game-crate> --lib` as child process
- Capture stdout/stderr
- Report build state: `BuildState::Idle | Compiling | Success(path) | Error(String)`
- Parse rustc error output for file/line information
- Editor status bar integration point (consumed in Phase 13)
- Tests: watcher detects file changes, build spawns and completes, errors captured with line info

### behavior-10.4 — Hot-reload cycle
- `reload(world: &mut World, registry: &mut GameplayRegistry, stable_index: &StableIdIndex, old_loader: DylibLoader, new_path: &Path) -> Result<DylibLoader>`
- Steps:
  1. For each gameplay component type in registry (using OLD function pointers):
     - Iterate entities that `has` this component
     - `serialize(world, entity)` → store `(entity, component_name, ron_string)`
     - `remove(world, entity)` — drop runs in old dylib
  2. `old_loader.unload()`
  3. `new_loader = DylibLoader::load(new_path)?`
  4. Verify ABI hash
  5. Clear registry gameplay entries (keep engine entries)
  6. `new_loader.call_register(registry)?`
  7. For each stored `(entity, component_name, ron_string)`:
     - Look up component_name in new registry
     - `deserialize_insert(world, entity, &ron_string)`
     - On failure: log warning, skip (entity keeps other components)
  8. Clear scheduler fault flags
- Tests: reload preserves component data, added field gets default, removed field dropped, renamed type logged, fault flags cleared

### behavior-10.5 — Game crate scaffolding
- `scaffold_game_crate(project_dir: &Path) -> Result<PathBuf>` — creates minimal game crate if none exists
- Template files:
  - `Cargo.toml`: cdylib, rkf-runtime dependency
  - `src/lib.rs`: automod::dir!, register() with collect_all()
  - `src/components/` (empty directory with `.gitkeep`)
  - `src/systems/` (empty directory with `.gitkeep`)
  - `src/blueprints/` (empty directory)
- Tests: scaffolded crate compiles, register() succeeds, empty component/system lists

### behavior-10.6 — Reload timing + queue
- Reload only applies in edit mode
- If build completes during play: store new dylib path as `pending_reload`
- On Stop (play → edit): if pending_reload exists, apply it immediately
- Status reporting: "Reload pending (in play mode)"
- Tests: reload blocked during play, queued reload applies on Stop

---

## Phase 11: Persistence + Save/Load + Blueprints

**Goal:** Implement `#[persist]` for component↔store sync, engine-automated save/load, and the blueprint system for reusable entity templates.

### behavior-11.1 — #[persist] annotation in macro
- Extend `#[component]` macro to parse `#[persist("key/pattern")]` on fields
- Generate `Persistable` trait impl:
  - `sync_to_store(world, entity, game)` — for each `#[persist]` field: read field, expand `{stable_id}`, call `game.set(key, value)`
  - `sync_from_store(world, entity, game)` — for each `#[persist]` field: expand key, call `game.get(key)`, if Some write to field
- `{stable_id}` expands to entity's StableId UUID string at sync time
- Supported field types: all GameValue-compatible types (bool, i64, f64, String, Vec3, WorldPosition, Quat, Color, and arbitrary via Ron)
- Missing key → field keeps current value (no error)
- Tests: sync_to_store writes correct keys, sync_from_store reads back, {stable_id} expansion, missing key preserves value, all supported types

### behavior-11.2 — SystemContext sync methods
- `ctx.sync_to_store::<T: Persistable>(entity: Entity)` — delegates to T's Persistable impl
- `ctx.sync_from_store::<T: Persistable>(entity: Entity)` — delegates to T's Persistable impl
- Requires `&World` (to read component + StableId) and `&mut GameManager` (to read/write store)
- Tests: called from mock SystemContext, correct delegation

### behavior-11.3 — Engine-automated save/load
- Engine writes reserved keys each frame:
  - `engine/scenes/loaded` → List of loaded scene names
  - `engine/camera/position` → WorldPosition
  - `engine/camera/rotation` → Quat
  - `engine/camera/fov` → Float
- On `load_from_ron`:
  1. Overwrite store
  2. Read `engine/` keys
  3. Compare loaded scenes vs currently loaded → unload extras, load missing
  4. Restore camera position/rotation/fov
  5. Emit `"state_loaded"` event (visible to gameplay systems that run later)
- Restoration happens in engine code, before gameplay systems see the event
- Tests: engine keys written each frame, load triggers scene restore, camera restored, event emitted after restore

### behavior-11.4 — Blueprint file format
- `.rkblueprint` RON file:
  ```
  Blueprint(
      name: "Guard",
      components: {
          "Health": "(current: 100.0, max: 100.0)",
          "Patrol": "(waypoints: [], speed: 2.5)",
          "EditorMetadata": "(name: \"Guard\", tags: [\"enemy\"], locked: false)",
          "SdfTree": "(root: ...)",
      },
  )
  ```
- Excludes: Transform, WorldTransform, Parent, StableId, SceneOwnership
- `Blueprint` struct: name, components: HashMap\<String, String\>
- Parse/validate on load
- Tests: parse valid blueprint, reject invalid, excluded components stripped

### behavior-11.5 — Blueprint catalog + loading
- `GameplayRegistry` holds `blueprint_catalog: HashMap<String, Blueprint>`
- `scan_blueprints(dir: &Path)` — scan directory for `.rkblueprint` files, parse and add to catalog
- File watcher: reload individual blueprints on change (reparse file, update catalog entry)
- `registry.blueprint(name) -> Option<&Blueprint>`
- Tests: scan populates catalog, file change triggers reload, query works

### behavior-11.6 — Blueprint instantiation
- CommandQueue `spawn_blueprint` variants:
  1. Look up blueprint in registry by name → error if not found
  2. Create EntityBuilder
  3. For each component in blueprint: deserialize from RON via ComponentEntry.deserialize_insert (after entity created)
  4. Set Transform.position from provided position argument
  5. Tag with appropriate SceneOwnership
  6. Return TempEntity
- Sequence `.spawn_blueprint(name)`: inherits owning entity's SceneOwnership (not "current scene")
- Asset resolution: if blueprint includes SdfTree with asset path, the streaming system resolves it after entity materializes (same as scene load)
- Tests: spawn from blueprint creates entity with correct components, position set, scene ownership correct, asset path handed to streaming

### behavior-11.7 — Editor blueprint actions
- **Save as Blueprint**: select entity → right-click → "Save as Blueprint" → serialize entity's components (minus exclusions) to `.rkblueprint` file, add to catalog
- **Create from Blueprint**: menu lists catalog entries → selecting one spawns entity at camera look-at point via edit pipeline (undoable)
- Tests: save creates valid file, instantiate creates entity with correct data, undo removes spawned entity

---

## Phase 12: Play/Stop Mode

**Goal:** Implement the two-world Play/Stop system — clone the edit world for play, run gameplay systems on the clone, discard on stop. This is the core testing loop.

### behavior-12.1 — Play world clone
- `clone_world(edit: &World, registry: &GameplayRegistry, stable_index: &StableIdIndex) -> (World, StableIdIndex)`
- Create new empty World (play world) and new StableIdIndex
- Iterate edit world entities:
  1. Create corresponding entity in play world
  2. Assign same StableId → index in play StableIdIndex
  3. Build entity mapping: `edit_entity → play_entity` (via StableId)
- Engine components: clone directly from edit world, remap Entity references (Parent, etc.) using the mapping
- Gameplay components: for each registered component on entity:
  - `serialize(edit_world, entity)` → RON (Entity refs become StableId UUIDs)
  - `deserialize_insert(play_world, play_entity, ron)` → StableId UUIDs resolve to play world entities via play StableIdIndex
- Tests: all entities cloned, StableIds match, engine component data matches, gameplay component data matches, entity references (Parent, gameplay Entity fields) point to play world entities

### behavior-12.2 — Store snapshot/restore
- `GameManager::snapshot(&self) -> StoreSnapshot` — deep clone of key-value store (not events)
- `GameManager::restore(&mut self, snapshot: StoreSnapshot)` — overwrite store, clear events
- Tests: snapshot captures all keys, restore overwrites all keys, events cleared

### behavior-12.3 — Renderer switching
- Editor holds `active_world: ActiveWorld` enum (Edit, Play)
- Renderer receives reference to active world each frame
- Switch to play world on Play, switch to edit world on Stop
- Tests: renderer queries correct world after switch

### behavior-12.4 — Frame execution integration
- Full frame loop during play mode:
  1. Input polling, window events
  2. Pre-tick bridge: update dt, read UI rinch signals → inject into store/events
  3. `scheduler.tick(play_world, game, dt, registry)` — runs all systems on play world
  4. Post-tick bridge: push store values → rinch signals
  5. Transform bake on play world → BVH refit → render
- During edit mode: skip steps 2–4, only run edit loop (gizmo, tools, etc.)
- Tests: play frame executes systems, store changes reflect in post-tick, edit frame skips systems

### behavior-12.5 — Edit tool disabling during play
- Gizmo, sculpt, paint, inspector property editing: disabled during play mode
- Input routing: play mode routes input to gameplay (future input system), not to editor tools
- Menu items grayed out or hidden
- Tests: tool operations rejected during play, re-enabled on stop

### behavior-12.6 — Stop: discard + restore
- Stop sequence:
  1. Stop running systems (scheduler no longer ticked)
  2. Drop play world entirely
  3. Restore store from snapshot
  4. Switch renderer to edit world
  5. Re-enable edit tools
  6. If pending hot-reload: apply it now (see Phase 10.6)
- Tests: stop restores exact pre-play state (store values, edit world unchanged), pending reload applies

### behavior-12.7 — Scene transitions during play
- `load_scene_play(world: &mut World, path: &str, registry: &GameplayRegistry, stable_index: &mut StableIdIndex)` — deserialize .rkscene into play world
- `unload_scene_play(world: &mut World, scene_name: &str)` — despawn all entities with `SceneOwnership { scene: Some(scene_name) }`
- Persistent entities (`scene: None`) survive transitions
- Edit world untouched throughout
- API available via SystemContext (specific method TBD — likely `ctx.commands.load_scene(path)` / `ctx.commands.unload_scene(name)` as deferred commands)
- Tests: load adds entities to play world, unload removes scene-owned entities, persistent entities survive, edit world unchanged

### behavior-12.8 — Copy-on-write brick pool
- Play world SdfTree components reference same brick pool data as edit world (shared by default)
- `BrickPoolManager` tracks ownership: edit-owned vs play-owned bricks
- On runtime geometry modification during play:
  1. Check if brick is edit-owned
  2. If so: allocate new brick slot, copy data, update play world's reference
  3. Modify the play-owned copy
- On Stop: free all play-owned brick slots (bulk deallocation)
- Tests: shared bricks read correctly from play world, CoW on modification allocates new slot, edit world data unchanged, Stop frees play bricks

### behavior-12.9 — Push to edit
- Play mode UI shows "push to edit" button next to individual property values
- Click: read value from play world (via get_field), write to edit world (via edit pipeline SetProperty)
- Goes through edit pipeline → captured in undo stack
- Tests: value transfers from play to edit world correctly, undo reverses the push

---

## Phase 13: Editor UI Integration

**Goal:** Rework the editor UI to be data-driven from ComponentMeta. Add the compiler integration, systems panel, and play mode UI.

### behavior-13.1 — Inspector rework (ComponentMeta-driven)
- Replace hardcoded property panels with generic ComponentMeta-driven rendering
- For each component on selected entity: read ComponentMeta, render fields
- Field rendering by FieldType:
  - Float/Int: slider (if FieldMeta.range set) or text input
  - Bool: checkbox
  - Vec3/WorldPosition/Quat/Color: multi-field editors
  - String: text input
  - Entity: tree picker (click to select from hierarchy)
  - Enum: dropdown (variant selector), with data fields for data variants
  - List: expandable list with add/remove
- Transient fields (`transient: true`): hidden in edit mode, shown grayed in play mode
- All edits → edit pipeline (SetProperty) → undoable
- Tests: all field types render correctly, edits produce correct EditOps

### behavior-13.2 — Add/Remove Component
- **Add Component** dropdown below component list
- Lists all registered components (engine + gameplay) minus those already on entity
- Selecting adds component with `Default::default()` values via edit pipeline (ComponentAdd) → undoable
- **Remove** button on each component header (except mandatory: Transform, StableId, EditorMetadata)
- Remove via edit pipeline (ComponentRemove) → undoable
- Tests: add component appears on entity, remove component disappears, undo/redo both, list excludes existing

### behavior-13.3 — Systems panel
- Debug panel listing all registered systems
- Grouped by phase (Update, LateUpdate)
- Shows resolved execution order (from scheduler's topological sort)
- Per-system frame time (measured during scheduler.tick)
- Faulted system indicator (red/warning icon)
- Tests: panel populates from registry, timing displayed, faulted systems marked

### behavior-13.4 — Compiler integration UI
- Status bar segment: Idle → "Compiling..." → "Reloaded" / "Error: ..."
- Error console panel: shows compile errors with file/line/column (clickable to open in external editor)
- Build triggered by file watcher (Phase 10.3)
- Editor remains fully interactive during compilation
- Tests: status transitions display correctly, errors shown with location

### behavior-13.5 — Play mode UI
- Separate panel activated during play mode, replacing the edit inspector
- Reads component values from play world (not edit world)
- Displays same ComponentMeta-driven fields but read-only or play-editable
- "Push to edit" button next to each field value
- Cannot add/remove components during play
- Tests: play UI reads play world, push-to-edit button triggers correct action

---

## Phase 14: MCP Tools + Integration Testing

**Goal:** Expose the behavior system to AI agents via new MCP tools. Build an example game crate. Run comprehensive integration tests.

### behavior-14.1 — Component MCP tools
- `component_list` — list all registered components (name + field metadata for each)
- `component_get(entity_id, component_name)` — read all fields as key-value pairs (via get_field)
- `component_set(entity_id, component_name, fields: {name: value, ...})` — partial update (via set_field per field)
- `component_add(entity_id, component_name, fields: {name: value, ...})` — add component with values
- `component_remove(entity_id, component_name)` — remove component
- All mutations go through edit pipeline (undoable)
- Add AutomationApi methods, implement in EditorAutomationApi, forward in BridgeAutomationApi
- Tests: all tools work via MCP, mutations undoable

### behavior-14.2 — System + blueprint MCP tools
- `system_list` — registered systems with phase, order, timing, faulted status
- `blueprint_list` — available blueprints with component names
- `blueprint_spawn(name, position)` — instantiate blueprint, returns entity id
- Add AutomationApi methods + bridge forwarding
- Tests: list returns correct data, spawn creates entity

### behavior-14.3 — State MCP tools
- `state_get(key)` — read value from store
- `state_set(key, value, type)` — write value to store
- `state_list(prefix)` — list keys with optional prefix filter
- Tests: CRUD via MCP, prefix filtering

### behavior-14.4 — Play control MCP tools
- `play_start` — enter play mode (clones world, starts systems)
- `play_stop` — exit play mode (discards play world, restores store)
- Returns current mode status
- Tests: mode transitions via MCP, state verified after each

### behavior-14.5 — Example game crate
- `examples/example-game/` — sample game crate demonstrating the full behavior system
- Components:
  - `Health { current: f32, max: f32 }` with `#[persist]`
  - `Patrol { waypoints: Vec<Vec3>, speed: f32, current_index: usize (transient) }`
  - `Collectible { value: i32, spin_speed: f32 }`
  - `DoorState { open: bool, key_required: Option<String> }`
  - `GuardState` enum: Patrolling, Alerted, Chasing, Attacking, Returning
- Systems:
  - `patrol_system` — move along waypoints
  - `guard_ai_system` — state machine transitions, spawns Sequences for animations
  - `door_system` — listens for "interact" events, checks keys in store
  - `death_system(after = "combat_system")` — checks Health ≤ 0, despawns
  - `collectible_system` — spin animation, emit "collected" event on proximity
  - `camera_follow_system(phase = LateUpdate)` — follows player
- Blueprints: `guard.rkblueprint`, `collectible.rkblueprint`
- Demonstrates: queries, events, store, sequences, ordering, push-to-edit, save/load
- Tests: example game compiles and loads, systems execute without panics

### behavior-14.6 — End-to-end integration testing
- **Full workflow test:** create component file → file watcher triggers build → hot-reload → component appears in inspector → configure values → Play → systems tick → Stop → values restored
- **Reload test:** modify component field (add/remove/rename) → reload → data preserved/defaulted correctly
- **Play isolation test:** modify entities during play → Stop → edit world unchanged
- **Save/load test:** persist component → save game → modify store → load game → values restored, scene transitions correct
- **Stress test:** 1000 entities with components, 20 systems → measure frame time overhead of scheduler
- **MCP workflow test:** agent-driven scenario (spawn blueprint via MCP, set components, play, screenshot, stop)
- Tests: all scenarios pass, frame time overhead < 1ms for scheduler with 20 systems

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 0 | 6 | Proc macro foundation + core types |
| 1 | 4 | Game state store + events |
| 2 | 4 | SystemContext + CommandQueue + Scheduler |
| 3 | 3 | StableId + entity identity |
| 4 | 6 | Entity hierarchy + naming + lookup |
| 5 | 5 | Sequences + scene ownership |
| 6 | 5 | Renderer migration to hecs |
| 7 | 6 | .rkscene v3 + scene I/O |
| 8 | 4 | Edit pipeline + undo/redo rework |
| 9 | 3 | Engine component registration |
| 10 | 6 | Dylib infrastructure + hot-reload |
| 11 | 7 | Persistence + save/load + blueprints |
| 12 | 9 | Play/Stop mode |
| 13 | 5 | Editor UI integration |
| 14 | 6 | MCP tools + integration testing |
| **Total** | **79** | |
