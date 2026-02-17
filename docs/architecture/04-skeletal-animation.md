# Skeletal Animation

> **Status: DECIDED**

Novel approach: Segmented + Joint Rebaking. No known prior art for this specific technique in SDF engines.

### Decision: Segmented + Joint Rebaking

**Chosen over:** Runtime space warping (per-step bone eval too expensive, Lipschitz violations), per-frame full rebake (too expensive for many characters), pure rigid segments (robotic joints, no smooth skin).

Characters are decomposed into **rigid segments** (one per major body part) and **joint regions** (small blending volumes at articulation points). Segments are transformed rigidly by their bone. Joints are rebaked each frame from overlapping segments using smooth-min blending.

**Character decomposition:**
```
  [Head]──(neck)──[Torso]──(shoulder)──[Upper Arm]──(elbow)──[Lower Arm]──(wrist)──[Hand]
                     │
                  (hip)
                     │
            [Upper Leg]──(knee)──[Lower Leg]──(ankle)──[Foot]

  Segments: Rigid transform, rest-pose SDF, cheap
  Joints:   Rebaked each frame, smooth blending, small volume
```

**Segment:** A collection of bricks in the brick pool storing the rest-pose SDF. Rigidly attached to one bone. At render time, the ray marcher transforms the ray into the segment's local space (inverse bone matrix) and evaluates the rest-pose SDF. Lipschitz is preserved — rigid transforms don't change distances.

**Joint region:** A small volume (~10-20cm radius) around each articulation point where adjacent segments overlap. Rebaked each frame as a compute shader pass:

```
For each joint (compute shader):
  For each voxel in the joint's brick region:
    1. Transform voxel world pos → segment A local space → evaluate SDF_A
    2. Transform voxel world pos → segment B local space → evaluate SDF_B
    3. result = smooth_min(SDF_A, SDF_B, blend_radius)
    4. material = from whichever segment is closer
    5. Write result to joint's bricks in the brick pool
```

**Smooth-min blending:**
```wgsl
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}
```
Parameter `k` is per-joint configurable: small (~0.02) for mechanical joints, large (~0.08) for organic skin. Shoulder blends wider than finger joints.

**Per-frame animation pipeline (compute):**

```
For each animated character:
  1. CPU: Evaluate animation keyframes → bone matrices
  2. CPU: Upload bone matrices to GPU uniform buffer
  3. GPU: Update spatial index (transform segment bounding boxes)
  4. GPU: Rebake joint regions (smooth-min blend from adjacent segments)
  5. GPU: Apply blend shapes to head/face region
```

**Performance — joint rebaking:**
- ~15-20 joints per character × ~5-10 bricks each = ~100-200 bricks
- 100-200 bricks × 512 voxels × 2 SDF evaluations = ~100K-200K SDF evals
- Small compute dispatch — easily under 1ms per character on modern GPUs

**Performance — ray marching:**
- No bone evaluation during ray marching at all
- Rigid segments: one matrix multiply when DDA enters the segment region
- Rebaked joints: evaluated exactly like static geometry
- Animated characters cost the same as static geometry of equivalent size

### Decision: Blend Shapes for Facial Animation

Blend shapes are **additive distance offsets** to the base SDF:

```
final_distance = base_distance
    + weight_smile * smile_delta
    + weight_blink * blink_delta
    + weight_frown * frown_delta
    + ...
```

Each blend shape is a **delta-SDF** stored as sparse bricks covering only the affected region.

```rust
struct BlendShape {
    name: String,
    weight: f32,                    // 0.0-1.0, animated per frame
    brick_offset: u32,              // into blend shape brick pool
    brick_count: u32,
    bounding_box: BoundingBox,
}
```

Applied during head segment rebaking — the head is always rebaked (not rigid) to support facial animation. Only active blend shapes (non-zero weight) are evaluated.

**Storage:** ~30 blend shapes × ~10 bricks × 4KB = ~1.2MB per character face. Trivial.

### Decision: Lipschitz Mitigation — Conservative Step Multiplier

Smooth-min blending can produce slightly non-Lipschitz SDF values in joint regions. Mitigation: joint bricks carry a flag telling the ray marcher to use a 0.8× step multiplier. The marcher takes slightly smaller steps in joints, avoiding overshoots.

```wgsl
let step = if brick.flags & JOINT_REGION != 0 {
    d * 0.8  // conservative step in joint regions
} else {
    d        // full step elsewhere
};
```

Simple, robust, costs ~20% more steps only in the small joint volumes.

**Upgrade path — Eikonal redistancing:**
After joint rebaking, run 1-2 iterations of fast marching / redistancing to correct SDF values. More mathematically correct. Only needed if conservative stepping produces visible over-stepping artifacts or performance issues.

### Decision: Spatial Index for Animated Characters

Animated characters add two entry types to the spatial index:

| Entry Type | Ray Marcher Behavior |
|-----------|---------------------|
| `RIGID_SEGMENT` | Transform ray by inverse bone matrix, evaluate rest-pose bricks |
| `REBAKED_JOINT` | Evaluate bricks directly in world space (already rebaked) |
| (existing) `SURFACE` | Evaluate bricks directly (static geometry) |

The type check happens once per DDA region entry, not per sphere-trace step.

### Character Complexity Limits

| Parameter | Limit | Rationale |
|-----------|-------|-----------|
| Bones per skeleton | ~128 | Standard game complexity. Uniform buffer. |
| Segments per character | ~20-30 | Major body parts |
| Joint regions per character | ~15-25 | Small bones can share joints |
| Active blend shapes | ~30-50 | FACS standard for facial animation |
| Simultaneous animated characters | ~20-50 | Depends on brick pool budget |

**Memory per animated character:**

| Component | Bricks | Memory |
|-----------|--------|--------|
| Segments (rest-pose SDF) | 500-2000 | 2-8MB |
| Joint rebake workspace | 100-200 | 0.4-0.8MB |
| Blend shape deltas | 100-300 | 0.4-1.2MB |
| **Total** | | **~3-10MB** |

### Decision: Cloth and Soft Body — Deferred

Out of scope for initial design. The architecture supports future extension.

**Upgrade path — Cloth simulation:**
Run PBD (position-based dynamics) on GPU. Voxelize deformed cloth surface into region bricks each frame. Same pipeline as joint rebaking, driven by physics instead of bones.

**Upgrade path — Soft body (jiggle physics):**
Model secondary motion as procedural blend shapes. A spring system per soft-body region outputs blend shape weights driven by velocity/acceleration. Cheap physics, reuses existing blend shape infrastructure.

### Session 5 Summary: All Skeletal Animation Decisions

| Decision | Choice | Upgrade Path |
|----------|--------|--------------|
| Animation approach | Segmented + joint rebaking | — (novel technique) |
| Segment transform | Rigid, one bone per segment | — |
| Joint blending | Smooth-min, per-joint k parameter | Eikonal redistancing |
| Facial animation | Additive blend shape delta-SDFs | — |
| Lipschitz mitigation | 0.8× conservative step in joints | Redistancing pass |
| Bone count | ~128 per skeleton | — |
| Character budget | ~20-50 simultaneous, 3-10MB each | Increase brick pool |
| Cloth/soft body | Deferred | PBD cloth; jiggle blend shapes |
