# WebGPU backend architecture

How `InstancedMesh2WebGPU` decides what to draw, and why it is built this way.
This is a design record, not a specification: each decision states the evidence
behind it and what would make it worth revisiting.

## The problem this solves

Before this work the WebGPU backend replaced only the *storage* layer. Instance
matrices and colors moved from `SquareDataTexture` into `StorageBufferAttribute`,
and the GLSL shader chunks became a TSL graph. Every render-time decision still
ran in JavaScript, inherited unchanged from `src/core/feature/FrustumCulling.ts`:
a linear scan (or BVH traversal) over every instance, per camera, per frame,
decoding matrices, testing spheres, classifying LOD levels and writing a
compacted `Uint32Array` that was then uploaded.

Measured on the WebGPU backend, one main-camera pass, ~26% of instances visible:

| instances | linear scan | BVH traversal | BVH build |
| --------- | ----------- | ------------- | --------- |
| 100k      | 4.6 ms      | 3.0 ms        | 0.3 s     |
| 250k      | 11.6 ms     | 9.2 ms        | 0.9 s     |
| 500k      | 24.3 ms     | 19.5 ms       | 2.1 s     |
| 1M        | 48.6 ms     | 37.3 ms       | 4.9 s     |

Shadow passes repeat the whole scan against their own camera. The BVH only pays
for itself when the visible fraction is small (3.6 ms at 1M with 3% visible) and
costs seconds to build.

## What runs where now

```text
CPU  ── owns logical state ────────────────────────────────────
     instance lifecycle, stable ids, free-list reuse
     matrices, colors, active/visible flags
     BVH  →  raycasting and spatial queries only
     one frustum-plane extraction and a handful of uniforms per pass
                      │
                      ▼  persistent storage buffers
GPU  ── owns render-time decisions ────────────────────────────
     matrices │ instance state │ visible-id lists │ indirect commands
                      │
     reset kernel      → zero every instanceCount for this pass
     cull kernel       → sphere vs 6 planes, LOD classification,
                         atomic append into the level's visible list
                      │
                      ▼
     drawIndexedIndirect, one command per level per material group
```

The CPU work per pass is now independent of instance count: extract six frustum
planes, write a few uniforms, and dispatch. Measured with a stubbed renderer, so
the figure is pure main-thread JavaScript (`test/bench-culling.mjs`):

| instances | CPU path | GPU path | main-thread reduction |
| --------- | -------- | -------- | --------------------- |
| 10k       | 0.52 ms  | 0.04 ms  | 13× |
| 100k      | 5.4 ms   | 0.05 ms  | 112× |
| 500k      | 22.9 ms  | 0.03 ms  | 731× |
| 1M        | 46.3 ms  | 0.24 ms  | 194× |

Submissions matter as much as CPU time, and cost more. That table measures one
mesh; 27 meshes of 5,000 instances, main pass plus one shadow map, costs 0.5 ms
of JavaScript and **2** GPU command submissions.

## Decisions

### Brute force over hierarchy on the GPU

**Chosen.** The cull kernel tests every allocated instance slot against the
frustum. No GPU BVH, no two-phase occlusion.

**Why.** A sphere-versus-six-planes test is a few dozen ALU operations over a
64-byte matrix read. At 1M instances that is one dispatch of ~15,600 workgroups
reading 64 MB — well inside what any WebGPU-capable GPU does in well under a
millisecond, and entirely off the main thread. A hierarchy would add build cost,
update cost on every transform change, and divergent traversal, to save work that
is already close to free.

**Revisit if** instance counts reach the tens of millions, or if profiling on
real hardware shows the matrix read (not the test) dominating — at which point a
separate packed position/radius buffer is the cheaper first move, not a
hierarchy.

### Bounding spheres, not AABBs

**Chosen.** One object-space sphere from the owner's geometry, scaled per
instance by the largest column length of its matrix.

**Why.** It matches what the CPU path already does, so the GPU and CPU paths
cannot disagree about what is visible. A sphere test is six dot products; an
oriented box is far more work for a tighter fit that matters mainly for long thin
geometry. The CPU path has shipped with this conservatism for years.

**Revisit if** a workload has instances whose geometry is extremely elongated and
the over-inclusion becomes measurable.

### Atomic append, not a prefix sum

**Chosen.** Each surviving instance does one `atomicAdd` on its level's
`instanceCount` word and writes its id at the returned slot.

**Why.** It is one kernel and no scratch buffers. A prefix-sum compaction would
give a deterministic, ordered visible list, but needs at least three dispatches
and a scan buffer, and the ordering it buys is not useful here: the draw order of
opaque instances is irrelevant, and anything that *does* need ordering
(`sortObjects`) stays on the CPU path anyway.

**Consequence.** Visible-list order is nondeterministic between frames. This is
why transparent sorting is not available on the GPU path.

**Revisit if** a workload needs deterministic ordering without full CPU sorting —
for example front-to-back ordering to reduce overdraw.

### The indirect buffer is addressed by word, not by struct

**Chosen.** The compute binding is `array<atomic<u32>>` and the kernel touches
word `command * 5 + 1`.

**Why, concretely.** Three's WGSL builder emits a struct-typed storage binding as
a bare struct rather than a runtime-sized array when the attribute holds exactly
one element (`WGSLNodeBuilder.isCustomStruct`). A mesh with no LODs and one
material has exactly one command, so the natural struct formulation fails to
compile with `cannot index type`. Addressing words avoids the heuristic entirely,
and word 1 is `instanceCount` in both the indexed and the non-indexed WebGPU
command layout, so the kernel never needs to know which it is looking at.

**Revisit if** Three changes that heuristic; the struct form would then read a
little better.

### Material arrays get one command per group and a publish kernel

**Chosen.** A mesh with a material array allocates one indirect command per LOD
level *per geometry group*. The cull kernel counts once, into the level's first
command; a third one-invocation kernel copies that count into the level's other
commands.

**Why.** Every group of a level draws the same visible instances but a different
index range, so the counts must agree while `firstIndex`/`indexCount` differ.
Counting into every group with its own atomic would work but costs one extra
atomic per surviving instance per group; the publish kernel costs one dispatch of
one thread. `onBeforeRender` receives the group, so each render object points at
its own command's byte offset.

### Compute is dispatched from `onBeforeRender` / `onBeforeShadow`

**Chosen.** The mesh dispatches its own culling from the render callbacks rather
than requiring the application to orchestrate anything.

**Why it is correct.** `WebGPUBackend.beginCompute` creates its own command
encoder and `finishCompute` submits it immediately, while the enclosing render
pass is submitted later at `finishRender`. Queue submissions execute in order, so
a dispatch issued during a render callback always completes before the draw that
reads its result. This holds for shadow passes too, which Three renders through a
nested `renderer.render()` with its own encoder.

**Why it matters.** `ez-plants`' `PlantField` gets GPU culling without
orchestrating compute or indirect draws itself, which was an explicit goal.

**Cost, and what it turned into.** Three submits a compute group on its own
command buffer, so dispatching per mesh cost one GPU submission per mesh per
pass. `ez-plants`' pooled field reached **54 submissions a frame** across 27
meshes and two passes, which a mobile GPU spends more time scheduling than
culling — the first hardware run of that field was unusably slow while the CPU
side sat at 0.4 ms.

A per-renderer coordinator now batches every mesh's kernels into one
`renderer.compute([...])` per pass: **2 submissions a frame**, and still 2 at 60
meshes. Membership is learned rather than declared — the batch for a pass is
whatever rendered in the previous pass of the same kind, a mesh appearing
outside it dispatches once on its own and joins the next batch, and a disposed
mesh is dropped. A changing scene self-corrects within one frame and the steady
state is always one submission per pass.

**Revisit if** a scene renders many meshes in passes that share a render call.
The coordinator keys only on `info.render.calls`, which is enough because Three
increments it for every pass, each shadow map included.

### The compute graph is rebuilt only when the structure changes

**Chosen.** Each pass keeps its own "needs rebuild" flag, cleared by that pass.

**Why this is called out.** `WebGPUCullingPass.build()` regenerates the TSL graph
and allocates fresh `ComputeNode`s, and Three keys its compute pipelines on node
identity — so rebuilding is a shader compilation, not a bookkeeping update.
The first implementation kept one shared flag and cleared it only on a shadow
pass, so any mesh that never cast a shadow rebuilt both kernels **every frame**.
On hardware that was one frame per second; it never showed up in a single-mesh
benchmark or in the software-WebGPU correctness run, because both still produced
correct pictures. `test/webgpu-culling.test.mjs` now pins zero rebuilds across
steady frames with and without a shadow pass, and exactly one rebuild per pass
after growth.

**Revisit if** a new structural input is added to the descriptor: it belongs in
`sameStructure`, which is the cheap per-frame check, rather than in a flag.

### Per-level visible buffers, capped by the device limit

**Chosen.** Each LOD level keeps its own visible-id storage buffer, exactly as
the CPU path already allocated. The cull kernel unrolls the append over the level
count, and the GPU path is refused when `levels + 3` exceeds
`maxStorageBuffersPerShaderStage` (8 by default, so five levels).

**Why not one merged buffer.** A single `levels * capacity` buffer would need
only one binding and lift the level cap, but it would require rewriting
`getInstanceIndexForPass` and every CPU-path consumer to work through offset
views. The cap it removes is one nobody is against: real LOD ladders are three or
four levels.

**Revisit if** a workload genuinely needs more than five LOD levels, or if a
target device reports a limit of 4 (WebGPU compatibility mode), which would leave
no room for LODs at all.

### LOD selection is data, not a callback

**Chosen.** `resolveLODIndex` (a per-instance JavaScript callback) forces the CPU
path. `setLODOverrideAt(id, level, isShadowPass)` writes the same intent into the
per-instance state word, where the kernel can read it.

**Why.** A callback cannot run in a compute shader, and it was the last thing
keeping `ez-plants`' wood meshes on the CPU. Two 8-bit fields in the state word
(render and shadow, `0` meaning "use the distance test") express the same
decision as GPU-readable state.

**Hysteresis, for the record**, is not per-instance history in this library and
never was: `getObjectLODIndexForDistance` reduces each threshold by `(1 - h)²`.
It is a static threshold bias, which ports to the kernel unchanged. Real
hysteresis would need per-instance previous-level storage; the state word has
room if that is ever wanted.

### `.count` is an upper bound on the GPU path

**Chosen.** `count` becomes the number of active instances — enough to stop
Three's `getDrawParameters` from skipping the draw — and the real instance count
lives in the indirect buffer. `getVisibleCountsAsync()` reads it back on demand.

**Why.** The exact figure is produced on the GPU. Reading it back every frame
would reintroduce a CPU/GPU synchronisation point, which is the cost this whole
change exists to remove.

**Consequence.** `renderer.info` over-reports triangles and instances for these
meshes, and code that read `.count` after `performFrustumCulling()` to learn how
many instances survived now needs `getVisibleCountsAsync()` or the CPU path.

### The CPU path stays, and stays correct

**Chosen.** The shared culling code is the fallback, selected automatically.

It runs when the mesh uses `sortObjects`, `onFrustumEnter` or `resolveLODIndex`,
when a wireframe material needs a CPU-expanded index range, when the level count
exceeds the device's storage-binding budget, or when `culling: 'cpu'` is set. It
is also the only path the WebGL backend has.

This is a genuine fallback rather than leftover code: each condition names a
feature the GPU cannot express, and `culling: 'gpu'` warns once, naming the
reason, when one of them forces the fallback.

### Raycasting does not follow the visible list on the GPU path

**Chosen.** `usesCPUVisibleList()` is `false` while GPU culling is active, which
makes `raycastOnlyFrustum` fall back to scanning every allocated slot.

**Why.** That option walks `instanceIndex` for `count` entries. The GPU path
never compacts that list and `count` is only an upper bound, so following it
would test ids that are not the live ones — with a hole at the front of the
array it reliably returns no hits at all. Scanning every slot is a superset:
`checkObjectIntersection` still filters by the active and visible flags, so the
answer is correct, just not narrowed to the last frame's frustum.

**Revisit if** the visible counts are ever read back for other reasons; the same
readback could narrow this too, at the cost of a frame of latency.

### The BVH is for raycasting now

**Chosen.** `computeBVH()` still builds and still accelerates `raycast()`,
`intersectBox`, `intersectSphere` and application queries. It is simply not in
the render hot path when the GPU path is active.

**Why.** The measurements above: at realistic visible fractions the BVH is within
25% of a linear scan while costing seconds to build and needing maintenance on
every transform change. The GPU beats both by two orders of magnitude, and
raycasting is where a CPU hierarchy still has no competitor.

## Shadows

Each shadow pass runs its own dispatch against its own camera, writing its own
visible lists and its own indirect commands. Main-camera visibility is never
reused: a shadow caster outside the view frustum still has to cast.

Multiple lights work because each shadow map is a separate `renderer.render()`
with its own submission, so the sequence is dispatch-A, draw-A, dispatch-B,
draw-B. The shadow-pass buffers are shared across lights for the same reason the
CPU path could share them.

Three r185 hands `onBeforeShadow` the shadow camera in **both** camera slots, so
the render camera has to be remembered from the preceding render pass. Without
that, shadow LOD levels get picked by distance from the *light*, which puts every
instance on the coarsest shadow level for a light behind the viewer — and
disagrees with the WebGL path, which levels by distance from the camera. So
visibility is the light's decision and level selection is the viewer's, on both
backends.

## Verification

`test/webgpu-culling.test.mjs` and `test/webgpu.test.mjs` cover the CPU-side
contracts: path selection and fallback reasons, state-buffer mirroring and
coalescing, capacity growth, LOD overrides, indirect command layout, dispatch
deduplication across LOD children, and manual culling with `autoUpdate` off.

`test/webgpu-browser.mjs`, driven by `test/run-webgpu-browser.mjs`, is the part
that cannot be faked: it runs the real kernels in a real WebGPU browser and reads
the instance counts back out of the indirect commands. It asserts full
visibility, hidden instances, removed and reused slots, a camera facing away, a
strict partial subset, actual lit pixels from the indirect draw, both groups of a
material array rendering, a mesh under a translated/rotated/scaled parent, LOD
splitting by distance, and a shadow pass classifying independently of the main
camera.

Neither suite measures GPU time. `test/bench-culling.mjs` measures main-thread
cost only, deliberately, with the renderer stubbed.
