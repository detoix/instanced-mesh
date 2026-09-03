import {
  BufferAttribute,
  type BufferGeometry,
  type Camera,
  Material,
  type Mesh,
  type Object3DEventMap,
  type Scene,
  type Skeleton,
  MeshBasicNodeMaterial,
  Sphere,
  type WebGPURenderer
} from 'three/webgpu';
import { InstancedMesh2, type InstancedMesh2Params } from './InstancedMesh2.js';
import type { LODLevel, LODRenderList } from './feature/LOD.js';
import {
  INSTANCE_STATE_ACTIVE,
  INSTANCE_STATE_RENDER_LOD_SHIFT,
  INSTANCE_STATE_SHADOW_LOD_SHIFT,
  INSTANCE_STATE_VISIBLE,
  WebGPUFloatStorageBuffer,
  WebGPUInstanceStateBuffer,
  WebGPUVisibleIndexBuffer
} from './utils/WebGPUStorageBuffer.js';
import {
  WebGPUCullingPass,
  fitsStorageBindings,
  type CullingDrawRange,
  type CullingLevelDescriptor,
  type CullingPassDescriptor
} from './utils/WebGPUCullingPipeline.js';
import {
  createIndexedInstancingNodes,
  getWebGPUInstancePositionNodeFactory,
  setWebGPUInstancePositionNode,
  type NodeCompatibleMaterial,
  type WebGPUInstancePositionNodeFactory
} from '../shaders/tsl/IndexedInstancing.js';

/**
 * Selects where per-instance frustum culling and LOD selection run.
 *
 * `'auto'` takes the GPU path whenever the mesh's configuration allows it and
 * keeps the shared CPU path otherwise. `'gpu'` additionally warns when a
 * configuration forces the fallback. See `docs/webgpu-architecture.md`.
 */
export type WebGPUCullingMode = 'auto' | 'gpu' | 'cpu';

export interface InstancedMesh2WebGPUParams extends Omit<InstancedMesh2Params, 'renderer'> {
  /**
   * Accepted for constructor parity. WebGPU buffers do not need a renderer in
   * order to be allocated, so initialization remains eager without it.
   */
  renderer?: WebGPURenderer;
  /**
   * Where per-instance frustum culling and LOD selection run.
   * @default 'auto'
   */
  culling?: WebGPUCullingMode;
}

interface MaterialBaseNodes {
  castShadowPositionNode: NodeCompatibleMaterial['castShadowPositionNode'];
  colorNode: NodeCompatibleMaterial['colorNode'];
  normalNode: NodeCompatibleMaterial['normalNode'];
  positionNode: NodeCompatibleMaterial['positionNode'];
  instancePositionNodeFactory: WebGPUInstancePositionNodeFactory | null;
}

interface WebGPUMaterialState {
  ready: boolean;
  value: Material | Material[];
}

interface PendingCullRequest {
  camera: Camera;
  cameraLOD: Camera;
}

const DEFAULT_MAX_STORAGE_BUFFERS_PER_STAGE = 8;

/**
 * Batches every mesh's culling kernels for one render pass into a single
 * `renderer.compute()` call.
 *
 * Three submits a compute group on its own command buffer, so dispatching per
 * mesh costs one GPU submission per mesh per pass. A field with a few dozen
 * pooled meshes reached fifty-four submissions a frame, which a mobile GPU
 * spends more time scheduling than culling. See docs/webgpu-architecture.md.
 *
 * Membership is learned rather than guessed: the batch for a pass is whatever
 * rendered in the previous pass of the same kind. A mesh that turns up outside
 * that set dispatches once on its own and joins the next batch, so the steady
 * state is one submission per pass and a changing scene self-corrects.
 */
class WebGPUCullCoordinator {
  private readonly _accumulating = new Map<number, Set<InstancedMesh2WebGPU>>();
  private readonly _dispatched = new Map<number, Set<InstancedMesh2WebGPU>>();
  private _renderCall = -1;

  public update(renderer: WebGPURenderer, renderCall: number, owner: InstancedMesh2WebGPU, camera: Camera, cameraLOD: Camera, isShadowPass: boolean): void {
    const pass = passIndex(isShadowPass);
    const nodes: any[] = [];

    if (this._renderCall !== renderCall) {
      this._renderCall = renderCall;

      const batch = this._accumulating.get(pass) ?? new Set<InstancedMesh2WebGPU>();
      batch.add(owner);
      this._accumulating.set(pass, new Set([owner]));
      this._dispatched.set(pass, batch);

      for (const member of batch) {
        member.collectCullingKernels(renderer, renderCall, camera, cameraLOD, isShadowPass, nodes);
      }
    } else {
      this._accumulating.get(pass)?.add(owner);
      if (this._dispatched.get(pass)?.has(owner)) return;
      owner.collectCullingKernels(renderer, renderCall, camera, cameraLOD, isShadowPass, nodes);
    }

    if (nodes.length > 0) renderer.compute(nodes);
  }

  public forget(owner: InstancedMesh2WebGPU): void {
    for (const set of this._accumulating.values()) set.delete(owner);
    for (const set of this._dispatched.values()) set.delete(owner);
  }
}

const _cullCoordinators = new WeakMap<WebGPURenderer, WebGPUCullCoordinator>();

function cullCoordinatorFor(renderer: WebGPURenderer): WebGPUCullCoordinator {
  let coordinator = _cullCoordinators.get(renderer);
  if (!coordinator) {
    coordinator = new WebGPUCullCoordinator();
    _cullCoordinators.set(renderer, coordinator);
  }
  return coordinator;
}

const _materialBaseNodes = new WeakMap<Material, MaterialBaseNodes>();
const _webgpuMaterialStates = new WeakMap<object, WebGPUMaterialState>();
const _sphere = new Sphere();

/**
 * WebGPU renderer backend for InstancedMesh2.
 *
 * Instance allocation, stable ids, BVH and raycasting remain inherited from
 * the shared core, and so does the CPU culling path used as a fallback. On
 * top of the WebGL texture/attribute replacement this class adds a GPU-driven
 * render path: persistent storage buffers, a TSL compute pass that culls and
 * classifies every instance, and indirect draws whose instance counts are
 * produced on the GPU. See `docs/webgpu-architecture.md`.
 */
export class InstancedMesh2WebGPU<
  TData = {},
  TGeometry extends BufferGeometry = BufferGeometry,
  TMaterial extends Material | Material[] = Material | Material[],
  TEventMap extends Object3DEventMap = Object3DEventMap
> extends InstancedMesh2<TData, TGeometry, TMaterial, TEventMap> {
  private declare _ownedWebGPUMaterials: NodeCompatibleMaterial[];
  private declare _shadowInstanceIndex: WebGPUVisibleIndexBuffer;
  private declare _shadowPassDepth: number;
  private declare _webgpuBindingRevision: number;
  private declare _webgpuDisposed: boolean;
  private declare _instanceState: WebGPUInstanceStateBuffer;
  private declare _cullingMode: WebGPUCullingMode;
  private declare _cullPasses: (WebGPUCullingPass | null)[];
  private declare _cullDescriptors: (CullingPassDescriptor | null)[];
  private declare _pendingCull: (PendingCullRequest | null)[];
  private declare _cullNeedsRebuild: boolean[];
  private declare _cullFallbackWarned: boolean;
  private declare _webgpuRenderer: WebGPURenderer;
  private declare _gpuCullingCached: boolean;
  private declare _gpuCullingRenderCall: number;
  private declare _singleLevel: LODLevel[];
  private declare _lastRenderCamera: Camera;

  // @ts-expect-error Three declares material as a property; the accessor is
  // needed to rebuild mesh-local storage bindings on replacement.
  public override get material(): TMaterial {
    return (_webgpuMaterialStates.get(this)?.value ?? null) as TMaterial;
  }

  public override set material(value: TMaterial) {
    const state = _webgpuMaterialStates.get(this);
    if (!state) {
      _webgpuMaterialStates.set(this, { ready: false, value });
      return;
    }

    if (!state.ready) {
      state.value = value;
      return;
    }

    this.installWebGPUMaterial(value);
  }

  // eslint-disable-next-line @typescript-eslint/unified-signatures
  constructor(geometry: TGeometry, material: TMaterial, params?: InstancedMesh2WebGPUParams, LOD?: InstancedMesh2WebGPU);
  constructor(geometry: TGeometry, material: TMaterial, params?: InstancedMesh2WebGPUParams);
  constructor(geometry: TGeometry, material: TMaterial, params: InstancedMesh2WebGPUParams = {}, LOD?: InstancedMesh2WebGPU) {
    const { renderer: _renderer, culling, ...commonParams } = params;
    super(geometry, material, commonParams, LOD);

    // Keep Three's native marker so every mesh gets an object-specific node
    // binding cache, including the count === 1 case. A plain BufferAttribute
    // deliberately fails NodeMaterial's InstancedBufferAttribute guard, so
    // only our storage-buffer transform is applied. It also keeps toJSON()
    // safe without allocating a duplicate matrix payload.
    this.isInstancedMesh = true;
    this.instanceMatrix = new BufferAttribute(new Float32Array(0), 16) as unknown as typeof this.instanceMatrix;
    this.instanceColor = null;

    this._shadowPassDepth = 0;
    this._webgpuBindingRevision = 0;
    this._webgpuDisposed = false;
    this._cullingMode = culling ?? LOD?._cullingMode ?? 'auto';
    this._cullPasses = [null, null];
    this._cullDescriptors = [null, null];
    this._pendingCull = [null, null];
    this._cullNeedsRebuild = [true, true];
    this._cullFallbackWarned = false;
    this._webgpuRenderer = null;
    this._gpuCullingCached = false;
    this._gpuCullingRenderCall = -1;
    this._singleLevel = [{ distance: 0, hysteresis: 0, object: this as unknown as InstancedMesh2 }];
    this._lastRenderCamera = null;
    _webgpuMaterialStates.get(this).ready = true;
    this.installWebGPUMaterial(material);
  }

  /* --------------------------------------------------------------------- *
   * GPU-driven culling: public surface
   * --------------------------------------------------------------------- */

  /**
   * Where per-instance frustum culling and LOD selection run. Setting this on
   * a LOD child forwards to its owner.
   */
  public get culling(): WebGPUCullingMode {
    return this.webgpuOwner._cullingMode;
  }

  public set culling(value: WebGPUCullingMode) {
    const owner = this.webgpuOwner;
    if (owner._cullingMode === value) return;
    owner._cullingMode = value;
    owner._cullFallbackWarned = false;
    owner.invalidateCulling();
  }

  /**
   * Whether the GPU-driven path is currently in use. `false` means the shared
   * CPU culling path runs instead, which is always a correct fallback.
   */
  public get gpuCullingActive(): boolean {
    return this.webgpuOwner.resolveGPUCulling() !== null;
  }

  /**
   * Forces one instance onto a specific LOD level, replacing the distance
   * test for that instance. This is the GPU-readable equivalent of the CPU
   * `resolveLODIndex` callback, which cannot run inside a compute shader.
   *
   * @param id The instance id.
   * @param level The level index, or `-1` to restore distance selection.
   * @param isShadowPass Target the shadow LOD list instead of the render one.
   */
  public setLODOverrideAt(id: number, level: number, isShadowPass = false): void {
    const owner = this.webgpuOwner;
    owner._instanceState.setLODOverride(id, level, lodOverrideShift(isShadowPass));
  }

  /**
   * The LOD level forced on one instance, or `-1` when the distance test
   * decides.
   */
  public getLODOverrideAt(id: number, isShadowPass = false): number {
    const owner = this.webgpuOwner;
    return owner._instanceState.getLODOverride(id, lodOverrideShift(isShadowPass));
  }

  /**
   * Reads back the instance count each LOD level actually drew.
   *
   * On the GPU path `count` is only an upper bound, because the real number
   * is produced by the compute pass. This is the exact figure, one or more
   * frames late, and it costs a GPU readback: call it for statistics, never
   * per frame in a render loop.
   *
   * @returns One count per level of the requested pass, or `null` when the
   * GPU path is not active.
   */
  public async getVisibleCountsAsync(isShadowPass = false): Promise<number[] | null> {
    const owner = this.webgpuOwner;
    const pass = owner._cullPasses[passIndex(isShadowPass)];
    const renderer = owner._webgpuRenderer;
    if (!pass || !renderer) return null;

    const attribute = pass.indirect.attribute;
    const buffer = await renderer.getArrayBufferAsync(attribute as any);
    const commands = new Uint32Array(buffer instanceof ArrayBuffer ? buffer : (buffer as any).buffer);
    const counts: number[] = [];
    for (let level = 0; level < pass.levelCount; level++) {
      counts.push(commands[(pass.commandByteOffset(level, 0) / Uint32Array.BYTES_PER_ELEMENT) + 1] ?? 0);
    }
    (buffer as any).release?.();
    return counts;
  }

  /* --------------------------------------------------------------------- *
   * Core overrides
   * --------------------------------------------------------------------- */

  protected override initIndexAttribute(): void {
    this.instanceIndex = new WebGPUVisibleIndexBuffer(this._capacity) as unknown as typeof this.instanceIndex;
    this._shadowInstanceIndex = new WebGPUVisibleIndexBuffer(this._capacity, 'ezShadowInstanceIndex');
    this.count = 0;
  }

  /** @internal */
  public override usesCPUVisibleList(): boolean {
    return this.webgpuOwner.resolveGPUCulling() === null;
  }

  /** @internal */
  public override getInstanceIndexForPass(isShadowPass = false): typeof this.instanceIndex {
    return (isShadowPass ? this._shadowInstanceIndex : this.instanceIndex) as typeof this.instanceIndex;
  }

  protected override initMatricesTexture(): void {
    if (this._parentLOD) return;

    this.matricesTexture = new WebGPUFloatStorageBuffer(16, this._capacity, 'ezInstanceMatrices') as unknown as typeof this.matricesTexture;
    this._instanceState = new WebGPUInstanceStateBuffer(this._capacity);
  }

  protected override initColorsTexture(): void {
    if (this._parentLOD) return;

    const colors = new WebGPUFloatStorageBuffer(4, this._capacity, 'ezInstanceColors');
    colors._data.fill(1);
    colors.enqueueFullUpdate();
    this.colorsTexture = colors as unknown as typeof this.colorsTexture;
    this.rebuildLODMaterialNodes();
  }

  /** WebGPU does not add a synthetic vertex attribute to user geometry. */
  protected override patchGeometry(_geometry: TGeometry): void {
    // Mesh's constructor reaches this override before the subclass fields are
    // initialized. Later geometry replacements must refresh the conditional
    // normal transform captured by the TSL graph.
    if (this._ownedWebGPUMaterials) {
      this.rebuildMaterialNodes();
      this.webgpuOwner.invalidateCulling();
    }
  }

  protected override materialsNeedsUpdate(): void {
    if (!this._ownedWebGPUMaterials) return;
    this.rebuildLODMaterialNodes();
  }

  public override setVisibilityAt(id: number, visible: boolean): void {
    super.setVisibilityAt(id, visible);
    this.webgpuOwner._instanceState.setFlag(id, INSTANCE_STATE_VISIBLE, visible);
  }

  public override setActiveAt(id: number, active: boolean): void {
    super.setActiveAt(id, active);
    this.webgpuOwner._instanceState.setFlag(id, INSTANCE_STATE_ACTIVE, active);
  }

  public override setActiveAndVisibilityAt(id: number, value: boolean): void {
    super.setActiveAndVisibilityAt(id, value);
    this.webgpuOwner._instanceState.setFlags(id, INSTANCE_STATE_ACTIVE | INSTANCE_STATE_VISIBLE, value);
  }

  public override onBeforeShadow(
    renderer: any,
    _scene: Scene,
    _camera: Camera,
    shadowCamera: Camera,
    _geometry: BufferGeometry,
    _depthMaterial: Material,
    group: any
  ): void {
    this._shadowPassDepth++;

    const owner = this.webgpuOwner;
    if (owner.instanceIndex) {
      // WebGPURenderer renders the shadow scene with `shadow.camera`, so
      // Three r185 supplies that same camera in both callback camera slots.
      // Pass identity must therefore be explicit rather than inferred from
      // camera reference inequality, and the LOD camera has to be remembered
      // from the render pass: shadow levels are chosen by distance from the
      // viewer, not from the light, exactly as the WebGL path does.
      owner.updatePass(renderer, shadowCamera, owner._lastRenderCamera ?? shadowCamera, true);
    }

    this.prepareDraw(renderer, true, group);
  }

  public override onBeforeRender(
    renderer: any,
    _scene: Scene,
    camera: Camera,
    _geometry: BufferGeometry,
    _material: Material,
    group: any
  ): void {
    const owner = this.webgpuOwner;
    if (owner.instanceIndex && this._shadowPassDepth === 0) {
      owner._lastRenderCamera = camera;
      owner.updatePass(renderer, camera, camera, false);
    }

    this.prepareDraw(renderer, this._shadowPassDepth > 0, group);
  }

  public override onAfterShadow(
    _renderer: any,
    _scene: Scene,
    _camera: Camera,
    _shadowCamera: Camera,
    _geometry: BufferGeometry,
    _depthMaterial: Material,
    _group: any
  ): void {
    this._shadowPassDepth = Math.max(0, this._shadowPassDepth - 1);
  }

  public override onAfterRender(
    _renderer: any,
    _scene: Scene,
    _camera: Camera,
    _geometry: BufferGeometry,
    _material: Material,
    _group: any
  ): void {}

  /**
   * On the GPU path this records the request and the dispatch happens at draw
   * time, when a renderer is available. The CPU path runs immediately, as
   * before.
   */
  public override performFrustumCulling(camera: Camera, cameraLOD: Camera = camera, isShadowPass = camera !== cameraLOD): void {
    const owner = this.webgpuOwner;
    if (owner.resolveGPUCulling() === null) {
      InstancedMesh2.prototype.performFrustumCulling.call(owner, camera, cameraLOD, isShadowPass);
      return;
    }

    owner._pendingCull[passIndex(isShadowPass)] = { camera, cameraLOD };
  }

  public override resizeBuffers(capacity: number): this {
    super.resizeBuffers(capacity);
    const owner = this.webgpuOwner;
    owner.shadowIndexStorage.resize(capacity);
    owner._instanceState.resize(capacity);
    for (const object of owner.LODinfo?.objects ?? []) {
      if (object !== (owner as unknown as InstancedMesh2) && object instanceof InstancedMesh2WebGPU) {
        object.shadowIndexStorage.resize(capacity);
      }
    }
    owner.rebuildLODMaterialNodes();
    owner.invalidateCulling();
    return this;
  }

  /** @internal */
  public override addLevel(
    renderList: LODRenderList,
    geometry: BufferGeometry,
    material: Material | Material[],
    distance: number,
    hysteresis: number
  ): InstancedMesh2WebGPU {
    const objectsList = this.LODinfo.objects as unknown as InstancedMesh2WebGPU<TData>[];
    const levels = renderList.levels;
    const squaredDistance = distance ** 2;
    let object: InstancedMesh2WebGPU<TData>;

    const objectIndex = objectsList.findIndex((entry) => entry.geometry === geometry);
    if (objectIndex === -1) {
      const temporaryShadowMaterial = material ? null : this.createShadowLODSourceMaterial();
      const levelMaterial = material ?? temporaryShadowMaterial ?? new MeshBasicNodeMaterial();
      try {
        object = new InstancedMesh2WebGPU<TData>(geometry, levelMaterial, { capacity: this._capacity }, this);
      } finally {
        if (temporaryShadowMaterial) {
          const temporaryMaterials = Array.isArray(temporaryShadowMaterial)
            ? temporaryShadowMaterial
            : [temporaryShadowMaterial];
          for (const temporaryMaterial of temporaryMaterials) temporaryMaterial.dispose();
        }
      }
      object.frustumCulled = false;
      this.patchLevel(object);
      objectsList.push(object);
      this.add(object);
    } else {
      object = objectsList[objectIndex];
      if (material) object.installWebGPUMaterial(material);
    }

    let index = 0;
    for (; index < levels.length; index++) {
      if (squaredDistance < levels[index].distance) break;
    }

    levels.splice(index, 0, { distance: squaredDistance, hysteresis, object });
    renderList.count.push(0);
    this.invalidateCulling();
    return object;
  }

  /** @internal */
  public override disposeLOD(object: InstancedMesh2): void {
    object.geometry.dispose();
    this.invalidateCulling();

    if (object instanceof InstancedMesh2WebGPU) {
      object.dispose();
      return;
    }

    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) material.dispose();
  }

  public override initUniformsPerInstance(_schema: unknown): void {
    throw unsupportedFeature('per-instance uniforms');
  }

  public override initSkeleton(_skeleton: Skeleton, _disableMatrixAutoUpdate = true): void {
    throw unsupportedFeature('instanced skinning');
  }

  public override setMorphAt(_id: number, _object: Mesh): void {
    throw unsupportedFeature('per-instance morph targets');
  }

  public override clone(_recursive?: boolean): this {
    throw unsupportedFeature('clone/copy');
  }

  public override copy(_source: InstancedMesh2, _recursive?: boolean): this {
    throw unsupportedFeature('clone/copy');
  }

  public override dispose(): void {
    if (this._webgpuDisposed) return;
    this._webgpuDisposed = true;

    this.dispatchEvent<any>({ type: 'dispose' });
    if (this._webgpuRenderer) _cullCoordinators.get(this._webgpuRenderer)?.forget(this);
    this.indexStorage.dispose();
    this.shadowIndexStorage.dispose();

    if (!this._parentLOD) {
      this.matrixStorage.dispose();
      this.colorStorage?.dispose();
      this._instanceState.dispose();
      for (const pass of this._cullPasses) pass?.dispose();
      this._cullPasses = [null, null];

      for (const object of this.LODinfo?.objects ?? []) {
        if (object !== (this as unknown as InstancedMesh2) && object instanceof InstancedMesh2WebGPU) object.dispose();
      }
    }

    for (const material of this._ownedWebGPUMaterials) material.dispose();
    this._ownedWebGPUMaterials.length = 0;
  }

  /* --------------------------------------------------------------------- *
   * GPU-driven culling: internals
   * --------------------------------------------------------------------- */

  /** The path decision for one pass, resolved at most once per render call. */
  private gpuCullingForRenderCall(renderCall: number): boolean {
    if (this._gpuCullingRenderCall !== renderCall) {
      this._gpuCullingRenderCall = renderCall;
      this._gpuCullingCached = this.resolveGPUCulling() !== null;
    }
    return this._gpuCullingCached;
  }

  /** Marks the compute graphs as structurally stale. */
  private invalidateCulling(): void {
    const flags = this.webgpuOwner._cullNeedsRebuild;
    flags[0] = true;
    flags[1] = true;
  }

  /**
   * Returns `'gpu'` when the GPU path may run, or `null` when the mesh's
   * configuration requires the CPU fallback.
   */
  private resolveGPUCulling(): 'gpu' | null {
    if (this._cullingMode === 'cpu') return null;

    const blocker = this.gpuCullingBlocker();
    if (blocker === null) return 'gpu';

    if (this._cullingMode === 'gpu' && !this._cullFallbackWarned) {
      this._cullFallbackWarned = true;
      console.warn(`InstancedMesh2 WebGPU: falling back to CPU culling because ${blocker}.`);
    }
    return null;
  }

  /** A human-readable reason the GPU path cannot run, or `null`. */
  private gpuCullingBlocker(): string | null {
    if (this._sortObjects) return 'sortObjects reorders instances on the CPU';
    if (this.onFrustumEnter) return 'onFrustumEnter is a CPU callback';
    if (this.resolveLODIndex) return 'resolveLODIndex is a CPU callback; use setLODOverrideAt instead';

    const levels = this.levelsForPass(false);
    const shadowLevels = this.levelsForPass(true);
    const limit = this.maxStorageBuffersPerShaderStage();
    if (!fitsStorageBindings(Math.max(levels.length, shadowLevels.length), limit)) {
      return `this device binds at most ${limit} storage buffers per shader stage`;
    }

    for (const level of levels) {
      if (hasWireframeMaterial(level.object)) return 'wireframe rendering uses a CPU-expanded index range';
    }

    return null;
  }

  private maxStorageBuffersPerShaderStage(): number {
    const limits = (this._webgpuRenderer as any)?.backend?.device?.limits;
    return limits?.maxStorageBuffersPerShaderStage ?? DEFAULT_MAX_STORAGE_BUFFERS_PER_STAGE;
  }

  private levelsForPass(isShadowPass: boolean): LODLevel[] {
    const info = this.LODinfo;
    const list = !isShadowPass ? info?.render : (info?.shadowRender ?? info?.render);
    if (list?.levels.length > 0) return list.levels;
    return this._singleLevel;
  }

  /**
   * Runs whatever this pass needs before its draws: either one GPU dispatch
   * for the whole mesh, or the shared CPU culling plus its uploads.
   */
  private updatePass(renderer: WebGPURenderer, camera: Camera, cameraLOD: Camera, isShadowPass: boolean): void {
    this._webgpuRenderer = renderer;
    const renderCall = getRenderCall(renderer);
    const alreadyPerformed = this.frustumCullingAlreadyPerformed(renderCall, camera, isShadowPass ? camera : null);

    // Resolved once per pass rather than once per draw: `prepareDraw` runs for
    // every render object and the check walks the level lists.
    if (!this.gpuCullingForRenderCall(renderCall)) {
      if (!alreadyPerformed && this.autoUpdate) {
        InstancedMesh2.prototype.performFrustumCulling.call(this, camera, cameraLOD, isShadowPass);
      }
      this.flushPayloads();
      return;
    }

    if (alreadyPerformed) return;

    cullCoordinatorFor(renderer).update(renderer, renderCall, this, camera, cameraLOD, isShadowPass);
  }

  /**
   * Prepares this mesh's culling for one pass and appends its kernels to the
   * pass's shared submission. Called by the coordinator, never directly.
   *
   * @internal
   */
  public collectCullingKernels(renderer: WebGPURenderer, renderCall: number, camera: Camera, cameraLOD: Camera, isShadowPass: boolean, out: any[]): void {
    if (this._webgpuDisposed || !this.gpuCullingForRenderCall(renderCall)) return;

    // Uploading the instance payloads is not conditional on culling running.
    // An application that freezes culling with `autoUpdate = false` still
    // moves, recolours and hides its instances, and the CPU path has always
    // uploaded those edits unconditionally.
    this.flushPayloads();

    const pass = passIndex(isShadowPass);
    const request = this.autoUpdate ? { camera, cameraLOD } : this._pendingCull[pass];
    this._pendingCull[pass] = null;
    if (!request) return;

    const kernels = this.prepareCulling(renderer, request.camera, request.cameraLOD, isShadowPass);
    if (kernels) out.push(...kernels);
  }

  private flushPayloads(): void {
    this.matrixStorage.flush(this._instancesArrayCount);
    this.colorStorage?.flush(this._instancesArrayCount);
    this._instanceState.flush(this._instancesArrayCount);
  }

  /**
   * Prepared from a render callback on purpose: Three submits a compute group
   * on its own encoder immediately, ahead of the enclosing render pass.
   * See docs/webgpu-architecture.md.
   */
  private prepareCulling(renderer: WebGPURenderer, camera: Camera, cameraLOD: Camera, isShadowPass: boolean): any[] | null {
    const index = passIndex(isShadowPass);
    const levels = this.levelsForPass(isShadowPass);
    const descriptor = this.buildDescriptor(levels, isShadowPass);
    if (!descriptor) return null;

    let pass = this._cullPasses[index];
    if (!pass) {
      pass = new WebGPUCullingPass(isShadowPass ? 'ezShadowCulling' : 'ezCulling');
      this._cullPasses[index] = pass;
      this._cullNeedsRebuild[index] = true;
    }

    // Rebuilding regenerates the TSL graph and allocates new compute nodes,
    // which costs a pipeline compilation. It must happen only when the
    // structure actually changed -- the flag is per pass because a mesh that
    // never casts a shadow would otherwise never clear it.
    if (this._cullNeedsRebuild[index] || !sameStructure(this._cullDescriptors[index], descriptor)) {
      pass.build(descriptor);
      this._cullDescriptors[index] = descriptor;
      this._cullNeedsRebuild[index] = false;
    } else {
      pass.writeCommands(descriptor);
      pass.setLevelDistances(descriptor.levels);
    }

    const geometrySphere = this.cullingSphere();
    if (!geometrySphere) return null;

    pass.setCamera(camera, cameraLOD, this.matrixWorld, this._perObjectFrustumCulled, geometrySphere);
    return pass.prepare(this._instancesArrayCount);
  }

  private cullingSphere(): Sphere | null {
    const geometry = this._geometry;
    if (!geometry.boundingSphere) geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere;
    if (!sphere) return null;
    _sphere.copy(sphere);
    return _sphere;
  }

  private buildDescriptor(levels: LODLevel[], isShadowPass: boolean): CullingPassDescriptor | null {
    const levelDescriptors: CullingLevelDescriptor[] = [];

    for (const level of levels) {
      const object = level.object as InstancedMesh2WebGPU;
      const indexBuffer = object.getInstanceIndexForPass(isShadowPass) as unknown as WebGPUVisibleIndexBuffer;
      const visibleAttribute = indexBuffer?.attribute;
      if (!visibleAttribute) return null;

      levelDescriptors.push({
        visibleAttribute,
        distanceSquared: level.distance,
        hysteresis: level.hysteresis,
        indexed: object.geometry.index !== null,
        ranges: drawRangesOf(object)
      });
    }

    return {
      matrixAttribute: this.matrixStorage.attribute,
      stateAttribute: this._instanceState.attribute,
      capacity: this._capacity,
      lodOverrideShift: lodOverrideShift(isShadowPass),
      levels: levelDescriptors
    };
  }

  /**
   * Points this object's draw at the right indirect command, or falls back to
   * the CPU-known instance count.
   */
  private prepareDraw(renderer: WebGPURenderer, isShadowPass: boolean, group: any): void {
    const owner = this.webgpuOwner;
    const geometry = this._geometry;

    if (!owner._gpuCullingCached) {
      geometry.indirect = null;
      geometry.indirectOffset = 0;
      // Each LOD level owns its visible-index prefix, so the upload stays per
      // object rather than per owner.
      const indexStorage = isShadowPass ? this.shadowIndexStorage : this.indexStorage;
      indexStorage.update(renderer, this.count);
      return;
    }

    const pass = owner._cullPasses[passIndex(isShadowPass)];
    const levels = owner.levelsForPass(isShadowPass);
    const levelIndex = levels.findIndex((level) => level.object === (this as unknown as InstancedMesh2));
    if (!pass || levelIndex === -1) {
      geometry.indirect = null;
      geometry.indirectOffset = 0;
      return;
    }

    // With nothing allocated no kernel runs, so the commands still hold the
    // previous frame's counts. `count` is 0 too, which already stops Three
    // issuing the draw, but the indirect binding is cleared as well so the
    // two cannot drift apart.
    if (owner._instancesArrayCount === 0) {
      geometry.indirect = null;
      geometry.indirectOffset = 0;
      this.count = 0;
      return;
    }

    const rangeIndex = group ? Math.max(0, geometry.groups.indexOf(group)) : 0;
    geometry.indirect = pass.indirect.attribute as any;
    geometry.indirectOffset = pass.commandByteOffset(levelIndex, rangeIndex);
    // `count` only has to keep Three from skipping the draw; the GPU owns the
    // real instance count. See docs/webgpu-architecture.md.
    this.count = owner._instancesCount;
  }

  private get webgpuOwner(): InstancedMesh2WebGPU {
    return (this._parentLOD as unknown as InstancedMesh2WebGPU) ?? this;
  }

  private get indexStorage(): WebGPUVisibleIndexBuffer {
    return this.instanceIndex as unknown as WebGPUVisibleIndexBuffer;
  }

  private get shadowIndexStorage(): WebGPUVisibleIndexBuffer {
    return this._shadowInstanceIndex;
  }

  private get matrixStorage(): WebGPUFloatStorageBuffer {
    return this.matricesTexture as unknown as WebGPUFloatStorageBuffer;
  }

  private get colorStorage(): WebGPUFloatStorageBuffer | null {
    return this.colorsTexture as unknown as WebGPUFloatStorageBuffer | null;
  }

  private installWebGPUMaterial(material: Material | Material[]): void {
    const sources = Array.isArray(material) ? material : [material];
    for (const source of sources) assertWebGPUCompatibleMaterial(source);

    const materials = sources.map((source) => {
      const clone = source.clone() as NodeCompatibleMaterial;
      const sourceProgramCacheKey = clone.customProgramCacheKey.bind(clone);
      clone.customProgramCacheKey = () => `${sourceProgramCacheKey()}|ez-webgpu-bindings:${this._webgpuBindingRevision}`;
      const sourceNodes = source as NodeCompatibleMaterial;
      // If another WebGPU InstancedMesh2-owned material is deliberately reused
      // as a source, unwrap its source graph. Treating the composed
      // positionNode as a new base would apply indexed instancing twice.
      const inheritedBaseNodes = _materialBaseNodes.get(source);
      const castShadowPositionNode = inheritedBaseNodes
        ? inheritedBaseNodes.castShadowPositionNode
        : (sourceNodes.castShadowPositionNode ?? null);
      const colorNode = inheritedBaseNodes
        ? inheritedBaseNodes.colorNode
        : (sourceNodes.colorNode ?? null);
      const normalNode = inheritedBaseNodes
        ? inheritedBaseNodes.normalNode
        : (sourceNodes.normalNode ?? null);
      const positionNode = inheritedBaseNodes
        ? inheritedBaseNodes.positionNode
        : (sourceNodes.positionNode ?? null);
      const instancePositionNodeFactory = getWebGPUInstancePositionNodeFactory(source)
        ?? inheritedBaseNodes?.instancePositionNodeFactory
        ?? null;
      // Classic materials do not promise to copy NodeMaterial extension
      // fields. Preserve all source nodes explicitly before composing the
      // mesh-local indexed-instancing graph.
      clone.castShadowPositionNode = castShadowPositionNode;
      clone.colorNode = colorNode;
      clone.normalNode = normalNode;
      clone.positionNode = positionNode;
      _materialBaseNodes.set(clone, {
        castShadowPositionNode,
        colorNode,
        normalNode,
        positionNode,
        instancePositionNodeFactory
      });
      return clone;
    });

    const previousMaterials = this._ownedWebGPUMaterials;
    const previousValue = _webgpuMaterialStates.get(this).value;
    this._ownedWebGPUMaterials = materials;
    _webgpuMaterialStates.get(this).value = Array.isArray(material) ? materials : materials[0];

    try {
      this.rebuildMaterialNodes();
    } catch (error) {
      this._ownedWebGPUMaterials = previousMaterials;
      _webgpuMaterialStates.get(this).value = previousValue;
      for (const ownedMaterial of materials) ownedMaterial.dispose();
      throw error;
    }

    for (const ownedMaterial of previousMaterials ?? []) ownedMaterial.dispose();
  }

  private rebuildLODMaterialNodes(): void {
    const owner = this.webgpuOwner;
    owner.rebuildMaterialNodes();

    for (const object of owner.LODinfo?.objects ?? []) {
      if (object !== (owner as unknown as InstancedMesh2) && object instanceof InstancedMesh2WebGPU) {
        object.rebuildMaterialNodes();
      }
    }
  }

  private rebuildMaterialNodes(): void {
    if (!this._ownedWebGPUMaterials) return;

    // StorageBufferAttribute cannot be resized in place. The storage wrappers
    // therefore replace their attributes on growth, and Three must discard
    // the RenderObject that still owns bindings for the previous attributes.
    // Including this revision in customProgramCacheKey makes needsUpdate take
    // the cache-miss path instead of retaining those stale bindings.
    this._webgpuBindingRevision++;

    const owner = this.webgpuOwner;
    const colorAttribute = owner.colorStorage?.attribute ?? null;

    for (const material of this._ownedWebGPUMaterials) {
      const baseNodes = _materialBaseNodes.get(material);
      const nodes = createIndexedInstancingNodes(
        this.geometry,
        this.indexStorage.attribute,
        owner.matrixStorage.attribute,
        colorAttribute,
        owner._capacity,
        baseNodes?.positionNode ?? null,
        baseNodes?.colorNode ?? null,
        baseNodes?.instancePositionNodeFactory ?? null
      );
      const shadowNodes = createIndexedInstancingNodes(
        this.geometry,
        this.shadowIndexStorage.attribute,
        owner.matrixStorage.attribute,
        null,
        owner._capacity,
        baseNodes?.castShadowPositionNode ?? baseNodes?.positionNode ?? null,
        null,
        baseNodes?.instancePositionNodeFactory ?? null
      );

      material.positionNode = nodes.positionNode;
      material.castShadowPositionNode = shadowNodes.positionNode;
      material.colorNode = nodes.colorNode ?? baseNodes?.colorNode ?? null;
      material.normalNode = baseNodes?.normalNode ?? null;
      material.needsUpdate = true;
    }
  }

  /**
   * A shadow-only LOD has no material parameter of its own. Clone the owner's
   * uncomposed source state so alpha masking, custom normals and local vertex
   * deformation survive without wrapping the owner's already-instanced graph
   * a second time.
   */
  private createShadowLODSourceMaterial(): Material | Material[] | null {
    if (!this._ownedWebGPUMaterials?.length) return null;

    const sources = this._ownedWebGPUMaterials.map((ownedMaterial) => {
      const baseNodes = _materialBaseNodes.get(ownedMaterial);
      const source = ownedMaterial.clone() as NodeCompatibleMaterial;
      source.castShadowPositionNode = baseNodes?.castShadowPositionNode ?? null;
      source.colorNode = baseNodes?.colorNode ?? null;
      source.normalNode = baseNodes?.normalNode ?? null;
      source.positionNode = baseNodes?.positionNode ?? null;
      if (baseNodes?.instancePositionNodeFactory) {
        setWebGPUInstancePositionNode(source, baseNodes.instancePositionNodeFactory);
      }
      return source;
    });

    return Array.isArray(this.material) ? sources : sources[0];
  }
}

function passIndex(isShadowPass: boolean): number {
  return isShadowPass ? 1 : 0;
}

function lodOverrideShift(isShadowPass: boolean): number {
  return isShadowPass ? INSTANCE_STATE_SHADOW_LOD_SHIFT : INSTANCE_STATE_RENDER_LOD_SHIFT;
}

/**
 * `info.render.calls` advances once per render or shadow-map pass and, unlike
 * `info.calls`, is not moved by the compute dispatches this backend issues.
 */
function getRenderCall(renderer: WebGPURenderer): number {
  return renderer.info.render.calls;
}

function hasWireframeMaterial(object: InstancedMesh2): boolean {
  const material = object.material;
  if (!Array.isArray(material)) return (material as any)?.wireframe === true;
  for (const entry of material) {
    if ((entry as any)?.wireframe === true) return true;
  }
  return false;
}

function drawRangesOf(object: InstancedMesh2): CullingDrawRange[] {
  const geometry = object.geometry;
  const index = geometry.index;
  const itemCount = index ? index.count : (geometry.attributes.position?.count ?? 0);
  const drawRange = geometry.drawRange;
  const first = Math.max(0, drawRange.start);
  const last = Math.min(itemCount, drawRange.start + drawRange.count);
  const groups = geometry.groups;

  if (!Array.isArray(object.material) || groups.length === 0) {
    return [{ firstElement: first, elementCount: Math.max(0, last - first) }];
  }

  return groups.map((group) => {
    const start = Math.max(first, group.start);
    const end = Math.min(last, group.start + group.count);
    return { firstElement: start, elementCount: Math.max(0, end - start) };
  });
}

/**
 * Whether a rebuilt descriptor still matches the compute graph that was
 * generated from it. Only structure matters: distances and draw ranges are
 * refreshed without a rebuild.
 */
function sameStructure(previous: CullingPassDescriptor | null, next: CullingPassDescriptor): boolean {
  if (!previous) return false;
  if (previous.capacity !== next.capacity) return false;
  if (previous.matrixAttribute !== next.matrixAttribute) return false;
  if (previous.stateAttribute !== next.stateAttribute) return false;
  if (previous.levels.length !== next.levels.length) return false;

  for (let i = 0; i < next.levels.length; i++) {
    const a = previous.levels[i];
    const b = next.levels[i];
    if (a.visibleAttribute !== b.visibleAttribute) return false;
    if (a.indexed !== b.indexed) return false;
    if (a.ranges.length !== b.ranges.length) return false;
  }

  return true;
}

function assertWebGPUCompatibleMaterial(material: Material): void {
  if (isShaderMaterial(material)
    && !(material as Material & { isNodeMaterial?: boolean }).isNodeMaterial) {
    throw new Error('InstancedMesh2 WebGPU does not support ShaderMaterial. Use a NodeMaterial/TSL material.');
  }

  const isNodeMaterial = !!(material as Material & { isNodeMaterial?: boolean }).isNodeMaterial;
  if (!isNodeMaterial && material.onBeforeCompile !== Material.prototype.onBeforeCompile) {
    throw new Error(
      'InstancedMesh2 WebGPU cannot run a GLSL onBeforeCompile customization. '
      + 'Port the material customization to NodeMaterial/TSL.'
    );
  }
}

function isShaderMaterial(material: Material): boolean {
  return !!(material as Material & { isShaderMaterial?: boolean }).isShaderMaterial;
}

function unsupportedFeature(feature: string): Error {
  return new Error(`InstancedMesh2 WebGPU does not support ${feature} yet.`);
}
