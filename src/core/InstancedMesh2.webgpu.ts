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
  type WebGPURenderer
} from 'three/webgpu';
import { InstancedMesh2, type InstancedMesh2Params } from './InstancedMesh2.js';
import type { LODRenderList } from './feature/LOD.js';
import { WebGPUFloatStorageBuffer, WebGPUVisibleIndexBuffer } from './utils/WebGPUStorageBuffer.js';
import {
  createIndexedInstancingNodes,
  getWebGPUInstancePositionNodeFactory,
  setWebGPUInstancePositionNode,
  type NodeCompatibleMaterial,
  type WebGPUInstancePositionNodeFactory
} from '../shaders/tsl/IndexedInstancing.js';

export interface InstancedMesh2WebGPUParams extends Omit<InstancedMesh2Params, 'renderer'> {
  /**
   * Accepted for constructor parity. WebGPU buffers do not need a renderer in
   * order to be allocated, so initialization remains eager without it.
   */
  renderer?: WebGPURenderer;
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

const _materialBaseNodes = new WeakMap<Material, MaterialBaseNodes>();
const _webgpuMaterialStates = new WeakMap<object, WebGPUMaterialState>();

/**
 * WebGPU renderer backend for InstancedMesh2.
 *
 * Instance allocation, stable ids, visibility, CPU frustum culling, BVH and
 * LOD remain inherited from the shared core. This class replaces only the
 * WebGL texture/attribute shader path with persistent storage buffers and TSL
 * visible-index indirection.
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
    const { renderer: _renderer, ...commonParams } = params;
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
    _webgpuMaterialStates.get(this).ready = true;
    this.installWebGPUMaterial(material);
  }

  protected override initIndexAttribute(): void {
    this.instanceIndex = new WebGPUVisibleIndexBuffer(this._capacity) as unknown as typeof this.instanceIndex;
    this._shadowInstanceIndex = new WebGPUVisibleIndexBuffer(this._capacity, 'ezShadowInstanceIndex');
    this.count = 0;
  }

  /** @internal */
  public override getInstanceIndexForPass(isShadowPass = false): typeof this.instanceIndex {
    return (isShadowPass ? this._shadowInstanceIndex : this.instanceIndex) as typeof this.instanceIndex;
  }

  protected override initMatricesTexture(): void {
    if (!this._parentLOD) {
      this.matricesTexture = new WebGPUFloatStorageBuffer(16, this._capacity, 'ezInstanceMatrices') as unknown as typeof this.matricesTexture;
    }
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
    if (this._ownedWebGPUMaterials) this.rebuildMaterialNodes();
  }

  protected override materialsNeedsUpdate(): void {
    if (!this._ownedWebGPUMaterials) return;
    this.rebuildLODMaterialNodes();
  }

  public override onBeforeShadow(
    renderer: any,
    _scene: Scene,
    _camera: Camera,
    shadowCamera: Camera,
    _geometry: BufferGeometry,
    _depthMaterial: Material,
    _group: any
  ): void {
    this._shadowPassDepth++;

    const owner = this.webgpuOwner;
    if (owner.instanceIndex && owner.autoUpdate) {
      const renderCall = getRenderCall(renderer);
      if (!owner.frustumCullingAlreadyPerformed(renderCall, shadowCamera, shadowCamera)) {
        // WebGPURenderer renders the shadow scene with `shadow.camera`, so
        // Three r185 supplies that same camera in both callback camera slots.
        // Pass identity must therefore be explicit rather than inferred from
        // camera reference inequality.
        owner.performFrustumCulling(shadowCamera, shadowCamera, true);
      }
    }
  }

  public override onBeforeRender(
    renderer: any,
    _scene: Scene,
    camera: Camera,
    _geometry: BufferGeometry,
    _material: Material,
    _group: any
  ): void {
    const owner = this.webgpuOwner;
    if (owner.instanceIndex && owner.autoUpdate && this._shadowPassDepth === 0) {
      const renderCall = getRenderCall(renderer);
      if (!owner.frustumCullingAlreadyPerformed(renderCall, camera, null)) {
        owner.performFrustumCulling(camera);
      }
    }

    owner.matrixStorage.flush(owner._instancesArrayCount);
    owner.colorStorage?.flush(owner._instancesArrayCount);
    const indexStorage = this._shadowPassDepth > 0
      ? this.shadowIndexStorage
      : this.indexStorage;
    indexStorage.update(renderer, this.count);
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

  public override resizeBuffers(capacity: number): this {
    super.resizeBuffers(capacity);
    const owner = this.webgpuOwner;
    owner.shadowIndexStorage.resize(capacity);
    for (const object of owner.LODinfo?.objects ?? []) {
      if (object !== (owner as unknown as InstancedMesh2) && object instanceof InstancedMesh2WebGPU) {
        object.shadowIndexStorage.resize(capacity);
      }
    }
    owner.rebuildLODMaterialNodes();
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
    return object;
  }

  /** @internal */
  public override disposeLOD(object: InstancedMesh2): void {
    object.geometry.dispose();

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
    this.indexStorage.dispose();
    this.shadowIndexStorage.dispose();

    if (!this._parentLOD) {
      this.matrixStorage.dispose();
      this.colorStorage?.dispose();

      for (const object of this.LODinfo?.objects ?? []) {
        if (object !== (this as unknown as InstancedMesh2) && object instanceof InstancedMesh2WebGPU) object.dispose();
      }
    }

    for (const material of this._ownedWebGPUMaterials) material.dispose();
    this._ownedWebGPUMaterials.length = 0;
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

function getRenderCall(renderer: WebGPURenderer): number {
  return renderer.info.calls;
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
