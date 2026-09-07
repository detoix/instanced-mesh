import { InstancedBufferAttribute, UnsignedIntType, type BufferGeometry, type Material, type StorageBufferAttribute } from 'three/webgpu';
import { Fn, attribute, materialColor, normalLocal, positionLocal, storage, transformNormal, vec4 } from 'three/tsl';

const EZ_DRAW_SLOT = 'ezDrawSlot';

/**
 * Per-instance draw slot as a vertex attribute, not the instance_index builtin.
 * Dawn's D3D12 backend selects an expanded indirect command signature whenever a
 * vertex shader uses vertex_index or instance_index, and that signature hangs
 * Intel Iris Xe / UHD 700 drivers (DXGI_ERROR_DEVICE_HUNG). Feeding the slot in
 * as a static attribute keeps both builtins out of the generated WGSL, so Dawn
 * selects the simple signature. Verified: 3/12 hangs before, 0/12 after.
 * https://www.intel.com/content/www/us/en/support/articles/000102979/graphics.html
 */
function drawSlotAttribute(geometry: BufferGeometry, capacity: number): void {
  const existing = geometry.getAttribute(EZ_DRAW_SLOT);
  if (existing && existing.count >= capacity) return;
  const slots = new Uint32Array(capacity);
  for (let i = 0; i < capacity; i += 1) slots[i] = i;
  const attr = new InstancedBufferAttribute(slots, 1);
  (attr as any).gpuType = UnsignedIntType;
  geometry.setAttribute(EZ_DRAW_SLOT, attr);
}

/**
 * Nodes and immutable CPU context available to a WebGPU instance-position
 * factory. `instanceId` is the stable id resolved through the compact visible
 * index, while `drawInstanceIndex` is that compact draw slot itself.
 */
export interface WebGPUInstancePositionNodeContext {
  readonly positionNode: any;
  readonly normalNode: any;
  readonly instanceMatrix: any;
  readonly instanceId: any;
  readonly drawInstanceIndex: any;
  readonly geometry: BufferGeometry;
}

/** Return a local-space TSL `vec3`; indexed instancing is applied afterwards. */
export type WebGPUInstancePositionNodeFactory = (
  context: WebGPUInstancePositionNodeContext
) => any;

const _instancePositionNodeFactories = new WeakMap<Material, WebGPUInstancePositionNodeFactory>();

/**
 * Attach an instance-aware local-position stage to a WebGPU source material.
 *
 * The backend captures this factory before cloning the material, so it remains
 * attached to every mesh-local graph rebuilt for LODs and buffer growth. Pass
 * `null` to remove a previously attached factory before constructing a mesh.
 */
export function setWebGPUInstancePositionNode<TMaterial extends Material>(
  material: TMaterial,
  factory: WebGPUInstancePositionNodeFactory | null
): TMaterial {
  if (!material?.isMaterial) {
    throw new TypeError('setWebGPUInstancePositionNode requires a Three.js material.');
  }
  if (factory !== null && typeof factory !== 'function') {
    throw new TypeError('The WebGPU instance position node factory must be a function or null.');
  }

  if (factory === null) _instancePositionNodeFactories.delete(material);
  else _instancePositionNodeFactories.set(material, factory);

  return material;
}

/** @internal */
export function getWebGPUInstancePositionNodeFactory(
  material: Material
): WebGPUInstancePositionNodeFactory | null {
  return _instancePositionNodeFactories.get(material) ?? null;
}

export interface IndexedInstancingNodes {
  positionNode: any;
  colorNode: any | null;
}

/**
 * Builds a material-local TSL graph for visible-index indirection.
 *
 * `instanceIndex` addresses the compact visible list. That list resolves the
 * stable instance id used to fetch the matrix and optional color. Keeping the
 * lookup in the shader lets CPU culling and LOD reorder instances without
 * copying their matrix/color payloads.
 */
export function createIndexedInstancingNodes(
  geometry: BufferGeometry,
  indexAttribute: StorageBufferAttribute,
  matrixAttribute: StorageBufferAttribute,
  colorAttribute: StorageBufferAttribute | null,
  capacity: number,
  basePositionNode: any | null,
  baseColorNode: any | null,
  instancePositionNodeFactory: WebGPUInstancePositionNodeFactory | null
): IndexedInstancingNodes {
  drawSlotAttribute(geometry, capacity);
  const drawSlot = attribute(EZ_DRAW_SLOT, 'uint');
  const indexStorage = storage(indexAttribute, 'uint', capacity).toReadOnly();
  const matrixStorage = storage(matrixAttribute, 'mat4', capacity).toReadOnly();
  const stableInstanceId = indexStorage.element(drawSlot);
  const instanceMatrix = matrixStorage.element(stableInstanceId);
  const localPositionNode = basePositionNode ?? positionLocal;
  // Capture the geometry-local normal before the Fn below assigns the
  // instance-transformed value back to Three's normalLocal property. A
  // deformation factory using this context must not observe that later write.
  const localNormalNode = normalLocal.toVar();
  const deformedPositionNode = instancePositionNodeFactory?.({
    positionNode: localPositionNode,
    normalNode: localNormalNode,
    instanceMatrix,
    instanceId: stableInstanceId,
    drawInstanceIndex: drawSlot,
    geometry
  }) ?? localPositionNode;

  const positionNode = Fn(() => {
    if (geometry.hasAttribute('normal')) {
      normalLocal.assign(transformNormal(localNormalNode, instanceMatrix));
    }

    return instanceMatrix.mul(vec4(deformedPositionNode, 1)).xyz;
  })();

  let colorNode: any | null = null;
  if (colorAttribute !== null) {
    const colorStorage = storage(colorAttribute, 'vec4', capacity).toReadOnly();
    const instanceColor = colorStorage.element(stableInstanceId);
    colorNode = vec4(baseColorNode ?? materialColor).mul(instanceColor);
  }

  return { positionNode, colorNode };
}

/**
 * Node fields exist on NodeMaterial and are copied by WebGPURenderer's
 * material adapter for compatible classic materials.
 */
export interface NodeCompatibleMaterial extends Material {
  castShadowPositionNode?: any | null;
  colorNode?: any | null;
  normalNode?: any | null;
  positionNode?: any | null;
}
