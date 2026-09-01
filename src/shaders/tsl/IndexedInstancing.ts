import { type BufferGeometry, type Material, type StorageBufferAttribute } from 'three/webgpu';
import { Fn, instanceIndex, materialColor, normalLocal, positionLocal, storage, transformNormal, vec4 } from 'three/tsl';

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
  const indexStorage = storage(indexAttribute, 'uint', capacity).toReadOnly();
  const matrixStorage = storage(matrixAttribute, 'mat4', capacity).toReadOnly();
  const stableInstanceId = indexStorage.element(instanceIndex);
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
    drawInstanceIndex: instanceIndex,
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
