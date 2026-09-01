export * from './index.common.js';
export {
  InstancedMesh2WebGPU as InstancedMesh2,
  type InstancedMesh2WebGPUParams as InstancedMesh2Params
} from './core/InstancedMesh2.webgpu.js';
export {
  setWebGPUInstancePositionNode,
  type WebGPUInstancePositionNodeContext,
  type WebGPUInstancePositionNodeFactory
} from './shaders/tsl/IndexedInstancing.js';
