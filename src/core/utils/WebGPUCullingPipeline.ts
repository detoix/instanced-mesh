import { Camera, Frustum, Matrix4, Sphere, Vector3, Vector4 } from 'three';
import type { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { Fn, If, atomicAdd, atomicLoad, atomicStore, instanceIndex, max, min, select, storage, uint, uniform, vec4 } from 'three/tsl';
import { DRAW_COMMAND_BYTES, DRAW_COMMAND_WORDS, WebGPUIndirectDrawBuffer } from './WebGPUStorageBuffer.js';

/**
 * Storage bindings a culling dispatch needs besides its per-level visible
 * lists: instance matrices, instance state, and the indirect draw commands.
 */
const FIXED_STORAGE_BINDINGS = 3;

/**
 * Word 1 of every WebGPU indirect draw command is `instanceCount`, for both
 * the indexed and the non-indexed layout, so the kernel needs no struct and
 * no knowledge of which geometry a command belongs to.
 */
const INSTANCE_COUNT_WORD = 1;

/** One geometry sub-range drawn with its own material. */
export interface CullingDrawRange {
  elementCount: number;
  firstElement: number;
}

/** One LOD level, or the single level of a mesh without LODs. */
export interface CullingLevelDescriptor {
  visibleAttribute: StorageBufferAttribute;
  /** Squared distance threshold, before hysteresis. Ignored for level 0. */
  distanceSquared: number;
  hysteresis: number;
  indexed: boolean;
  ranges: CullingDrawRange[];
}

export interface CullingPassDescriptor {
  matrixAttribute: StorageBufferAttribute;
  stateAttribute: StorageBufferAttribute;
  capacity: number;
  /** Bit offset of the LOD override field this pass reads. */
  lodOverrideShift: number;
  levels: CullingLevelDescriptor[];
}

/**
 * True when a descriptor's level count fits inside the device's per-stage
 * storage-buffer limit. Callers fall back to CPU culling when it does not.
 */
export function fitsStorageBindings(levelCount: number, maxStorageBuffersPerShaderStage: number): boolean {
  return levelCount + FIXED_STORAGE_BINDINGS <= maxStorageBuffersPerShaderStage;
}

const _projScreenMatrix = new Matrix4();
const _invMatrixWorld = new Matrix4();
const _cameraLODPos = new Vector3();
const _frustum = new Frustum();

/**
 * One GPU culling + LOD-classification dispatch for a single render pass.
 *
 * The compute kernel reads the persistent instance matrices and per-instance
 * state, tests an object-space bounding sphere against the six object-space
 * frustum planes, classifies the survivor into a LOD level, and appends its
 * id to that level's visible list through an atomic increment of the level's
 * indirect `instanceCount`. See `docs/webgpu-architecture.md`.
 */
export class WebGPUCullingPass {
  public readonly indirect: WebGPUIndirectDrawBuffer;
  /** Increments whenever the compute graph is rebuilt. */
  public revision = 0;

  private readonly _name: string;
  private readonly _planeUniforms: any[] = [];
  private _levelCount = 0;
  private _rangeCounts: number[] = [];
  private _commandBase: number[] = [];
  private _cameraPosUniform: any = null;
  private _sphereCenterUniform: any = null;
  private _sphereRadiusUniform: any = null;
  private _cullEnabledUniform: any = null;
  private _thresholdUniforms: any[] = [];
  private _resetNode: any = null;
  private _cullNode: any = null;
  private _publishNode: any = null;
  private _computeNodes: any[] = [];
  private _disposed = false;

  public constructor(name: string) {
    this._name = name;
    this.indirect = new WebGPUIndirectDrawBuffer(1, `${name} draws`);
    for (let i = 0; i < 6; i++) this._planeUniforms.push(uniform(new Vector4()));
    this._cameraPosUniform = uniform(new Vector3());
    this._sphereCenterUniform = uniform(new Vector3());
    this._sphereRadiusUniform = uniform(0);
    this._cullEnabledUniform = uniform(1, 'uint');
  }

  /** Byte offset of the indirect command for one level and draw range. */
  public commandByteOffset(level: number, range: number): number {
    return (this._commandBase[level] + range) * DRAW_COMMAND_BYTES;
  }

  /**
   * (Re)builds the compute graph. Cheap to call when nothing structural
   * changed: it returns early unless a rebuild is actually required.
   */
  public build(descriptor: CullingPassDescriptor): void {
    const levels = descriptor.levels;
    const rangeCounts = levels.map((level) => Math.max(1, level.ranges.length));

    this._levelCount = levels.length;
    this._rangeCounts = rangeCounts;
    this._commandBase = [];

    let commandCount = 0;
    for (const count of rangeCounts) {
      this._commandBase.push(commandCount);
      commandCount += count;
    }

    // Three caches compute pipelines, bindings and node state on node
    // identity and releases them from the node's own dispose event, so the
    // graph being replaced has to be disposed or its GPU pipeline is stranded
    // for the renderer's lifetime.
    this.disposeKernels();

    this.indirect.setCommandCount(commandCount);
    this.writeCommands(descriptor);
    this.indirect.upload();

    this._thresholdUniforms = levels.map((level) => uniform(effectiveThreshold(level)));

    const capacity = descriptor.capacity;
    const stateRead = storage(descriptor.stateAttribute, 'uint', capacity).toReadOnly();
    const matrixRead = storage(descriptor.matrixAttribute, 'mat4', capacity).toReadOnly();
    const drawStorage = storage(this.indirect.attribute, 'uint', commandCount * DRAW_COMMAND_WORDS).toAtomic();
    const visibleWrites = levels.map((level) => storage(level.visibleAttribute, 'uint', capacity));

    const planes = this._planeUniforms;
    const cameraPos = this._cameraPosUniform;
    const sphereCenter = this._sphereCenterUniform;
    const sphereRadius = this._sphereRadiusUniform;
    const cullEnabled = this._cullEnabledUniform;
    const thresholds = this._thresholdUniforms;
    const overrideShift = descriptor.lodOverrideShift;
    const levelCount = levels.length;
    const commandBase = this._commandBase;

    this._resetNode = Fn(() => {
      const word = instanceIndex.mul(uint(DRAW_COMMAND_WORDS)).add(uint(INSTANCE_COUNT_WORD));
      atomicStore(drawStorage.element(word), uint(0));
    })().compute(commandCount, [64]).setName(`${this._name} reset`);

    this._cullNode = Fn(() => {
      const state = stateRead.element(instanceIndex).toVar('ezState');

      // Active and visible are bits 1 and 0; both must be set.
      If(state.bitAnd(uint(3)).equal(uint(3)), () => {
        const matrix = matrixRead.element(instanceIndex);
        const center = matrix.mul(vec4(sphereCenter, 1)).xyz.toVar('ezCenter');
        // Column lengths give the per-axis scale without matrix indexing.
        const scaleX = matrix.mul(vec4(1, 0, 0, 0)).xyz.length();
        const scaleY = matrix.mul(vec4(0, 1, 0, 0)).xyz.length();
        const scaleZ = matrix.mul(vec4(0, 0, 1, 0)).xyz.length();
        const radius = sphereRadius.mul(max(max(scaleX, scaleY), scaleZ)).toVar('ezRadius');
        const negRadius = radius.negate();

        let inFrustum = planes[0].xyz.dot(center).add(planes[0].w).greaterThanEqual(negRadius);
        for (let i = 1; i < 6; i++) {
          inFrustum = inFrustum.and(planes[i].xyz.dot(center).add(planes[i].w).greaterThanEqual(negRadius));
        }

        If(cullEnabled.equal(uint(0)).or(inFrustum), () => {
          const level = uint(0).toVar('ezLevel');

          if (levelCount > 1) {
            const delta = center.sub(cameraPos);
            const distanceSquared = delta.dot(delta).toVar('ezDistanceSq');
            for (let i = 1; i < levelCount; i++) {
              level.assign(select(distanceSquared.greaterThanEqual(thresholds[i]), uint(i), level));
            }

            const override = state.shiftRight(uint(overrideShift)).bitAnd(uint(0xff)).toVar('ezOverride');
            const clamped = (min as any)(override.sub(uint(1)), uint(levelCount - 1));
            level.assign(select(override.greaterThan(uint(0)), clamped, level));
          }

          // Atomic append rather than a prefix sum, and one storage binding
          // per level rather than a merged buffer. WGSL also cannot index a
          // set of distinct bindings dynamically, so the append is unrolled
          // over the (small, capped) level count.
          // See docs/webgpu-architecture.md.
          for (let i = 0; i < levelCount; i++) {
            If(level.equal(uint(i)), () => {
              const countWord = uint(commandBase[i] * DRAW_COMMAND_WORDS + INSTANCE_COUNT_WORD);
              const slot = atomicAdd(drawStorage.element(countWord), uint(1));
              visibleWrites[i].element(slot).assign(instanceIndex);
            });
          }
        });
      });
    })().compute(1, [64]).setName(`${this._name} cull`);

    // A material array draws one command per geometry group, but every group
    // of a level draws the same visible list. The kernel counts once, into the
    // level's first command, and this publishes that count to the others.
    // Dispatches inside one compute pass are ordered, so it observes the count.
    const extraRanges: number[][] = [];
    for (let level = 0; level < levelCount; level++) {
      for (let range = 1; range < rangeCounts[level]; range++) {
        extraRanges.push([commandBase[level], commandBase[level] + range]);
      }
    }

    this._publishNode = extraRanges.length === 0
      ? null
      : Fn(() => {
          for (const [source, target] of extraRanges) {
            const total = atomicLoad(drawStorage.element(uint(source * DRAW_COMMAND_WORDS + INSTANCE_COUNT_WORD)));
            atomicStore(drawStorage.element(uint(target * DRAW_COMMAND_WORDS + INSTANCE_COUNT_WORD)), total);
          }
        })().compute(1, [1]).setName(`${this._name} publish`);

    this._computeNodes = this._publishNode === null
      ? [this._resetNode, this._cullNode]
      : [this._resetNode, this._cullNode, this._publishNode];
    this.revision++;
  }

  /**
   * Rewrites the geometry-dependent command words without a graph rebuild, and
   * uploads only when one actually changed. An upload also resets the counts
   * the previous frame's kernel wrote, so it must not happen every frame.
   */
  public writeCommands(descriptor: CullingPassDescriptor): void {
    const levels = descriptor.levels;
    let changed = false;

    for (let level = 0; level < levels.length; level++) {
      const ranges = levels[level].ranges;
      const base = this._commandBase[level];
      for (let range = 0; range < Math.max(1, ranges.length); range++) {
        const entry = ranges[range] ?? { elementCount: 0, firstElement: 0 };
        changed = this.indirect.writeCommand(base + range, entry.elementCount, entry.firstElement, levels[level].indexed) || changed;
      }
    }

    if (changed) this.indirect.upload();
  }

  /**
   * Uploads the frustum planes and LOD camera position for this dispatch,
   * both expressed in the mesh's object space so the kernel can use the
   * instance matrices directly.
   */
  public setCamera(camera: Camera, cameraLOD: Camera, matrixWorld: Matrix4, perObjectFrustumCulled: boolean, sphere: Sphere): void {
    // Reuses Three's own plane extraction so the GPU test cannot drift from
    // the CPU fallback, including the WebGPU clip-space and reversed-depth
    // cases the culling camera may be in.
    _projScreenMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(matrixWorld);
    _frustum.setFromProjectionMatrix(_projScreenMatrix, camera.coordinateSystem, (camera as any).reversedDepth);
    for (let i = 0; i < 6; i++) {
      const plane = _frustum.planes[i];
      this._planeUniforms[i].value.set(plane.normal.x, plane.normal.y, plane.normal.z, plane.constant);
    }

    _invMatrixWorld.copy(matrixWorld).invert();
    _cameraLODPos.setFromMatrixPosition(cameraLOD.matrixWorld).applyMatrix4(_invMatrixWorld);
    this._cameraPosUniform.value.copy(_cameraLODPos);

    this._sphereCenterUniform.value.copy(sphere.center);
    this._sphereRadiusUniform.value = sphere.radius;
    this._cullEnabledUniform.value = perObjectFrustumCulled ? 1 : 0;
  }

  /** Refreshes the LOD thresholds without rebuilding the compute graph. */
  public setLevelDistances(levels: CullingLevelDescriptor[]): void {
    for (let i = 0; i < this._thresholdUniforms.length; i++) {
      this._thresholdUniforms[i].value = effectiveThreshold(levels[i]);
    }
  }

  /**
   * Sizes the cull dispatch and returns the kernels to submit, or `null` when
   * there is nothing to cull. Submitting is the caller's job so that every
   * mesh in a pass can share one command buffer.
   */
  public prepare(instanceCount: number): any[] | null {
    if (this._cullNode === null || instanceCount === 0) return null;

    this._cullNode.count = instanceCount;
    return this._computeNodes;
  }

  public get levelCount(): number {
    return this._levelCount;
  }

  public get rangeCounts(): number[] {
    return this._rangeCounts;
  }

  public dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this.disposeKernels();
    this.indirect.dispose();
  }

  private disposeKernels(): void {
    this._resetNode?.dispose();
    this._cullNode?.dispose();
    this._publishNode?.dispose();
    this._resetNode = null;
    this._cullNode = null;
    this._publishNode = null;
    this._computeNodes = [];
  }
}

/**
 * Hysteresis is a fraction of the linear threshold, so it is applied before
 * squaring, exactly as `getObjectLODIndexForDistance` does on the CPU.
 */
function effectiveThreshold(level: CullingLevelDescriptor): number {
  const factor = 1 - level.hysteresis;
  return level.distanceSquared * factor * factor;
}
