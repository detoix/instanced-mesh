import { IndirectStorageBufferAttribute, StorageBufferAttribute, type WebGPURenderer } from 'three/webgpu';

/**
 * Minimal image-shaped view used by {@link WebGPUFloatStorageBuffer}.
 *
 * `InstancedMesh2` historically stores instance data in a `DataTexture`, so
 * some of its common code accesses `image.data`. The accessor below keeps
 * that API without allocating a texture or duplicating the backing array.
 */
export interface WebGPUStorageBufferImage {
  data: Float32Array;
  readonly width: number;
  readonly height: number;
}

type StorageArray = Float32Array | Uint32Array;

/**
 * CPU-backed storage buffer with one contiguous dirty range.
 *
 * Calls to `enqueueUpdate()` only widen the pending range; they never create
 * one update range per instance.
 */
abstract class WebGPUStorageBuffer<TArray extends StorageArray> {
  public _data: TArray;
  public attribute: StorageBufferAttribute;
  public readonly itemSize: number;
  /** Increments whenever `attribute` is replaced. */
  public revision = 0;

  private _dirtyStart = Infinity;
  private _dirtyEnd = 0;

  public constructor(itemSize: number, capacity: number, name: string) {
    assertPositiveInteger(itemSize, 'itemSize');
    assertNonNegativeInteger(capacity, 'capacity');

    this.itemSize = itemSize;
    this._data = this.allocate(capacity * itemSize);
    this.attribute = createStorageAttribute(this._data, itemSize, name);
  }

  /** Number of logical items currently allocated. */
  public get capacity(): number {
    return this._data.length / this.itemSize;
  }

  /**
   * Marks a logical item as changed. Multiple calls are coalesced into one
   * component range by `flush()`.
   */
  public enqueueUpdate(index: number): void {
    assertIndex(index, this.capacity);

    const start = index * this.itemSize;
    this._dirtyStart = Math.min(this._dirtyStart, start);
    this._dirtyEnd = Math.max(this._dirtyEnd, start + this.itemSize);
  }

  /** Marks every allocated item as changed. */
  public enqueueFullUpdate(): void {
    if (this._data.length === 0) return;

    this._dirtyStart = 0;
    this._dirtyEnd = this._data.length;
  }

  /**
   * Publishes the pending CPU changes to Three.js.
   *
   * `count` optionally limits the upload to the first N logical items. Any
   * pending tail remains queued for a later flush. Existing, not-yet-consumed
   * Three.js update ranges are merged so there is always at most one range.
   *
   * @returns `true` when an upload was scheduled.
   */
  public flush(count = this.capacity): boolean {
    assertCount(count, this.capacity);

    if (this._dirtyStart === Infinity) return false;

    const componentLimit = count * this.itemSize;
    const uploadEnd = Math.min(this._dirtyEnd, componentLimit);
    if (this._dirtyStart >= uploadEnd) return false;

    let uploadStart = this._dirtyStart;
    let mergedEnd = uploadEnd;

    for (const range of this.attribute.updateRanges) {
      uploadStart = Math.min(uploadStart, range.start);
      mergedEnd = Math.max(mergedEnd, range.start + range.count);
    }

    this.attribute.clearUpdateRanges();
    this.attribute.addUpdateRange(uploadStart, mergedEnd - uploadStart);
    this.attribute.needsUpdate = true;

    if (uploadEnd < this._dirtyEnd) {
      this._dirtyStart = uploadEnd;
    } else {
      this.clearPendingRange();
    }

    return true;
  }

  /**
   * Reallocates the storage attribute while preserving the overlapping data.
   * Consumers holding a TSL storage node must rebuild it when `revision`
   * changes because the `attribute` identity changes.
   */
  public resize(capacity: number): void {
    assertNonNegativeInteger(capacity, 'capacity');
    if (capacity === this.capacity) return;

    const data = this.allocate(capacity * this.itemSize);
    data.set(this._data.subarray(0, Math.min(this._data.length, data.length)) as any);
    this.replaceData(data);
  }

  public dispose(): void {
    this.attribute.dispose();
    this.attribute.clearUpdateRanges();
    this.clearPendingRange();
  }

  protected abstract allocate(length: number): TArray;

  protected replaceData(data: TArray): void {
    if (data.length % this.itemSize !== 0) {
      throw new RangeError(`Storage data length must be divisible by itemSize (${this.itemSize}).`);
    }
    if (data === this._data) return;

    const previousAttribute = this.attribute;
    this._data = data;
    this.attribute = createStorageAttribute(data, this.itemSize, previousAttribute.name);
    this.revision++;
    this.clearPendingRange();
    previousAttribute.dispose();
  }

  private clearPendingRange(): void {
    this._dirtyStart = Infinity;
    this._dirtyEnd = 0;
  }
}

/**
 * Float storage buffer, suitable for matrices (`itemSize = 16`) and colors
 * (`itemSize = 4`).
 */
export class WebGPUFloatStorageBuffer extends WebGPUStorageBuffer<Float32Array> {
  public readonly image: WebGPUStorageBufferImage;

  public constructor(itemSize: number, capacity: number, name = '') {
    super(itemSize, capacity, name);
    this.image = this.createImageView();
  }

  public clone(): WebGPUFloatStorageBuffer {
    const clone = new WebGPUFloatStorageBuffer(this.itemSize, this.capacity, this.attribute.name);
    clone._data.set(this._data);
    return clone;
  }

  protected override allocate(length: number): Float32Array {
    return new Float32Array(length);
  }

  protected override replaceData(data: Float32Array): void {
    if (!(data instanceof Float32Array)) {
      throw new TypeError('WebGPUFloatStorageBuffer data must be a Float32Array.');
    }
    super.replaceData(data);
  }

  private createImageView(): WebGPUStorageBufferImage {
    const owner = this;
    const image = {} as WebGPUStorageBufferImage;

    Object.defineProperties(image, {
      data: {
        enumerable: true,
        get(): Float32Array {
          return owner._data;
        },
        set(value: Float32Array): void {
          owner.replaceData(value);
        }
      },
      width: {
        enumerable: true,
        get(): number {
          return owner.capacity;
        }
      },
      height: {
        enumerable: true,
        get(): number {
          return 1;
        }
      }
    });

    return image;
  }
}

/** Bit 0 of an instance state word: the instance is visible. */
export const INSTANCE_STATE_VISIBLE = 1;
/** Bit 1 of an instance state word: the instance is active (not deleted). */
export const INSTANCE_STATE_ACTIVE = 2;
/** Bit offset of the render-pass LOD override field (0 means "no override"). */
export const INSTANCE_STATE_RENDER_LOD_SHIFT = 8;
/** Bit offset of the shadow-pass LOD override field (0 means "no override"). */
export const INSTANCE_STATE_SHADOW_LOD_SHIFT = 16;
/** Mask of one LOD override field. */
export const INSTANCE_STATE_LOD_MASK = 0xff;

/**
 * One `u32` of GPU-readable per-instance state: the active and visible flags
 * the CPU already tracks in `availabilityArray`, plus optional per-pass LOD
 * overrides. See `docs/webgpu-architecture.md`.
 */
export class WebGPUInstanceStateBuffer extends WebGPUStorageBuffer<Uint32Array> {
  public constructor(capacity: number, name = 'ezInstanceState') {
    super(1, capacity, name);
  }

  public setFlag(index: number, flag: number, value: boolean): void {
    const previous = this._data[index];
    const next = value ? previous | flag : previous & ~flag;
    if (next === previous) return;
    this._data[index] = next;
    this.enqueueUpdate(index);
  }

  public setFlags(index: number, mask: number, value: boolean): void {
    const previous = this._data[index];
    const next = value ? previous | mask : previous & ~mask;
    if (next === previous) return;
    this._data[index] = next;
    this.enqueueUpdate(index);
  }

  /** `level < 0` clears the override and restores distance-based selection. */
  public setLODOverride(index: number, level: number, shift: number): void {
    const stored = level < 0 ? 0 : Math.min(level + 1, INSTANCE_STATE_LOD_MASK);
    const previous = this._data[index];
    const next = (previous & ~(INSTANCE_STATE_LOD_MASK << shift)) | (stored << shift);
    if (next === previous) return;
    this._data[index] = next;
    this.enqueueUpdate(index);
  }

  public getLODOverride(index: number, shift: number): number {
    return ((this._data[index] >>> shift) & INSTANCE_STATE_LOD_MASK) - 1;
  }

  protected override allocate(length: number): Uint32Array {
    return new Uint32Array(length);
  }
}

/** `u32` words in one indexed indirect draw command. */
export const DRAW_COMMAND_WORDS = 5;
/** Bytes in one indirect draw command slot. */
export const DRAW_COMMAND_BYTES = DRAW_COMMAND_WORDS * Uint32Array.BYTES_PER_ELEMENT;

/**
 * Indirect draw commands whose `instanceCount` word is written by the culling
 * compute pass rather than by the CPU.
 *
 * A single 5-word stride serves both layouts. WebGPU reads
 * `(indexCount, instanceCount, firstIndex, baseVertex, firstInstance)` for an
 * indexed draw and `(vertexCount, instanceCount, firstVertex, firstInstance)`
 * for a non-indexed one, so word 1 is `instanceCount` either way and the
 * shader never needs to know which geometry it is looking at.
 */
export class WebGPUIndirectDrawBuffer {
  public attribute: IndirectStorageBufferAttribute;
  /** Increments whenever `attribute` is replaced. */
  public revision = 0;

  private _data: Uint32Array;

  public constructor(commandCount: number, name = 'ezIndirectDraws') {
    assertPositiveInteger(commandCount, 'commandCount');
    this._data = new Uint32Array(commandCount * DRAW_COMMAND_WORDS);
    this.attribute = createIndirectAttribute(this._data, name);
  }

  public get commandCount(): number {
    return this._data.length / DRAW_COMMAND_WORDS;
  }

  public byteOffset(command: number): number {
    return command * DRAW_COMMAND_BYTES;
  }

  public setCommandCount(commandCount: number): void {
    assertPositiveInteger(commandCount, 'commandCount');
    if (commandCount === this.commandCount) return;

    const previousAttribute = this.attribute;
    this._data = new Uint32Array(commandCount * DRAW_COMMAND_WORDS);
    this.attribute = createIndirectAttribute(this._data, previousAttribute.name);
    this.revision++;
    previousAttribute.dispose();
  }

  /**
   * Writes the geometry-dependent words of one command. `instanceCount` is
   * deliberately left at zero: the compute pass owns it, and re-uploading it
   * would race with the counts already on the GPU.
   *
   * @returns `true` when the command actually changed.
   */
  public writeCommand(command: number, elementCount: number, firstElement: number, _indexed: boolean): boolean {
    // The remaining words -- baseVertex and firstInstance when indexed,
    // firstInstance when not -- are zero in both layouts.
    const base = command * DRAW_COMMAND_WORDS;
    const data = this._data;
    if (data[base] === elementCount && data[base + 2] === firstElement) return false;

    data[base] = elementCount;
    data[base + 2] = firstElement;
    return true;
  }

  public upload(): void {
    this.attribute.needsUpdate = true;
  }

  public dispose(): void {
    this.attribute.dispose();
  }
}

/**
 * CPU-written visible-instance indirection buffer for WebGPU.
 *
 * Its `array`, `_needsUpdate`, and `update(renderer, count)` surface mirrors
 * the existing WebGL index attribute closely enough for the shared culling
 * and LOD code. Each update uploads one `[0, count)` range. On the GPU-driven
 * path the same attribute is written by a compute shader instead and no CPU
 * upload happens at all.
 */
export class WebGPUVisibleIndexBuffer {
  public attribute: StorageBufferAttribute;
  public _needsUpdate = true;
  /** Increments whenever `attribute` is replaced. */
  public revision = 0;

  private _array: Uint32Array;

  public constructor(capacity: number, name = 'ezInstanceIndex') {
    assertNonNegativeInteger(capacity, 'capacity');

    this._array = new Uint32Array(capacity);
    fillIdentity(this._array);
    this.attribute = createStorageAttribute(this._array, 1, name);
  }

  public get array(): Uint32Array {
    return this._array;
  }

  public set array(value: Uint32Array) {
    this.replaceArray(value);
  }

  public get capacity(): number {
    return this._array.length;
  }

  /**
   * Reallocates the storage attribute, preserves existing indices, and fills
   * newly allocated entries with their identity index.
   */
  public resize(capacity: number): void {
    assertNonNegativeInteger(capacity, 'capacity');
    if (capacity === this.capacity) return;

    const previousCapacity = this.capacity;
    const array = new Uint32Array(capacity);
    array.set(this._array.subarray(0, Math.min(previousCapacity, capacity)));

    for (let i = previousCapacity; i < capacity; i++) {
      array[i] = i;
    }

    this.replaceArray(array);
  }

  /**
   * Schedules one contiguous upload for the visible prefix. The renderer
   * parameter is intentionally unused; it is retained for WebGL API parity.
   */
  public update(_renderer: WebGPURenderer | null | undefined, count: number): boolean {
    if (!this._needsUpdate || count === 0) return false;
    assertCount(count, this.capacity);

    this.attribute.clearUpdateRanges();
    this.attribute.addUpdateRange(0, count);
    this.attribute.needsUpdate = true;
    this._needsUpdate = false;
    return true;
  }

  /** Renderer-independent alias for `update()`. */
  public flush(count: number): boolean {
    return this.update(undefined, count);
  }

  public clone(): WebGPUVisibleIndexBuffer {
    const clone = new WebGPUVisibleIndexBuffer(this.capacity, this.attribute.name);
    clone._array.set(this._array);
    clone._needsUpdate = this._needsUpdate;
    return clone;
  }

  public dispose(): void {
    this.attribute.dispose();
    this.attribute.clearUpdateRanges();
    this._needsUpdate = false;
  }

  private replaceArray(array: Uint32Array): void {
    if (!(array instanceof Uint32Array)) {
      throw new TypeError('WebGPUVisibleIndexBuffer array must be a Uint32Array.');
    }
    if (array === this._array) return;

    const previousAttribute = this.attribute;
    const previousCapacity = this.capacity;
    for (let i = previousCapacity; i < array.length; i++) {
      array[i] = i;
    }
    this._array = array;
    this.attribute = createStorageAttribute(array, 1, previousAttribute.name);
    this._needsUpdate = true;
    this.revision++;
    previousAttribute.dispose();
  }
}

function createStorageAttribute<TArray extends StorageArray>(array: TArray, itemSize: number, name: string): StorageBufferAttribute {
  const attribute = new StorageBufferAttribute(array, itemSize);
  attribute.name = name;
  return attribute;
}

/**
 * The attribute is declared one `u32` per item on purpose. A struct-typed
 * storage binding whose array holds exactly one element is emitted by Three as
 * a bare struct rather than a runtime-sized array, which the compute kernel
 * then cannot index. Words are addressed explicitly instead.
 */
function createIndirectAttribute(array: Uint32Array, name: string): IndirectStorageBufferAttribute {
  const attribute = new IndirectStorageBufferAttribute(array, 1);
  attribute.name = name;
  return attribute;
}

function fillIdentity(array: Uint32Array): void {
  for (let i = 0; i < array.length; i++) {
    array[i] = i;
  }
}

function assertPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new RangeError(`${name} must be a positive integer.`);
  }
}

function assertNonNegativeInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative integer.`);
  }
}

function assertIndex(index: number, capacity: number): void {
  if (!Number.isInteger(index) || index < 0 || index >= capacity) {
    throw new RangeError(`index must be an integer in [0, ${capacity}).`);
  }
}

function assertCount(count: number, capacity: number): void {
  if (!Number.isInteger(count) || count < 0 || count > capacity) {
    throw new RangeError(`count must be an integer in [0, ${capacity}].`);
  }
}
