import { StorageBufferAttribute, type WebGPURenderer } from 'three/webgpu';

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

/**
 * CPU-backed float storage buffer with one contiguous dirty range.
 *
 * The buffer is suitable for matrices (`itemSize = 16`) and colors
 * (`itemSize = 4`). Calls to `enqueueUpdate()` only widen the pending range;
 * they never create one update range per instance.
 */
export class WebGPUFloatStorageBuffer {
  public _data: Float32Array;
  public attribute: StorageBufferAttribute;
  public readonly image: WebGPUStorageBufferImage;
  public readonly itemSize: number;
  /** Increments whenever `attribute` is replaced. */
  public revision = 0;

  private _dirtyStart = Infinity;
  private _dirtyEnd = 0;

  public constructor(itemSize: number, capacity: number, name = '') {
    assertPositiveInteger(itemSize, 'itemSize');
    assertNonNegativeInteger(capacity, 'capacity');

    this.itemSize = itemSize;
    this._data = new Float32Array(capacity * itemSize);
    this.attribute = createStorageAttribute(this._data, itemSize, name);
    this.image = this.createImageView();
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

    const data = new Float32Array(capacity * this.itemSize);
    data.set(this._data.subarray(0, Math.min(this._data.length, data.length)));
    this.replaceData(data);
  }

  public clone(): WebGPUFloatStorageBuffer {
    const clone = new WebGPUFloatStorageBuffer(this.itemSize, this.capacity, this.attribute.name);
    clone._data.set(this._data);
    return clone;
  }

  public dispose(): void {
    this.attribute.dispose();
    this.attribute.clearUpdateRanges();
    this.clearPendingRange();
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

  private replaceData(data: Float32Array): void {
    if (!(data instanceof Float32Array)) {
      throw new TypeError('WebGPUFloatStorageBuffer data must be a Float32Array.');
    }
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
 * CPU-written visible-instance indirection buffer for WebGPU.
 *
 * Its `array`, `_needsUpdate`, and `update(renderer, count)` surface mirrors
 * the existing WebGL index attribute closely enough for the shared culling
 * and LOD code. Each update uploads one `[0, count)` range.
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

function createStorageAttribute<TArray extends Float32Array | Uint32Array>(array: TArray, itemSize: number, name: string): StorageBufferAttribute {
  const attribute = new StorageBufferAttribute(array, itemSize);
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
