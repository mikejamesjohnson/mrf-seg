/**
 * MetaImage, the format ITK writes: `.mha` (header and voxels in one file) or `.mhd` (header
 * only; voxels in a `.raw` or zlib-compressed `.zraw`). vtk.js has no reader, so this is a
 * small one: bytes in, voxels and geometry out, no vtk.js.
 *
 * **ITK's geometry convention is the app's**: LPS mm, `Offset` the world position of voxel
 * (0, 0, 0), and row n of `TransformMatrix` the direction of voxel axis n -- already the
 * layout vtk.js expects, `[i axis, j axis, k axis]`.
 */

import { IDENTITY_MAT3, type Mat3, type Vec3 } from "@/lib/world";

export type ScalarArray =
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

type ScalarArrayConstructor = {
  new (buffer: ArrayBufferLike, byteOffset?: number, length?: number): ScalarArray;
  new (length: number): ScalarArray;
  readonly BYTES_PER_ELEMENT: number;
};

/** MetaIO element types. `MET_LONG` is 32-bit in MetaIO; 64-bit integers are not supported. */
const ELEMENT_TYPES: Record<string, ScalarArrayConstructor> = {
  MET_CHAR: Int8Array,
  MET_UCHAR: Uint8Array,
  MET_SHORT: Int16Array,
  MET_USHORT: Uint16Array,
  MET_INT: Int32Array,
  MET_UINT: Uint32Array,
  MET_LONG: Int32Array,
  MET_ULONG: Uint32Array,
  MET_FLOAT: Float32Array,
  MET_DOUBLE: Float64Array,
};

export class MetaImageError extends Error {}

export interface MetaImageHeader {
  dimensions: Vec3;
  spacing: Vec3;
  /** World position (LPS mm) of voxel (0, 0, 0). */
  origin: Vec3;
  /** World directions of the i, j, k voxel axes, one per row (vtk.js layout). */
  direction: Mat3;
  elementType: string;
  compressed: boolean;
  bigEndian: boolean;
  /** `LOCAL` when the voxels follow the header in the same file; otherwise a file name. */
  dataFile: string;
  /** Bytes to skip at the start of a separate data file; -1 means "the data is at the end". */
  headerSize: number;
}

export const LOCAL_DATA = "LOCAL";

function numbers(value: string | undefined, count: number, key: string): number[] | null {
  if (value === undefined) return null;
  const parsed = value.trim().split(/\s+/).map(Number);
  if (parsed.length < count || parsed.slice(0, count).some((n) => !Number.isFinite(n))) {
    throw new MetaImageError(`MetaImage header: ${key} should be ${count} numbers, got "${value}"`);
  }
  return parsed.slice(0, count);
}

function flag(value: string | undefined): boolean {
  return value !== undefined && value.trim().toLowerCase() === "true";
}

/**
 * Parse the header at the start of `bytes`.
 *
 * Returns where the header ends, which for an `.mha` is where the voxels begin. The
 * header is ASCII `Key = Value` lines and **must end with `ElementDataFile`**; anything
 * after that line is data.
 */
export function readMetaImageHeader(bytes: Uint8Array): { header: MetaImageHeader; dataOffset: number } {
  // Latin-1 decodes one byte to one character, so string offsets are byte offsets.
  const text = new TextDecoder("latin1").decode(bytes.subarray(0, Math.min(bytes.length, 1 << 16)));
  const fields: Record<string, string> = {};
  let dataOffset = -1;

  const linePattern = /([^\r\n]*)(\r?\n|$)/g;
  let match: RegExpExecArray | null;
  while ((match = linePattern.exec(text)) !== null && match[0].length > 0) {
    const line = match[1];
    const equals = line.indexOf("=");
    if (equals > 0) {
      const key = line.slice(0, equals).trim().toLowerCase();
      fields[key] = line.slice(equals + 1).trim();
      if (key === "elementdatafile") {
        dataOffset = match.index + match[0].length;
        break;
      }
    }
  }
  if (dataOffset < 0) {
    throw new MetaImageError("Not a MetaImage file: no ElementDataFile line in the header");
  }

  if (fields.objecttype && fields.objecttype.toLowerCase() !== "image") {
    throw new MetaImageError(`MetaImage ObjectType is ${fields.objecttype}, not Image`);
  }
  if (fields.binarydata !== undefined && !flag(fields.binarydata)) {
    throw new MetaImageError("MetaImage with ASCII (BinaryData = False) voxels is not supported");
  }

  const nDims = Number(fields.ndims ?? 3);
  if (nDims !== 2 && nDims !== 3) {
    throw new MetaImageError(`MetaImage with NDims = ${fields.ndims} is not supported (2D or 3D only)`);
  }
  const pad = (values: number[] | null, fill: number): Vec3 =>
    values ? ([values[0], values[1], nDims === 3 ? values[2] : fill] as Vec3) : [fill, fill, fill];

  const dimensions = pad(numbers(fields.dimsize, nDims, "DimSize"), 1);
  if (!fields.dimsize || dimensions.some((n) => !Number.isInteger(n) || n < 1)) {
    throw new MetaImageError(`MetaImage header: invalid DimSize "${fields.dimsize ?? ""}"`);
  }

  const channels = Number(fields.elementnumberofchannels ?? 1);
  if (channels !== 1) {
    throw new MetaImageError(`MetaImage with ${channels} channels per voxel is not supported yet`);
  }

  const elementType = (fields.elementtype ?? "").toUpperCase();
  if (!ELEMENT_TYPES[elementType]) {
    throw new MetaImageError(`MetaImage ElementType "${fields.elementtype ?? ""}" is not supported`);
  }

  let direction: Mat3 = IDENTITY_MAT3;
  const matrix = fields.transformmatrix ?? fields.rotation ?? fields.orientation;
  if (matrix !== undefined) {
    const values = numbers(matrix, nDims * nDims, "TransformMatrix")!;
    direction =
      nDims === 3
        ? (values as Mat3)
        : [values[0], values[1], 0, values[2], values[3], 0, 0, 0, 1];
  }

  return {
    header: {
      dimensions,
      spacing: pad(numbers(fields.elementspacing ?? fields.elementsize, nDims, "ElementSpacing"), 1),
      origin: pad(numbers(fields.offset ?? fields.origin ?? fields.position, nDims, "Offset"), 0),
      direction,
      elementType,
      compressed: flag(fields.compresseddata),
      bigEndian: flag(fields.binarydatabyteordermsb ?? fields.elementbyteordermsb),
      dataFile: fields.elementdatafile,
      headerSize: Number(fields.headersize ?? 0),
    },
    dataOffset,
  };
}

/** The file holding the voxels, resolved against the header's own location. */
export function dataFilePath(headerPath: string, dataFile: string): string {
  if (dataFile.startsWith("/")) return dataFile;
  if (/[*%]|^LIST$/i.test(dataFile)) {
    throw new MetaImageError(`MetaImage with one file per slice (${dataFile}) is not supported`);
  }
  return headerPath.slice(0, headerPath.lastIndexOf("/") + 1) + dataFile;
}

/** zlib-inflate a byte stream (MetaImage `CompressedData = True`). */
export async function inflate(data: Uint8Array | ReadableStream<Uint8Array>): Promise<Uint8Array> {
  const source =
    data instanceof Uint8Array ? new Blob([data as Uint8Array<ArrayBuffer>]).stream() : data;
  // "deflate" in the Compression Streams API is the zlib format, which is what MetaIO
  // writes; "deflate-raw" would be the headerless one.
  const inflated = source.pipeThrough(new DecompressionStream("deflate") as ReadableWritablePair<Uint8Array, Uint8Array>);
  return new Uint8Array(await new Response(inflated).arrayBuffer());
}

/** Swap byte order in place, for big-endian (`BinaryDataByteOrderMSB = True`) data. */
function swapBytes(bytes: Uint8Array, size: number): void {
  for (let start = 0; start + size <= bytes.length; start += size) {
    for (let a = start, b = start + size - 1; a < b; a += 1, b -= 1) {
      const t = bytes[a];
      bytes[a] = bytes[b];
      bytes[b] = t;
    }
  }
}

/** Turn the (already inflated) voxel bytes into a typed array of the header's type. */
export function decodeVoxels(header: MetaImageHeader, bytes: Uint8Array): ScalarArray {
  const Type = ELEMENT_TYPES[header.elementType];
  const count = header.dimensions[0] * header.dimensions[1] * header.dimensions[2];
  const byteLength = count * Type.BYTES_PER_ELEMENT;

  // A header size only applies to an uncompressed, separate data file.
  let start = 0;
  if (!header.compressed && header.dataFile.toUpperCase() !== LOCAL_DATA) {
    start = header.headerSize < 0 ? bytes.length - byteLength : header.headerSize;
  }
  if (start < 0 || bytes.length - start < byteLength) {
    throw new MetaImageError(
      `MetaImage voxel data is ${bytes.length - Math.max(start, 0)} bytes; ` +
        `${header.dimensions.join(" × ")} ${header.elementType} needs ${byteLength}`,
    );
  }

  // A typed array must start on a multiple of its element size, so copy when the data
  // does not (an .mha's header can be any length).
  const offset = bytes.byteOffset + start;
  const voxels =
    offset % Type.BYTES_PER_ELEMENT === 0
      ? new Type(bytes.buffer, offset, count)
      : new Type(bytes.slice(start, start + byteLength).buffer, 0, count);
  if (header.bigEndian && Type.BYTES_PER_ELEMENT > 1) {
    swapBytes(new Uint8Array(voxels.buffer, voxels.byteOffset, byteLength), Type.BYTES_PER_ELEMENT);
  }
  return voxels;
}
