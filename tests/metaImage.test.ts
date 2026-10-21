import { describe, expect, it } from "vitest";

import { dataFilePath, decodeVoxels, readMetaImageHeader } from "@/lib/io/metaImage";

const encode = (text: string) => new TextEncoder().encode(text);

/** An `.mha`: the header, then the voxels. */
function mha(header: string, voxels: Uint8Array): Uint8Array {
  const head = encode(header);
  const bytes = new Uint8Array(head.length + voxels.length);
  bytes.set(head);
  bytes.set(voxels, head.length);
  return bytes;
}

const HEADER = [
  "ObjectType = Image",
  "NDims = 3",
  "BinaryData = True",
  "BinaryDataByteOrderMSB = False",
  "TransformMatrix = 0 1 0 -1 0 0 0 0 1",
  "Offset = -1.5 2 3.25",
  "ElementSpacing = 0.5 0.5 2",
  "DimSize = 2 3 1",
  "ElementType = MET_SHORT",
  "ElementDataFile = LOCAL",
  "",
].join("\n");

describe("readMetaImageHeader", () => {
  it("reads the geometry as written: Offset is voxel 0, each matrix row a voxel axis", () => {
    const bytes = mha(HEADER, new Uint8Array(12));
    const { header, dataOffset } = readMetaImageHeader(bytes);
    expect(dataOffset).toBe(encode(HEADER).length);
    expect(header.dimensions).toEqual([2, 3, 1]);
    expect(header.spacing).toEqual([0.5, 0.5, 2]);
    expect(header.origin).toEqual([-1.5, 2, 3.25]);
    expect(header.direction).toEqual([0, 1, 0, -1, 0, 0, 0, 0, 1]);
    expect(header.elementType).toBe("MET_SHORT");
    expect(header.dataFile).toBe("LOCAL");
    expect(header.compressed).toBe(false);
  });

  it("pads a 2D image to one slice, and takes CRLF line ends", () => {
    const text = "NDims = 2\r\nDimSize = 4 5\r\nElementSpacing = 2 3\r\nElementType = MET_UCHAR\r\nElementDataFile = image.raw\r\n";
    const { header } = readMetaImageHeader(encode(text));
    expect(header.dimensions).toEqual([4, 5, 1]);
    expect(header.spacing).toEqual([2, 3, 1]);
    expect(header.origin).toEqual([0, 0, 0]);
    expect(header.dataFile).toBe("image.raw");
  });

  it("refuses what it cannot read, saying why", () => {
    expect(() => readMetaImageHeader(encode("NDims = 3\n"))).toThrow(/ElementDataFile/);
    expect(() => readMetaImageHeader(encode("DimSize = 2 2 2\nElementType = MET_LONG_LONG\nElementDataFile = LOCAL\n"))).toThrow(/ElementType/);
    expect(() => readMetaImageHeader(encode("DimSize = 2 x 2\nElementType = MET_SHORT\nElementDataFile = LOCAL\n"))).toThrow(/DimSize/);
    expect(() =>
      readMetaImageHeader(encode("DimSize = 2 2 2\nElementNumberOfChannels = 3\nElementType = MET_SHORT\nElementDataFile = LOCAL\n")),
    ).toThrow(/channels/);
  });
});

describe("decodeVoxels", () => {
  it("reads little-endian voxels at any offset, and swaps big-endian ones", () => {
    // 2 x 3 x 1 int16: 1, -2, 300, 4, 5, 6, after a header of odd length.
    const values = new Int16Array([1, -2, 300, 4, 5, 6]);
    const little = mha(HEADER, new Uint8Array(values.buffer));
    const { header, dataOffset } = readMetaImageHeader(little);
    expect(dataOffset % 2).toBe(1);
    expect([...decodeVoxels(header, little.subarray(dataOffset))]).toEqual([1, -2, 300, 4, 5, 6]);

    const big = new Uint8Array(values.buffer.slice(0));
    for (let i = 0; i < big.length; i += 2) [big[i], big[i + 1]] = [big[i + 1], big[i]];
    expect([...decodeVoxels({ ...header, bigEndian: true }, big)]).toEqual([1, -2, 300, 4, 5, 6]);
  });

  it("says how much data there is when it is too little", () => {
    const { header } = readMetaImageHeader(mha(HEADER, new Uint8Array(0)));
    expect(() => decodeVoxels(header, new Uint8Array(10))).toThrow(/10 bytes.*needs 12/);
  });

  it("skips a separate data file's header, or takes the data from its end", () => {
    const { header } = readMetaImageHeader(encode(HEADER.replace("LOCAL", "image.raw") + ""));
    const file = new Uint8Array([9, 9, 9, ...new Uint8Array(new Int16Array([1, 2, 3, 4, 5, 6]).buffer)]);
    expect([...decodeVoxels({ ...header, headerSize: 3 }, file)]).toEqual([1, 2, 3, 4, 5, 6]);
    expect([...decodeVoxels({ ...header, headerSize: -1 }, file)]).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe("dataFilePath", () => {
  it("resolves the data file next to its header, and refuses a file per slice", () => {
    expect(dataFilePath("/data/ct/scan.mhd", "scan.zraw")).toBe("/data/ct/scan.zraw");
    expect(dataFilePath("/data/ct/scan.mhd", "/elsewhere/scan.raw")).toBe("/elsewhere/scan.raw");
    expect(() => dataFilePath("/data/scan.mhd", "LIST")).toThrow(/one file per slice/);
    expect(() => dataFilePath("/data/scan.mhd", "slice%03d.raw")).toThrow(/one file per slice/);
  });
});
