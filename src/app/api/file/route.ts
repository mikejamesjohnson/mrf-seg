/**
 * `GET /api/file?path=/absolute/path/to/image.mhd[&part=data]`: a local image's bytes, which is
 * how the editor opens an image by path (a browser cannot read the disk). Parsing stays in the
 * browser (`lib/io/`). Only image files are served, and a MetaImage data file only as the one
 * its `.mhd` header names (`part=data`), whatever it is called. Callers: `lib/server/local.ts`.
 */

import { open, type FileHandle } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";

import type { NextRequest } from "next/server";

import { dataFilePath, LOCAL_DATA, readMetaImageHeader } from "@/lib/io/metaImage";
import { refusal, textResponse as fail } from "@/lib/server/local";

const IMAGE_EXTENSIONS = new Set([".vti", ".mha", ".mhd"]);
/** Where a MetaImage header must end. */
const HEADER_BYTES = 1 << 16;

/** A file opened for reading, or the response that says why it cannot be. */
async function openFile(filePath: string): Promise<FileHandle | Response> {
  try {
    const handle = await open(filePath, "r");
    if ((await handle.stat()).isFile()) return handle;
    await handle.close();
    return fail(404, `Not a file: ${filePath}`);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "EACCES" || code === "EPERM") {
      return fail(403, `Permission denied: ${filePath} (on macOS, the terminal running the editor may need access to that folder)`);
    }
    return fail(404, `File not found: ${filePath}`);
  }
}

/** The data file the `.mhd` at `headerPath` names, or the response that says why there is none. */
async function dataFileOf(headerPath: string): Promise<string | Response> {
  const header = await openFile(headerPath);
  if (header instanceof Response) return header;
  try {
    const { buffer, bytesRead } = await header.read(Buffer.alloc(HEADER_BYTES), 0, HEADER_BYTES, 0);
    const { dataFile } = readMetaImageHeader(new Uint8Array(buffer.buffer, buffer.byteOffset, bytesRead)).header;
    if (dataFile.toUpperCase() === LOCAL_DATA) return fail(400, `${headerPath} keeps its voxels in the header file`);
    return path.resolve(dataFilePath(headerPath, dataFile));
  } catch (error) {
    return fail(415, error instanceof Error ? error.message : String(error));
  } finally {
    await header.close();
  }
}

export async function GET(request: NextRequest): Promise<Response> {
  const refused = refusal(request);
  if (refused) return fail(403, refused);

  const requested = request.nextUrl.searchParams.get("path");
  if (!requested) return fail(400, "Missing ?path=");
  if (!path.isAbsolute(requested)) return fail(400, `Not an absolute path: ${requested}`);
  let filePath = path.resolve(requested);
  if (!IMAGE_EXTENSIONS.has(path.extname(filePath).toLowerCase())) return fail(415, `Not an image file this editor reads: ${filePath}`);
  if (request.nextUrl.searchParams.get("part") === "data") {
    if (path.extname(filePath).toLowerCase() !== ".mhd") return fail(400, "Only a .mhd header names a separate data file");
    const dataFile = await dataFileOf(filePath);
    if (dataFile instanceof Response) return dataFile;
    filePath = dataFile;
  }

  const handle = await openFile(filePath);
  if (handle instanceof Response) return handle;
  const { size } = await handle.stat();
  return new Response(Readable.toWeb(handle.createReadStream()) as ReadableStream<Uint8Array>, {
    headers: { "Content-Type": "application/octet-stream", "Content-Length": String(size), "Cache-Control": "no-store" },
  });
}
