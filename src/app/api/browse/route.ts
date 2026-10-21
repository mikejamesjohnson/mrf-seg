/**
 * `GET /api/browse?path=/absolute/directory`: what is in a folder on this computer -- its
 * subfolders and the images this editor reads -- so the toolbar's Browse can walk to an image
 * and open it by path, which a browser's own file dialog cannot give. Without `?path=`, the
 * home folder. Hidden entries are left out. Caller: `lib/editor/browse.ts`.
 */

import type { Dirent } from "node:fs";
import { readdir, realpath, stat } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";

import type { NextRequest } from "next/server";

import type { BrowseEntry, BrowseListing } from "@/lib/editor/browse";
import { refusal, textResponse as fail } from "@/lib/server/local";

const IMAGE_EXTENSIONS = new Set([".vti", ".mha", ".mhd"]);

export async function GET(request: NextRequest): Promise<Response> {
  const refused = refusal(request);
  if (refused) return fail(403, refused);

  const requested = request.nextUrl.searchParams.get("path") || homedir();
  if (!path.isAbsolute(requested)) return fail(400, `Not an absolute path: ${requested}`);
  let directory: string;
  try {
    directory = await realpath(requested);
    if (!(await stat(directory)).isDirectory()) return fail(400, `Not a folder: ${requested}`);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "EACCES" || code === "EPERM") return fail(403, `Permission denied: ${requested}`);
    return fail(404, `Folder not found: ${requested}`);
  }

  let found: Dirent[];
  try {
    found = await readdir(directory, { withFileTypes: true });
  } catch {
    return fail(403, `Cannot list ${directory} (on macOS, the terminal running the editor may need access to that folder)`);
  }

  const entries: BrowseEntry[] = [];
  for (const entry of found) {
    if (entry.name.startsWith(".")) continue;
    const full = path.join(directory, entry.name);
    let isDirectory = entry.isDirectory();
    let isFile = entry.isFile();
    if (entry.isSymbolicLink()) {
      try {
        const target = await stat(full);
        isDirectory = target.isDirectory();
        isFile = target.isFile();
      } catch {
        continue; // a dangling link
      }
    }
    if (isDirectory) entries.push({ name: entry.name, path: full, kind: "directory" });
    else if (isFile && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) entries.push({ name: entry.name, path: full, kind: "image" });
  }
  const byName = (a: BrowseEntry, b: BrowseEntry) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" });
  entries.sort((a, b) => (a.kind === b.kind ? byName(a, b) : a.kind === "directory" ? -1 : 1));

  const parent = path.dirname(directory);
  const listing: BrowseListing = { path: directory, parent: parent === directory ? null : parent, entries };
  return Response.json(listing, { headers: { "Cache-Control": "no-store" } });
}
