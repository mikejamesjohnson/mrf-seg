/**
 * Walking this computer's folders from the browser, for the toolbar's Browse: the listing the
 * editor's `/api/browse` route gives, and the call that fetches it. A browser's own file dialog
 * never reveals a path, and the backend opens images by path, so the editor's server lists the
 * folders instead.
 */

/** One thing in a folder, as the dialog lists it. */
export interface BrowseEntry {
  name: string;
  path: string;
  kind: "directory" | "image";
}

export interface BrowseListing {
  /** The folder listed, resolved. */
  path: string;
  /** Its parent, or null at the root. */
  parent: string | null;
  /** Folders first, then images, each alphabetically. */
  entries: BrowseEntry[];
}

/** The folder at `path`, or the home folder without one. Throws with the server's message. */
export async function listFolder(path?: string): Promise<BrowseListing> {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  const response = await fetch(`/api/browse${query}`, { cache: "no-store" });
  if (!response.ok) throw new Error(await response.text());
  return (await response.json()) as BrowseListing;
}
