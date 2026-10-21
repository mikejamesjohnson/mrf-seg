"use client";

/**
 * Browse this computer's folders for an image, from the toolbar. The editor's server lists
 * each folder (`/api/browse`): subfolders to walk into, and the images this editor reads to
 * pick. Picking one hands its path back, and the toolbar opens it like a typed path.
 */

import { ChevronUp, FileImage, Folder, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { listFolder, type BrowseListing } from "@/lib/editor/browse";

/** The folder at `path` (home without one); if it cannot be listed, home instead, under the message. */
async function fetchListing(path?: string): Promise<{ listing: BrowseListing | null; error: string | null }> {
  try {
    return { listing: await listFolder(path), error: null };
  } catch (failure) {
    const error = failure instanceof Error ? failure.message : String(failure);
    if (!path) return { listing: null, error };
    try {
      return { listing: await listFolder(), error };
    } catch {
      return { listing: null, error };
    }
  }
}

export function BrowseDialog({
  open,
  onOpenChange,
  startAt,
  onPick,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The folder to open at; null for the home folder. */
  startAt: string | null;
  onPick: (path: string) => void;
}) {
  // The folder walked to in this dialog, over `startAt`; forgotten when the dialog closes.
  const [folder, setFolder] = useState<string | null>(null);
  const [listing, setListing] = useState<BrowseListing | null>(null);
  const [error, setError] = useState<string | null>(null);

  const target = folder ?? startAt ?? undefined;
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void fetchListing(target).then((result) => {
      if (cancelled) return;
      setListing(result.listing);
      setError(result.error);
    });
    return () => {
      cancelled = true;
    };
  }, [open, target]);

  const close = (next: boolean) => {
    if (!next) {
      setFolder(null);
      setListing(null);
      setError(null);
    }
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={close}>
      <DialogContent>
        <DialogTitle>Open an image</DialogTitle>
        <DialogDescription className="sr-only">Folders on this computer; choose an image file to open it.</DialogDescription>
        <div className="flex items-center gap-2">
          <Button
            size="xs"
            variant="outline"
            disabled={!listing?.parent}
            onClick={() => {
              if (listing?.parent) setFolder(listing.parent);
            }}
            aria-label="Up one folder"
          >
            <ChevronUp />
            Up
          </Button>
          <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-muted-foreground" title={listing?.path}>
            {listing?.path ?? "…"}
          </span>
          {!listing && !error && <Loader2 className="size-3.5 shrink-0 animate-spin text-muted-foreground" />}
        </div>
        {error && <p className="text-[11px] leading-4 break-words text-destructive">{error}</p>}
        <ul className="max-h-80 overflow-y-auto rounded-md border border-border/60 text-xs" aria-label="Folder contents">
          {listing && listing.entries.length === 0 && <li className="px-2 py-3 text-center text-muted-foreground">No folders or images here.</li>}
          {listing?.entries.map((entry) => (
            <li key={entry.path}>
              <button
                type="button"
                className="flex w-full items-center gap-2 px-2 py-1 text-left hover:bg-muted focus-visible:bg-muted focus-visible:outline-none"
                onClick={() => {
                  if (entry.kind === "directory") setFolder(entry.path);
                  else {
                    onPick(entry.path);
                    close(false);
                  }
                }}
              >
                {entry.kind === "directory" ? <Folder className="size-3.5 shrink-0 text-muted-foreground" /> : <FileImage className="size-3.5 shrink-0 text-primary" />}
                <span className="truncate">{entry.name}</span>
              </button>
            </li>
          ))}
        </ul>
      </DialogContent>
    </Dialog>
  );
}
