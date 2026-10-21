"use client";

import { Box, FolderOpen, FolderSearch } from "lucide-react";
import { useState } from "react";

import { BrowseDialog } from "@/components/browse-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { openImage } from "@/lib/editor/openImage";
import { SUPPORTED_EXTENSIONS } from "@/lib/io/loaders";
import { currentImage } from "@/lib/scene/datasets";
import { useSceneStore } from "@/lib/scene/store";
import { WORLD_SPACE } from "@/lib/world";

/** The folder a path is in, for Browse to start at; null when the field holds no path. */
function folderOf(path: string): string | null {
  const cut = path.lastIndexOf("/");
  return cut > 0 ? path.slice(0, cut) : cut === 0 ? "/" : null;
}

export function Toolbar() {
  const loadedPath = useSceneStore((s) => currentImage(s)?.path ?? "");
  const loading = useSceneStore((s) => s.status.state === "running");
  // What the field shows: the user's typing, or else whatever is loaded.
  const [draft, setDraft] = useState<string | null>(null);
  const [browsing, setBrowsing] = useState(false);
  const value = draft ?? loadedPath;

  // The typed path stays until it opens, so a typo can be fixed rather than retyped.
  const submit = async (path = value) => {
    if (path.trim() && (await openImage(path))) setDraft(null);
  };

  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-border/60 px-3">
      <div className="flex shrink-0 items-center gap-2">
        <Box className="size-4 text-primary" />
        <span className="text-sm font-semibold">MRFSeg</span>
        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
          {WORLD_SPACE}
        </span>
      </div>

      <form
        className="flex min-w-0 flex-1 items-center gap-1.5"
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
      >
        <Input
          value={value}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={`Path to an image on this computer (${SUPPORTED_EXTENSIONS.join(", ")})`}
          spellCheck={false}
          className="h-7 max-w-3xl font-mono text-xs"
          aria-label="Image path"
        />
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="h-7 gap-1.5 px-2.5 text-xs font-medium"
          disabled={loading}
          onClick={() => setBrowsing(true)}
        >
          <FolderSearch className="size-3.5" />
          Browse
        </Button>
        <Button
          type="submit"
          size="sm"
          variant="secondary"
          className="h-7 gap-1.5 px-2.5 text-xs font-medium"
          disabled={loading || !value.trim()}
        >
          <FolderOpen className="size-3.5" />
          Open
        </Button>
      </form>

      <BrowseDialog
        open={browsing}
        onOpenChange={setBrowsing}
        startAt={folderOf(value.trim())}
        onPick={(path) => {
          setDraft(path);
          void submit(path);
        }}
      />
    </header>
  );
}
