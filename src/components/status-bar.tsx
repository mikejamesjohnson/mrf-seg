"use client";

import { AlertCircle, Loader2 } from "lucide-react";

import { datasetBytes } from "@/lib/scene/datasets";
import { useSceneStore } from "@/lib/scene/store";

export function StatusBar() {
  const status = useSceneStore((s) => s.status);
  const megabytes = useSceneStore((s) => Object.values(s.datasets).reduce((sum, dataset) => sum + datasetBytes(dataset), 0) / 1e6);

  return (
    <footer className="flex h-7 shrink-0 items-center gap-3 border-t border-border/60 px-3 text-[11px] text-muted-foreground">
      <div className="flex min-w-0 flex-1 items-center gap-1.5">
        {status.state === "running" && (
          <>
            <Loader2 className="size-3 shrink-0 animate-spin" />
            <span className="truncate">{status.message}</span>
          </>
        )}
        {status.state === "error" && (
          <>
            <AlertCircle className="size-3 shrink-0 text-destructive" />
            <span className="truncate text-destructive">{status.message}</span>
          </>
        )}
        {status.state === "idle" && <span>Ready</span>}
      </div>
      <div className="flex min-w-0 flex-1 items-center justify-end">
        {megabytes > 0 && <span className="font-mono">{megabytes.toFixed(0)} MB loaded</span>}
      </div>
    </footer>
  );
}
