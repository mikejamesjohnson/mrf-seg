"use client";

import dynamic from "next/dynamic";
import { useEffect } from "react";

import { GeneratePanel } from "@/components/panels/generate-panel";
import { ImagePanel } from "@/components/panels/image-panel";
import { PointsPanel } from "@/components/panels/points-panel";
import { FeaturesPanel, InferencePanel, PairwisePanel, UnaryPanel } from "@/components/panels/segmentation-panel";
import { StatusBar } from "@/components/status-bar";
import { Toolbar } from "@/components/toolbar";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useImageFromUrl } from "@/hooks/use-image-from-url";
import { usePanelLayout } from "@/hooks/use-panel-layout";
import { usePointShortcuts } from "@/hooks/use-point-shortcuts";
import { installDebugHook } from "@/lib/editor/debug";
import { useSceneStore } from "@/lib/scene/store";
import { installAutoUpdate } from "@/lib/segmentation/autoUpdate";
import { installExtractors } from "@/lib/segmentation/extractors";
import { installSessionLifecycle } from "@/lib/segmentation/session";

/** vtk.js rendering touches WebGL at import time, so the viewports stay out of the server bundle. */
const ViewportGrid = dynamic(() => import("@/components/viewer/viewport-grid").then((m) => m.ViewportGrid), {
  ssr: false,
  loading: () => (
    <div className="grid h-full place-items-center text-xs text-muted-foreground">Starting renderer…</div>
  ),
});

/** One side-panel section: a header over its body. The panel scrolls as a whole. */
function PanelSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="flex h-8 items-center border-y border-border/40 bg-muted/20 px-3 text-[10px] font-semibold tracking-wider text-foreground/70 uppercase">
        {title}
      </h2>
      <div className="p-3">{children}</div>
    </section>
  );
}

/** Shown over the empty viewports until an image is open. */
function EmptyState() {
  const empty = useSceneStore((s) => s.datasetOrder.length === 0 && s.status.state !== "running");
  if (!empty) return null;
  return (
    <div className="pointer-events-none absolute inset-0 grid place-items-center">
      <div className="max-w-md rounded-lg border border-border/60 bg-background/85 p-4 text-center backdrop-blur-sm">
        <p className="text-sm font-medium">No image open</p>
        <p className="mt-1 text-xs text-muted-foreground">
          Browse for an image on this computer, or type its path in the bar above and press Open —
          or start the editor with <span className="font-mono">/editor?image=/path/to/image.mhd</span>.
        </p>
      </div>
    </div>
  );
}

export function EditorShell() {
  const layout = usePanelLayout("editor-shell");
  useImageFromUrl();
  usePointShortcuts();
  useEffect(installDebugHook, []);
  useEffect(installSessionLifecycle, []);
  useEffect(installAutoUpdate, []);
  useEffect(installExtractors, []);

  return (
    <div className="flex h-dvh flex-col bg-background text-foreground">
      <Toolbar />

      <ResizablePanelGroup
        orientation="horizontal"
        className="min-h-0 flex-1"
        defaultLayout={layout.defaultLayout}
        onLayoutChanged={layout.onLayoutChanged}
      >
        {/* No default size: the viewports take whatever the side panel leaves. */}
        <ResizablePanel id="viewport-panel" minSize="30%">
          <main className="relative h-full min-h-0 min-w-0">
            <ViewportGrid />
            <EmptyState />
          </main>
        </ResizablePanel>
        <ResizableHandle withHandle />

        <ResizablePanel
          id="image-panel"
          defaultSize="300px"
          minSize="240px"
          maxSize="40%"
          groupResizeBehavior="preserve-pixel-size"
        >
          <ScrollArea className="h-full">
            <PanelSection title="Image">
              <ImagePanel />
            </PanelSection>
            <PanelSection title="Feature extraction">
              <FeaturesPanel />
            </PanelSection>
            <PanelSection title="Unary potential">
              <UnaryPanel />
            </PanelSection>
            <PanelSection title="Pairwise potential">
              <PairwisePanel />
            </PanelSection>
            <PanelSection title="Inference">
              <InferencePanel />
            </PanelSection>
            <PanelSection title="Points">
              <PointsPanel />
            </PanelSection>
            <PanelSection title="Generate">
              <GeneratePanel />
            </PanelSection>
          </ScrollArea>
        </ResizablePanel>
      </ResizablePanelGroup>

      <StatusBar />
    </div>
  );
}
