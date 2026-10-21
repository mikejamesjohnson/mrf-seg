"use client";

import { useEffect } from "react";

import { IMAGE_URL_PARAM, openImage } from "@/lib/editor/openImage";

/**
 * Open the image named in the URL (`?image=/path/to/file`) once, when the editor starts. Once:
 * in development React runs the effect twice, and the first run's load is cancelled by the
 * cleanup rather than finished and thrown away.
 */
export function useImageFromUrl(): void {
  useEffect(() => {
    const path = new URLSearchParams(window.location.search).get(IMAGE_URL_PARAM);
    if (!path) return;
    const timer = setTimeout(() => void openImage(path), 0);
    return () => clearTimeout(timer);
  }, []);
}
