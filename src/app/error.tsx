"use client"; // Error boundaries must be Client Components.

import { useEffect } from "react";

/**
 * What a render error shows. Without this file a production build shows a blank page;
 * here the message stays on screen so it can be reported.
 */
export default function Error({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  useEffect(() => {
    console.error("Unhandled error:", error);
  }, [error]);

  return (
    <div className="grid h-dvh place-items-center bg-background p-6 text-foreground">
      <div className="w-full max-w-lg space-y-4">
        <div className="space-y-1">
          <h1 className="text-base font-semibold">Something went wrong</h1>
          <p className="text-sm text-muted-foreground">This is a bug. The detail below says what failed.</p>
        </div>

        <pre className="max-h-48 overflow-auto rounded-md border border-border/60 bg-muted/40 p-3 font-mono text-[11px] whitespace-pre-wrap">
          {error.message || "No message"}
          {error.digest ? `\n\ndigest: ${error.digest}` : ""}
        </pre>

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => retry()}
            className="rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground"
          >
            Try again
          </button>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium"
          >
            Reload
          </button>
        </div>
      </div>
    </div>
  );
}
