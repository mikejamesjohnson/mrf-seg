"use client"; // Error boundaries must be Client Components.

/**
 * The backstop for an error in the root layout itself, which `error.tsx` cannot catch.
 * It replaces the root layout, so it gets no global stylesheet: styling is inline.
 */
export default function GlobalError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          display: "grid",
          placeItems: "center",
          fontFamily: "system-ui, sans-serif",
          background: "#0b0b0d",
          color: "#e8e8ea",
        }}
      >
        <div style={{ maxWidth: "32rem", padding: "1.5rem" }}>
          <h1 style={{ fontSize: "1rem", margin: "0 0 0.5rem" }}>MRFSeg failed to start</h1>
          <pre
            style={{
              background: "#17171b",
              border: "1px solid #2a2a30",
              borderRadius: "0.5rem",
              padding: "0.75rem",
              fontSize: "0.7rem",
              whiteSpace: "pre-wrap",
              maxHeight: "12rem",
              overflow: "auto",
            }}
          >
            {error.message || "No message"}
            {error.digest ? `\n\ndigest: ${error.digest}` : ""}
          </pre>
          <button
            type="button"
            onClick={() => retry()}
            style={{
              marginTop: "1rem",
              background: "#e8e8ea",
              color: "#0b0b0d",
              border: 0,
              borderRadius: "0.5rem",
              padding: "0.4rem 0.8rem",
              fontSize: "0.8rem",
              fontWeight: 500,
              cursor: "pointer",
            }}
          >
            Try again
          </button>
        </div>
      </body>
    </html>
  );
}
