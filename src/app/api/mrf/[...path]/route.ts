/**
 * `/api/mrf/...`: the segmentation backend (`python/mrfseg/server/`, whose docstring lists the
 * endpoints), passed through. The browser only talks to this server, behind the same guard as
 * the file route (`lib/server/local.ts`), and a POST must be JSON, so another site cannot send
 * one without a preflight. Responses stream through, never held whole.
 */

import type { NextRequest } from "next/server";

import { refusal } from "@/lib/server/local";

const BACKEND = `http://127.0.0.1:${process.env.MRFSEG_BACKEND_PORT ?? "8004"}`;

/** The backend's own errors are `{"error": ...}`; these are shaped the same. */
function error(status: number, message: string): Response {
  return Response.json({ error: message }, { status });
}

async function forward(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }): Promise<Response> {
  const refused = refusal(request);
  if (refused) return error(403, refused);
  const hasBody = request.method === "POST";
  if (hasBody && !request.headers.get("content-type")?.startsWith("application/json")) return error(415, "Requests must be JSON.");

  const { path } = await params;
  const target = `${BACKEND}/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`;
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers: hasBody ? { "Content-Type": "application/json" } : undefined,
      body: hasBody ? await request.text() : undefined,
      cache: "no-store",
    });
  } catch {
    return error(
      503,
      "The segmentation backend is not running. `npm run dev` starts it, from python/.venv with " +
        "python/requirements.txt installed; its terminal says why it stopped.",
    );
  }

  const headers = new Headers({ "Cache-Control": "no-store" });
  for (const name of ["content-type", "content-length"]) {
    const value = upstream.headers.get(name);
    if (value) headers.set(name, value);
  }
  const empty = upstream.status === 204 || upstream.status === 304;
  return new Response(empty ? null : upstream.body, { status: upstream.status, headers });
}

export const GET = forward;
export const POST = forward;
export const DELETE = forward;
