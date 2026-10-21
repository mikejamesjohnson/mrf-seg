/**
 * Guards for the routes that reach this computer, its disk (`/api/file`) and the backend
 * (`/api/mrf`). Server-side only. The server binds to 127.0.0.1 (the npm scripts pass
 * `-H 127.0.0.1`); on top of that a request must name a loopback host, which stops DNS
 * rebinding, and must come from the editor's own page, not from another site.
 */

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);

export function isLoopbackHost(host: string | null): boolean {
  if (!host) return false;
  // Strip the port. An IPv6 host is bracketed, so its colons are inside the brackets.
  const name = host.startsWith("[") ? host.slice(0, host.indexOf("]") + 1) : host.split(":")[0];
  return LOOPBACK_HOSTS.has(name.toLowerCase());
}

/**
 * Why a request may not be served, or null. Browsers say where a request comes from
 * (`Sec-Fetch-Site`, and `Origin` on anything but a plain GET): only the editor's own page,
 * or the address bar, gets through.
 */
export function refusal(request: Request): string | null {
  const headers = request.headers;
  if (!isLoopbackHost(headers.get("host"))) return "Only this computer is served.";
  const site = headers.get("sec-fetch-site");
  if (site && site !== "same-origin" && site !== "none") return "Only the editor's own page is served.";
  const origin = headers.get("origin");
  if (origin && originHost(origin) !== headers.get("host")) return "Only the editor's own page is served.";
  return null;
}

/** The host an `Origin` names; null for an opaque one (`Origin: null`, from a sandbox or a file). */
function originHost(origin: string): string | null {
  try {
    return new URL(origin).host;
  } catch {
    return null;
  }
}

export function textResponse(status: number, message: string): Response {
  return new Response(message, { status, headers: { "Content-Type": "text/plain; charset=utf-8" } });
}
