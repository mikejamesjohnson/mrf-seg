export const size = {
  width: 32,
  height: 32,
};

export const contentType = "image/svg+xml";

/** Three orthogonal slice planes through a volume. */
export default function Icon() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none">
  <g stroke="#3b9dff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="m21 8-9-5-9 5 9 5 9-5Z"/>
    <path d="M3 8v8l9 5 9-5V8"/>
    <path d="M12 13v8"/>
  </g>
</svg>`;

  return new Response(svg, {
    headers: {
      "Content-Type": contentType,
    },
  });
}
