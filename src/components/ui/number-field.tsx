"use client";

import { useRef, useState } from "react";

import { Input } from "@/components/ui/input";

/**
 * A number that is committed when it makes sense: the field keeps what is being typed until
 * it is left or Enter is pressed, so it can be cleared and retyped, and a value outside
 * [min, max] or not a number snaps back to the last good one. Escape abandons the typing.
 * A null value shows as `placeholder` until one is typed.
 */
export function NumberField({
  value,
  onChange,
  min,
  max,
  step,
  integer = false,
  disabled,
  placeholder,
  ...rest
}: {
  value: number | null;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  integer?: boolean;
  disabled?: boolean;
  placeholder?: string;
  "aria-label"?: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  // Escape leaves the field, and leaving commits: this says not to. A ref, because the blur
  // arrives before a state change would have rendered.
  const abandoning = useRef(false);

  const commit = () => {
    const typed = abandoning.current ? null : draft;
    abandoning.current = false;
    setDraft(null);
    if (typed === null) return;
    const parsed = Number(typed);
    if (!typed.trim() || !Number.isFinite(parsed)) return;
    if ((min !== undefined && parsed < min) || (max !== undefined && parsed > max)) return;
    if (integer && !Number.isInteger(parsed)) return;
    if (parsed !== value) onChange(parsed);
  };

  return (
    <Input
      type="number"
      inputMode="decimal"
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      value={draft ?? (value === null ? "" : String(value))}
      placeholder={placeholder}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") event.currentTarget.blur();
        if (event.key === "Escape") {
          abandoning.current = true;
          event.currentTarget.blur();
        }
      }}
      className="h-6 w-20 px-1.5 text-right font-mono text-xs"
      {...rest}
    />
  );
}
