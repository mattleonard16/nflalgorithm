import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Normalise a graded result to "WIN" | "LOSS" | "PUSH". The backend stores
 * lowercase ("win"/"loss"/"push"); older rows and other feeds may use
 * uppercase or single letters. Null/empty means ungraded.
 */
export function normaliseResult(r: string | null | undefined): string | null {
  if (!r) return null
  const u = r.toUpperCase()
  if (u === "WIN" || u === "W") return "WIN"
  if (u === "LOSS" || u === "L") return "LOSS"
  if (u === "PUSH" || u === "P") return "PUSH"
  return u
}

/** Short side marker for a line: "U" for under, "O" for over/unknown legacy rows. */
export function sideLabel(side: string | null | undefined): string {
  return side?.toLowerCase() === "under" ? "U" : "O"
}
