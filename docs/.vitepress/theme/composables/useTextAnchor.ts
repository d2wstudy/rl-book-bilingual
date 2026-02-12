/**
 * TextQuoteSelector-based text anchoring.
 * Inspired by W3C Web Annotation Data Model (https://www.w3.org/TR/annotation-model/).
 *
 * Stores selected text along with surrounding context (prefix/suffix),
 * enabling robust re-anchoring even after document edits.
 */

/** Context length (chars) captured before and after the selection */
const CONTEXT_LEN = 32

export interface TextQuoteSelector {
  /** The exact selected text */
  exact: string
  /** Up to CONTEXT_LEN chars immediately before the selection */
  prefix: string
  /** Up to CONTEXT_LEN chars immediately after the selection */
  suffix: string
}

export interface ResolvedRange {
  startOffset: number
  endOffset: number
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/** Extract full concatenated text content from a DOM subtree */
export function getFullText(container: Node): string {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  let text = ''
  while (walker.nextNode()) {
    text += walker.currentNode.textContent || ''
  }
  return text
}

/** Capture a TextQuoteSelector for the given character offsets within a container */
export function captureSelector(
  container: Node,
  startOffset: number,
  endOffset: number,
): TextQuoteSelector {
  const full = getFullText(container)
  return {
    exact: full.slice(startOffset, endOffset),
    prefix: full.slice(Math.max(0, startOffset - CONTEXT_LEN), startOffset),
    suffix: full.slice(endOffset, endOffset + CONTEXT_LEN),
  }
}

// ---------------------------------------------------------------------------
// Resolve — multi-strategy matching
// ---------------------------------------------------------------------------

/**
 * Resolve a TextQuoteSelector back to character offsets in the (possibly changed)
 * container text.  Tries three strategies in order:
 *
 * 1. **Hint offsets** — if the stored offsets still point at the exact text, use them.
 * 2. **Exact search** — find all occurrences of `exact`; disambiguate with prefix/suffix.
 * 3. **Fuzzy search** — sliding-window best-match using prefix + exact + suffix.
 *
 * Returns `null` when no acceptable match is found.
 */
export function resolveSelector(
  container: Node,
  selector: TextQuoteSelector,
  hintStart?: number,
  hintEnd?: number,
): ResolvedRange | null {
  const full = getFullText(container)
  if (!selector.exact) return null

  // Strategy 1: hint offsets still valid
  if (hintStart !== undefined && hintEnd !== undefined) {
    if (full.slice(hintStart, hintEnd) === selector.exact) {
      return { startOffset: hintStart, endOffset: hintEnd }
    }
  }

  // Strategy 2: exact substring search
  const candidates = allIndexesOf(full, selector.exact)

  if (candidates.length === 1) {
    return { startOffset: candidates[0], endOffset: candidates[0] + selector.exact.length }
  }

  if (candidates.length > 1) {
    // Disambiguate using prefix/suffix context scoring
    const best = pickBestCandidate(full, candidates, selector)
    return { startOffset: best, endOffset: best + selector.exact.length }
  }

  // Strategy 3: fuzzy match — look for the best approximate match
  return fuzzyMatch(full, selector)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function allIndexesOf(haystack: string, needle: string): number[] {
  const results: number[] = []
  let pos = 0
  while (pos <= haystack.length - needle.length) {
    const idx = haystack.indexOf(needle, pos)
    if (idx === -1) break
    results.push(idx)
    pos = idx + 1
  }
  return results
}

/** Score each candidate position by how well its surrounding text matches prefix/suffix */
function pickBestCandidate(
  full: string,
  candidates: number[],
  selector: TextQuoteSelector,
): number {
  let bestIdx = candidates[0]
  let bestScore = -1

  for (const idx of candidates) {
    let score = 0
    if (selector.prefix) {
      const before = full.slice(Math.max(0, idx - selector.prefix.length), idx)
      score += commonSuffixLen(before, selector.prefix)
    }
    if (selector.suffix) {
      const after = full.slice(
        idx + selector.exact.length,
        idx + selector.exact.length + selector.suffix.length,
      )
      score += commonPrefixLen(after, selector.suffix)
    }
    if (score > bestScore) {
      bestScore = score
      bestIdx = idx
    }
  }
  return bestIdx
}

/**
 * Fuzzy match: build a "needle" from prefix+exact+suffix, slide it over the full
 * text, and pick the window with the highest character-level similarity.
 * Only returns a result if similarity exceeds a threshold.
 */
function fuzzyMatch(full: string, selector: TextQuoteSelector): ResolvedRange | null {
  const needle = selector.prefix + selector.exact + selector.suffix
  if (!needle) return null

  const windowLen = needle.length
  // Allow the window to be slightly shorter/longer to account for minor edits
  const minWin = Math.max(1, Math.floor(windowLen * 0.8))
  const maxWin = Math.ceil(windowLen * 1.2)

  let bestScore = 0
  let bestStart = -1
  let bestEnd = -1

  // Slide with the exact window length first (most common case)
  for (let i = 0; i <= full.length - minWin; i++) {
    for (const wLen of [windowLen, minWin, maxWin]) {
      if (i + wLen > full.length) continue
      const window = full.slice(i, i + wLen)
      const score = similarity(needle, window)
      if (score > bestScore) {
        bestScore = score
        // The "exact" portion starts after the prefix within the window
        const prefixRatio = selector.prefix.length / needle.length
        const suffixRatio = selector.suffix.length / needle.length
        const exactStart = i + Math.round(prefixRatio * wLen)
        const exactEnd = i + wLen - Math.round(suffixRatio * wLen)
        bestStart = exactStart
        bestEnd = exactEnd
      }
    }
  }

  // Require at least 60% similarity to accept a fuzzy match
  if (bestScore >= 0.6 && bestStart >= 0 && bestEnd > bestStart) {
    return { startOffset: bestStart, endOffset: bestEnd }
  }
  return null
}

/** Bigram-based similarity (Dice coefficient) — fast and effective for short strings */
function similarity(a: string, b: string): number {
  if (a === b) return 1
  if (a.length < 2 || b.length < 2) return 0

  const bigramsA = new Map<string, number>()
  for (let i = 0; i < a.length - 1; i++) {
    const bg = a.slice(i, i + 2)
    bigramsA.set(bg, (bigramsA.get(bg) || 0) + 1)
  }

  let intersection = 0
  for (let i = 0; i < b.length - 1; i++) {
    const bg = b.slice(i, i + 2)
    const count = bigramsA.get(bg)
    if (count && count > 0) {
      bigramsA.set(bg, count - 1)
      intersection++
    }
  }

  return (2 * intersection) / (a.length - 1 + b.length - 1)
}

function commonPrefixLen(a: string, b: string): number {
  const len = Math.min(a.length, b.length)
  let i = 0
  while (i < len && a[i] === b[i]) i++
  return i
}

function commonSuffixLen(a: string, b: string): number {
  const len = Math.min(a.length, b.length)
  let i = 0
  while (i < len && a[a.length - 1 - i] === b[b.length - 1 - i]) i++
  return i
}
