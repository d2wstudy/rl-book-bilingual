<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'
import { useAuth } from '../composables/useAuth'
import { useAnnotations, type AnnotationThread, type AnnotationAnchor } from '../composables/useAnnotations'
import { purgeWorkerCache } from '../composables/useGithubGql'
import { captureSelector, resolveSelector, getFullText, type ResolvedRange } from '../composables/useTextAnchor'
import NoteBubble from './NoteBubble.vue'
import AnnotationDrawer from './AnnotationDrawer.vue'

const { user, token, login } = useAuth()
const { annotations, loaded, loadAnnotations, addAnnotation, replyToAnnotation, toggleReaction } = useAnnotations()
const route = useRoute()

// Bubble state (step 1: small floating bubble)
const showBubble = ref(false)
const bubbleX = ref(0)
const bubbleY = ref(0)

// Editor state: pending note shown as inline card in sidebar
const pendingNote = ref<{ text: string } | null>(null)

// Selection info
const selectedInfo = ref<{
  paragraphId: string
  startOffset: number
  endOffset: number
  text: string
  prefix: string
  suffix: string
  segments?: AnnotationAnchor[]
} | null>(null)

// Prevent click event from immediately closing the bubble after mouseup
let justShownBubble = false

// Drawer state
const drawerOpen = ref(false)
const activeThreads = ref<AnnotationThread[]>([])

onMounted(() => {
  document.addEventListener('mouseup', onMouseUp)
  document.addEventListener('click', onDocClick)
})

onUnmounted(() => {
  document.removeEventListener('mouseup', onMouseUp)
  document.removeEventListener('click', onDocClick)
})

// Single watcher with immediate: true replaces both onMounted + watch(route.path)
watch(() => route.path, (path) => {
  loadAnnotations(path)
  drawerOpen.value = false
}, { immediate: true })

// Re-fetch with user's own token after login (or fall back to Worker after logout)
watch(token, () => {
  loadAnnotations(route.path, true)
})

watch([loaded, annotations, () => route.path], () => {
  nextTick(renderAnnotations)
})

function onMouseUp(e: MouseEvent) {
  const sel = window.getSelection()
  if (!sel || sel.isCollapsed || !sel.rangeCount) return

  const range = sel.getRangeAt(0)

  // Find lang blocks for start and end of selection
  const startLang = findParent(range.startContainer, '.bilingual-zh') || findParent(range.startContainer, '.bilingual-en')
  const endLang = findParent(range.endContainer, '.bilingual-zh') || findParent(range.endContainer, '.bilingual-en')
  if (!startLang || !endLang) return

  // Require same language type across selection
  const langClass = startLang.classList.contains('bilingual-zh') ? 'bilingual-zh' : 'bilingual-en'
  if (!endLang.classList.contains(langClass)) return

  const startPair = findParent(startLang, '.bilingual-pair')
  const endPair = findParent(endLang, '.bilingual-pair')
  if (!startPair || !endPair) return

  const selectedText = sel.toString().trim()
  if (!selectedText) return

  // Show bubble at selection position (for both logged-in and anonymous users)
  const rect = range.getBoundingClientRect()
  bubbleX.value = rect.left + rect.width / 2
  bubbleY.value = rect.top - 10
  showBubble.value = true
  justShownBubble = true

  // Only capture selection details for logged-in users
  if (!user.value) {
    selectedInfo.value = null
    return
  }

  if (startPair === endPair) {
    // Single-paragraph selection — existing logic
    const paragraphId = startPair.getAttribute('data-pair-id')
    if (!paragraphId) return

    const startOffset = getTextOffset(startLang, range.startContainer, range.startOffset)
    const endOffset = getTextOffset(startLang, range.endContainer, range.endOffset)
    const selector = captureSelector(startLang, startOffset, endOffset)

    selectedInfo.value = {
      paragraphId,
      startOffset,
      endOffset,
      text: selectedText,
      prefix: selector.prefix,
      suffix: selector.suffix,
    }
  } else {
    // Cross-paragraph selection — collect all pairs between start and end
    const allPairs = Array.from(document.querySelectorAll('.bilingual-pair[data-pair-id]'))
    const startIdx = allPairs.indexOf(startPair)
    const endIdx = allPairs.indexOf(endPair)
    if (startIdx === -1 || endIdx === -1 || startIdx > endIdx) return

    const segments: AnnotationAnchor[] = []
    for (let i = startIdx; i <= endIdx; i++) {
      const pair = allPairs[i] as HTMLElement
      const langBlock = pair.querySelector(`.${langClass}`) as HTMLElement | null
      if (!langBlock) continue

      const pid = pair.getAttribute('data-pair-id')!
      const fullText = getFullText(langBlock)
      const textLen = fullText.length
      if (textLen === 0) continue

      let sOff: number, eOff: number
      if (i === startIdx) {
        sOff = getTextOffset(langBlock, range.startContainer, range.startOffset)
        eOff = textLen
      } else if (i === endIdx) {
        sOff = 0
        eOff = getTextOffset(langBlock, range.endContainer, range.endOffset)
      } else {
        sOff = 0
        eOff = textLen
      }
      if (sOff >= eOff) continue

      const selector = captureSelector(langBlock, sOff, eOff)
      segments.push({
        paragraphId: pid,
        startOffset: sOff,
        endOffset: eOff,
        selectedText: selector.exact,
        prefix: selector.prefix,
        suffix: selector.suffix,
      })
    }

    if (!segments.length) return

    const firstSeg = segments[0]
    selectedInfo.value = {
      paragraphId: firstSeg.paragraphId,
      startOffset: firstSeg.startOffset,
      endOffset: firstSeg.endOffset,
      text: selectedText,
      prefix: firstSeg.prefix,
      suffix: firstSeg.suffix,
      segments: segments.length > 1 ? segments : undefined,
    }
  }
}

function onDocClick(e: MouseEvent) {
  // Skip the click that fires right after mouseup showed the bubble
  if (justShownBubble) {
    justShownBubble = false
    return
  }
  const target = e.target as HTMLElement
  if (!target.closest('.note-bubble') && !target.closest('.reader-anno') && !target.closest('.annotation-sidebar')) {
    showBubble.value = false
  }
}

function openEditor() {
  showBubble.value = false
  if (selectedInfo.value) {
    pendingNote.value = { text: selectedInfo.value.text }
  }
  activeThreads.value = []
  drawerOpen.value = true
  window.getSelection()?.removeAllRanges()
}

function handleLogin() {
  showBubble.value = false
  window.getSelection()?.removeAllRanges()
  login()
}

async function submitNote(note: string) {
  if (!selectedInfo.value) return

  await addAnnotation(
    route.path,
    selectedInfo.value.paragraphId,
    selectedInfo.value.startOffset,
    selectedInfo.value.endOffset,
    selectedInfo.value.text,
    note,
    selectedInfo.value.prefix,
    selectedInfo.value.suffix,
    selectedInfo.value.segments,
  )

  pendingNote.value = null
  selectedInfo.value = null
}

function cancelPendingNote() {
  pendingNote.value = null
  selectedInfo.value = null
}

function onAnnoClick(e: MouseEvent, threads: AnnotationThread[]) {
  e.stopPropagation()
  pendingNote.value = null
  activeThreads.value = threads
  drawerOpen.value = true
}

async function onDrawerReply(threadId: string, body: string) {
  await replyToAnnotation(route.path, threadId, body)
  // Update activeThreads from refreshed annotations
  syncActiveThreads()
}

async function onDrawerReact(subjectId: string, content: string) {
  const result = await toggleReaction(subjectId, content)
  await purgeWorkerCache(route.path, 'Notes', true,
    result ? { subjectId, content, delta: result.delta } : undefined)
}

function closeDrawer() {
  drawerOpen.value = false
  pendingNote.value = null
  selectedInfo.value = null
}

function syncActiveThreads() {
  if (!activeThreads.value.length) return
  const threadIds = new Set(activeThreads.value.map(t => t.id))
  // Collect all threads that share any ID with the active set
  const result: AnnotationThread[] = []
  const seen = new Set<string>()
  for (const threads of annotations.value.values()) {
    for (const t of threads) {
      if (!seen.has(t.id) && threadIds.has(t.id)) {
        seen.add(t.id)
        result.push(t)
      }
    }
  }
  activeThreads.value = result
}

/** Render reader annotations as highlighted spans */
function renderAnnotations() {
  if (typeof document === 'undefined') return

  document.querySelectorAll('.reader-anno').forEach((el) => {
    // Only collect direct text nodes — exclude .anno-inline-bubble content
    // to prevent badge numbers leaking into the text on HMR re-renders
    let raw = ''
    el.childNodes.forEach(child => {
      if (child.nodeType === Node.TEXT_NODE) raw += child.textContent || ''
    })
    el.parentNode?.replaceChild(document.createTextNode(raw), el)
  })

  const map = annotations.value
  if (!map.size) return

  document.querySelectorAll('.bilingual-pair[data-pair-id]').forEach((pair) => {
    const id = pair.getAttribute('data-pair-id')!
    const legacyId = pair.getAttribute('data-pair-id-legacy')

    // Collect annotations matching either the new hash-based ID or the legacy counter-based ID
    const allThreads = [
      ...(map.get(id) || []),
      ...(legacyId && legacyId !== id ? (map.get(legacyId) || []) : []),
    ]
    if (!allThreads.length) return

    // Deduplicate threads (cross-paragraph threads may appear under multiple IDs)
    const seen = new Set<string>()
    const uniqueThreads = allThreads.filter(t => {
      if (seen.has(t.id)) return false
      seen.add(t.id)
      return true
    })

    const zhBlock = pair.querySelector('.bilingual-zh')
    if (!zhBlock) return

    // Group threads by resolved highlight position
    const groups: { threads: AnnotationThread[]; range: ResolvedRange; showBubble: boolean }[] = []

    for (const thread of uniqueThreads) {
      // For cross-paragraph threads, find the segment matching this pair
      let anchor: { selectedText: string; prefix: string; suffix: string; startOffset: number; endOffset: number }
      let isLastSegment = true
      if (thread.segments) {
        const seg = thread.segments.find(s => s.paragraphId === id)
          || (legacyId ? thread.segments.find(s => s.paragraphId === legacyId) : undefined)
        if (!seg) continue
        anchor = seg
        isLastSegment = seg === thread.segments[thread.segments.length - 1]
      } else {
        anchor = thread.anchor
      }

      const selector = {
        exact: anchor.selectedText,
        prefix: anchor.prefix || '',
        suffix: anchor.suffix || '',
      }
      const range = resolveSelector(zhBlock, selector, anchor.startOffset, anchor.endOffset)
      if (!range) continue

      // Try to merge with an existing group at the same position
      const existing = groups.find(g =>
        g.range.startOffset === range.startOffset && g.range.endOffset === range.endOffset
      )
      if (existing) {
        existing.threads.push(thread)
        if (isLastSegment) existing.showBubble = true
      } else {
        groups.push({ threads: [thread], range, showBubble: isLastSegment })
      }
    }

    // === Atomic segment decomposition: split all ranges into non-overlapping atoms ===
    // This prevents DOM corruption when annotations partially overlap.
    const boundarySet = new Set<number>()
    for (const group of groups) {
      boundarySet.add(group.range.startOffset)
      boundarySet.add(group.range.endOffset)
    }
    const boundaries = [...boundarySet].sort((a, b) => a - b)

    const atoms: {
      startOffset: number
      endOffset: number
      threads: AnnotationThread[]
      depth: number
      bubbleCount: number
    }[] = []

    for (let i = 0; i < boundaries.length - 1; i++) {
      const start = boundaries[i]
      const end = boundaries[i + 1]
      const covering = groups.filter(g =>
        g.range.startOffset <= start && g.range.endOffset >= end
      )
      if (!covering.length) continue

      // Deduplicate threads across covering groups
      const seenT = new Set<string>()
      const allThreads: AnnotationThread[] = []
      for (const g of covering) {
        for (const t of g.threads) {
          if (!seenT.has(t.id)) { seenT.add(t.id); allThreads.push(t) }
        }
      }

      // Bubble: show if any covering group's range ends at this segment's end
      const endingGroups = covering.filter(g => g.range.endOffset === end && g.showBubble)
      const bubbleSeen = new Set<string>()
      const bubbleThreads: AnnotationThread[] = []
      for (const g of endingGroups) {
        for (const t of g.threads) {
          if (!bubbleSeen.has(t.id)) { bubbleSeen.add(t.id); bubbleThreads.push(t) }
        }
      }
      const bubbleCount = bubbleThreads.reduce((sum, t) => sum + 1 + t.replies.length, 0)

      atoms.push({ startOffset: start, endOffset: end, threads: allThreads, depth: covering.length, bubbleCount })
    }

    // Render in reverse so DOM mutations don't shift later offsets
    atoms.sort((a, b) => b.startOffset - a.startOffset)
    for (const atom of atoms) {
      highlightRange(zhBlock, atom.threads, { startOffset: atom.startOffset, endOffset: atom.endOffset }, atom.bubbleCount, atom.depth)
    }
  })
}

function highlightRange(container: Element, threads: AnnotationThread[], range: ResolvedRange, bubbleCount: number, depth: number) {
  // Collect all text nodes that overlap with the range
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  let offset = 0
  const hits: { node: Text; relStart: number; relEnd: number }[] = []

  let node: Text | null
  while ((node = walker.nextNode() as Text | null)) {
    const nodeEnd = offset + node.length
    if (nodeEnd > range.startOffset && offset < range.endOffset) {
      hits.push({
        node,
        relStart: Math.max(0, range.startOffset - offset),
        relEnd: Math.min(node.length, range.endOffset - offset),
      })
    }
    if (nodeEnd >= range.endOffset) break
    offset = nodeEnd
  }

  if (!hits.length) return

  // Process in reverse so earlier DOM mutations don't shift later nodes
  for (let i = hits.length - 1; i >= 0; i--) {
    const { node: textNode, relStart, relEnd } = hits[i]
    const before = textNode.textContent!.slice(0, relStart)
    const middle = textNode.textContent!.slice(relStart, relEnd)
    const after = textNode.textContent!.slice(relEnd)

    const span = document.createElement('span')
    span.className = 'reader-anno'
    if (depth > 1) span.classList.add('reader-anno-overlap')
    span.textContent = middle
    span.title = `${threads.length} 条笔记`
    span.setAttribute('data-anno-id', threads[0].id)

    // Inline bubble only on the very last span of this atom
    if (bubbleCount > 0 && i === hits.length - 1) {
      const bubble = document.createElement('span')
      bubble.className = 'anno-inline-bubble'
      bubble.innerHTML = `<svg viewBox="0 0 256 256" fill="none" stroke="currentColor" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"><path d="M132,24A100.11,100.11,0,0,0,32,124v84a16,16,0,0,0,16,16h84a100,100,0,0,0,0-200Z"/></svg><span class="anno-count">${bubbleCount}</span>`
      span.appendChild(bubble)
    }

    span.addEventListener('click', (e) => onAnnoClick(e as MouseEvent, threads))

    const parent = textNode.parentNode!
    if (after) parent.insertBefore(document.createTextNode(after), textNode.nextSibling)
    parent.insertBefore(span, textNode.nextSibling)
    if (before) parent.insertBefore(document.createTextNode(before), textNode.nextSibling)
    parent.removeChild(textNode)
  }
}

// ---- Utilities ----

function getTextOffset(root: Node, node: Node, offset: number): number {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT)
  let total = 0
  let current: Node | null
  while ((current = walker.nextNode())) {
    if (current === node) return total + offset
    total += (current.textContent || '').length
  }
  return total
}

function findParent(node: Node, selector: string): HTMLElement | null {
  let el: HTMLElement | null = node instanceof HTMLElement ? node : node.parentElement
  while (el) {
    if (el.matches(selector)) return el
    el = el.parentElement
  }
  return null
}
</script>

<template>
  <!-- Step 1: Selection bubble (shown for both logged-in and anonymous users) -->
  <NoteBubble
    :visible="showBubble"
    :x="bubbleX"
    :y="bubbleY"
    :logged-in="!!user"
    @open-editor="openEditor"
    @login="handleLogin"
  />

  <!-- Annotation sidebar -->
  <AnnotationDrawer
    :open="drawerOpen"
    :threads="activeThreads"
    :pending-note="pendingNote"
    @close="closeDrawer"
    @reply="onDrawerReply"
    @react="onDrawerReact"
    @submit-note="submitNote"
    @cancel-note="cancelPendingNote"
  />
</template>
