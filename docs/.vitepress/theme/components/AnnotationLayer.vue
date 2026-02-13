<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'
import { useAuth } from '../composables/useAuth'
import { useAnnotations, type AnnotationThread } from '../composables/useAnnotations'
import { purgeWorkerCache } from '../composables/useGithubGql'
import { captureSelector, resolveSelector, type ResolvedRange } from '../composables/useTextAnchor'
import NoteBubble from './NoteBubble.vue'
import NoteEditor from './NoteEditor.vue'
import AnnotationDrawer from './AnnotationDrawer.vue'

const { user, token, login } = useAuth()
const { annotations, loaded, loadAnnotations, addAnnotation, replyToAnnotation, toggleReaction } = useAnnotations()
const route = useRoute()

// Bubble state (step 1: small floating bubble)
const showBubble = ref(false)
const bubbleX = ref(0)
const bubbleY = ref(0)

// Editor state (step 2: markdown editor modal)
const showEditor = ref(false)

// Selection info
const selectedInfo = ref<{
  paragraphId: string
  startOffset: number
  endOffset: number
  text: string
  prefix: string
  suffix: string
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
  const container = range.commonAncestorContainer

  // Support both Chinese and English blocks
  const langBlock = findParent(container, '.bilingual-zh') || findParent(container, '.bilingual-en')
  if (!langBlock) return

  const pair = findParent(langBlock, '.bilingual-pair')
  if (!pair) return

  const paragraphId = pair.getAttribute('data-pair-id')
  if (!paragraphId) return

  const selectedText = sel.toString().trim()
  if (!selectedText) return

  // Show bubble at selection position (for both logged-in and anonymous users)
  const rect = range.getBoundingClientRect()
  bubbleX.value = rect.left + rect.width / 2
  bubbleY.value = rect.top - 10
  showBubble.value = true
  justShownBubble = true

  // Only capture selection details for logged-in users
  if (user.value) {
    const startOffset = getTextOffset(langBlock, range.startContainer, range.startOffset)
    const endOffset = getTextOffset(langBlock, range.endContainer, range.endOffset)
    const selector = captureSelector(langBlock, startOffset, endOffset)

    selectedInfo.value = {
      paragraphId,
      startOffset,
      endOffset,
      text: selectedText,
      prefix: selector.prefix,
      suffix: selector.suffix,
    }
  } else {
    selectedInfo.value = null
  }
}

function onDocClick(e: MouseEvent) {
  // Skip the click that fires right after mouseup showed the bubble
  if (justShownBubble) {
    justShownBubble = false
    return
  }
  const target = e.target as HTMLElement
  if (!target.closest('.note-bubble') && !target.closest('.note-editor-overlay') && !target.closest('.reader-anno') && !target.closest('.annotation-drawer')) {
    showBubble.value = false
  }
}

function openEditor() {
  showBubble.value = false
  showEditor.value = true
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
  )

  showEditor.value = false
  selectedInfo.value = null
}

function cancelEditor() {
  showEditor.value = false
  selectedInfo.value = null
}

function onAnnoClick(e: MouseEvent, threads: AnnotationThread[]) {
  e.stopPropagation()
  activeThreads.value = threads
  drawerOpen.value = true
}

function closeDrawer() {
  drawerOpen.value = false
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

async function onDrawerAddNote(text: string) {
  // Add a new annotation on the same highlight as the first active thread
  if (!activeThreads.value.length) return
  const anchor = activeThreads.value[0].anchor
  await addAnnotation(
    route.path,
    anchor.paragraphId,
    anchor.startOffset,
    anchor.endOffset,
    anchor.selectedText,
    text,
    anchor.prefix,
    anchor.suffix,
  )
  syncActiveThreads()
}

function syncActiveThreads() {
  if (!activeThreads.value.length) return
  const anchor = activeThreads.value[0].anchor
  const map = annotations.value
  const threads = [
    ...(map.get(anchor.paragraphId) || []),
  ]
  // Filter to threads matching the same highlight position
  activeThreads.value = threads.filter(t =>
    t.anchor.selectedText === anchor.selectedText &&
    t.anchor.startOffset === anchor.startOffset &&
    t.anchor.endOffset === anchor.endOffset
  )
}

/** Render reader annotations as highlighted spans */
function renderAnnotations() {
  if (typeof document === 'undefined') return

  document.querySelectorAll('.reader-anno').forEach((el) => {
    const text = document.createTextNode(el.textContent || '')
    el.parentNode?.replaceChild(text, el)
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

    const zhBlock = pair.querySelector('.bilingual-zh')
    if (!zhBlock) return

    // Group threads by resolved highlight position
    const groups: { threads: AnnotationThread[]; range: ResolvedRange }[] = []

    for (const thread of allThreads) {
      const selector = {
        exact: thread.anchor.selectedText,
        prefix: thread.anchor.prefix || '',
        suffix: thread.anchor.suffix || '',
      }
      const range = resolveSelector(zhBlock, selector, thread.anchor.startOffset, thread.anchor.endOffset)
      if (!range) continue

      // Try to merge with an existing group at the same position
      const existing = groups.find(g =>
        g.range.startOffset === range.startOffset && g.range.endOffset === range.endOffset
      )
      if (existing) {
        existing.threads.push(thread)
      } else {
        groups.push({ threads: [thread], range })
      }
    }

    // Sort by descending startOffset so DOM mutations don't shift later offsets
    groups.sort((a, b) => b.range.startOffset - a.range.startOffset)

    for (const group of groups) {
      highlightRange(zhBlock, group.threads, group.range)
    }
  })
}

function highlightRange(container: Element, threads: AnnotationThread[], range: ResolvedRange) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  let offset = 0
  let node: Text | null

  while ((node = walker.nextNode() as Text | null)) {
    const nodeEnd = offset + node.length
    if (offset <= range.startOffset && nodeEnd >= range.endOffset) {
      const relStart = range.startOffset - offset
      const relEnd = range.endOffset - offset

      const before = node.textContent!.slice(0, relStart)
      const middle = node.textContent!.slice(relStart, relEnd)
      const after = node.textContent!.slice(relEnd)

      const span = document.createElement('span')
      span.className = 'reader-anno'
      span.textContent = middle
      span.title = `${threads.length} 条笔记`
      span.setAttribute('data-anno-id', threads[0].id)

      // Inline bubble (Phosphor Chat Teardrop)
      const totalNotes = threads.reduce((sum, t) => sum + 1 + t.replies.length, 0)
      if (totalNotes > 0) {
        const bubble = document.createElement('span')
        bubble.className = 'anno-inline-bubble'
        bubble.innerHTML = `<svg viewBox="0 0 256 256" fill="none" stroke="currentColor" stroke-width="16" stroke-linecap="round" stroke-linejoin="round"><path d="M132,24A100.11,100.11,0,0,0,32,124v84a16,16,0,0,0,16,16h84a100,100,0,0,0,0-200Z"/></svg><span class="anno-count">${totalNotes}</span>`
        span.appendChild(bubble)
      }

      span.addEventListener('click', (e) => onAnnoClick(e as MouseEvent, threads))

      const parent = node.parentNode!
      if (after) parent.insertBefore(document.createTextNode(after), node.nextSibling)
      parent.insertBefore(span, node.nextSibling)
      if (before) parent.insertBefore(document.createTextNode(before), node.nextSibling)
      parent.removeChild(node)
      break
    }
    offset = nodeEnd
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

  <!-- Step 2: Markdown editor modal -->
  <NoteEditor
    v-if="showEditor && selectedInfo"
    :selected-text="selectedInfo.text"
    @submit="submitNote"
    @cancel="cancelEditor"
  />

  <!-- Step 3: Annotation drawer -->
  <AnnotationDrawer
    :open="drawerOpen"
    :threads="activeThreads"
    @close="closeDrawer"
    @reply="onDrawerReply"
    @react="onDrawerReact"
    @add-note="onDrawerAddNote"
  />
</template>
