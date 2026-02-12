<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'
import { useAuth } from '../composables/useAuth'
import { useAnnotations, type Annotation } from '../composables/useAnnotations'
import { useMarkdown } from '../composables/useMarkdown'
import NoteBubble from './NoteBubble.vue'
import NoteEditor from './NoteEditor.vue'

const { user, token } = useAuth()
const { annotations, loaded, loadAnnotations, addAnnotation } = useAnnotations()
const { renderMarkdown } = useMarkdown()
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
} | null>(null)

// Prevent click event from immediately closing the bubble after mouseup
let justShownBubble = false

// View annotation state
const viewAnno = ref<Annotation | null>(null)
const viewX = ref(0)
const viewY = ref(0)

const renderedViewNote = computed(() =>
  viewAnno.value ? renderMarkdown(viewAnno.value.note) : ''
)

onMounted(() => {
  loadAnnotations(route.path)
  document.addEventListener('mouseup', onMouseUp)
  document.addEventListener('click', onDocClick)
})

onUnmounted(() => {
  document.removeEventListener('mouseup', onMouseUp)
  document.removeEventListener('click', onDocClick)
})

watch(() => route.path, (path) => {
  loadAnnotations(path)
})

watch(token, () => loadAnnotations(route.path))

watch([loaded, annotations, () => route.path], () => {
  nextTick(renderAnnotations)
}, { deep: true })

function onMouseUp(e: MouseEvent) {
  if (!user.value) return

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

  const startOffset = getTextOffset(langBlock, range.startContainer, range.startOffset)
  const endOffset = getTextOffset(langBlock, range.endContainer, range.endOffset)

  selectedInfo.value = {
    paragraphId,
    startOffset,
    endOffset,
    text: selectedText,
  }

  const rect = range.getBoundingClientRect()
  bubbleX.value = rect.left + rect.width / 2
  bubbleY.value = rect.top - 10
  showBubble.value = true
  justShownBubble = true
}

function onDocClick(e: MouseEvent) {
  // Skip the click that fires right after mouseup showed the bubble
  if (justShownBubble) {
    justShownBubble = false
    return
  }
  const target = e.target as HTMLElement
  if (!target.closest('.note-bubble') && !target.closest('.note-editor-overlay') && !target.closest('.reader-anno')) {
    showBubble.value = false
    viewAnno.value = null
  }
}

function openEditor() {
  showBubble.value = false
  showEditor.value = true
  window.getSelection()?.removeAllRanges()
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
  )

  showEditor.value = false
  selectedInfo.value = null
}

function cancelEditor() {
  showEditor.value = false
  selectedInfo.value = null
}

function onAnnoClick(e: MouseEvent, anno: Annotation) {
  e.stopPropagation()
  const rect = (e.target as HTMLElement).getBoundingClientRect()
  viewX.value = rect.left + rect.width / 2
  viewY.value = rect.top - 10
  viewAnno.value = anno
}

/** Render reader annotations as highlighted spans in the zh blocks */
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
    const annos = map.get(id)
    if (!annos || !annos.length) return

    const zhBlock = pair.querySelector('.bilingual-zh')
    if (!zhBlock) return

    const sorted = [...annos].sort((a, b) => b.startOffset - a.startOffset)

    for (const anno of sorted) {
      highlightRange(zhBlock, anno)
    }
  })
}

function highlightRange(container: Element, anno: Annotation) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  let offset = 0
  let node: Text | null

  while ((node = walker.nextNode() as Text | null)) {
    const nodeEnd = offset + node.length
    if (offset <= anno.startOffset && nodeEnd >= anno.endOffset) {
      const relStart = anno.startOffset - offset
      const relEnd = anno.endOffset - offset

      const before = node.textContent!.slice(0, relStart)
      const middle = node.textContent!.slice(relStart, relEnd)
      const after = node.textContent!.slice(relEnd)

      const span = document.createElement('span')
      span.className = 'reader-anno'
      span.textContent = middle
      span.title = `${anno.author}: ${anno.note}`
      span.setAttribute('data-anno-id', anno.id)
      span.addEventListener('click', (e) => onAnnoClick(e as MouseEvent, anno))

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
  <!-- Step 1: Selection bubble -->
  <NoteBubble
    :visible="showBubble && !!selectedInfo"
    :x="bubbleX"
    :y="bubbleY"
    @open-editor="openEditor"
  />

  <!-- Step 2: Markdown editor modal -->
  <NoteEditor
    v-if="showEditor && selectedInfo"
    :selected-text="selectedInfo.text"
    @submit="submitNote"
    @cancel="cancelEditor"
  />

  <!-- View annotation popup -->
  <Teleport to="body">
    <div
      v-if="viewAnno"
      class="note-popup-view"
      :style="{ left: viewX + 'px', top: viewY + 'px' }"
    >
      <div class="note-view-header">
        <img :src="viewAnno.authorAvatar" class="note-avatar" />
        <span class="note-author">{{ viewAnno.author }}</span>
      </div>
      <div class="note-view-body" v-html="renderedViewNote" />
    </div>
  </Teleport>
</template>

<style scoped>
.note-popup-view {
  position: fixed;
  transform: translateX(-50%) translateY(-100%);
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  padding: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  z-index: 200;
  width: 300px;
}

.note-view-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.note-avatar {
  width: 20px;
  height: 20px;
  border-radius: 50%;
}

.note-author {
  font-size: 12px;
  color: var(--vp-c-text-2);
  font-weight: 500;
}

.note-view-body {
  font-size: 13px;
  line-height: 1.6;
  color: var(--vp-c-text-1);
}

.note-view-body :deep(p) { margin: 4px 0; }
.note-view-body :deep(code) {
  background: var(--vp-c-bg-soft);
  padding: 2px 4px;
  border-radius: 3px;
  font-size: 12px;
}
.note-view-body :deep(pre) {
  background: var(--vp-c-bg-soft);
  padding: 8px;
  border-radius: 6px;
  overflow-x: auto;
  margin: 6px 0;
}
.note-view-body :deep(pre code) {
  background: none;
  padding: 0;
}
.note-view-body :deep(blockquote) {
  border-left: 3px solid var(--vp-c-divider);
  padding-left: 8px;
  color: var(--vp-c-text-2);
  margin: 4px 0;
}
.note-view-body :deep(a) { color: var(--vp-c-brand-1); }
.note-view-body :deep(ul),
.note-view-body :deep(ol) { padding-left: 18px; }
</style>
