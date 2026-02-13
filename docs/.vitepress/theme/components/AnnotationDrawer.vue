<script setup lang="ts">
import { computed, ref, nextTick } from 'vue'
import { useAuth } from '../composables/useAuth'
import type { AnnotationThread } from '../composables/useAnnotations'
import CommentItem from './CommentItem.vue'
import MarkdownEditor from './MarkdownEditor.vue'

const props = defineProps<{
  open: boolean
  threads: AnnotationThread[]
  pendingNote?: { text: string } | null
}>()

const emit = defineEmits<{
  close: []
  reply: [threadId: string, body: string]
  react: [subjectId: string, content: string]
  'submit-note': [note: string]
  'cancel-note': []
}>()

const { user, login } = useAuth()
const editorRef = ref<InstanceType<typeof MarkdownEditor> | null>(null)

function threadQuote(thread: AnnotationThread): string {
  if (thread.segments && thread.segments.length > 1) {
    return thread.segments.map(s => s.selectedText).join(' … ')
  }
  return thread.anchor.selectedText
}

function onSubmitNote(text: string) {
  emit('submit-note', text)
}

function onCancelNote() {
  emit('cancel-note')
}

function onReply(threadId: string, body: string) {
  emit('reply', threadId, body)
}

function onReact(subjectId: string, content: string) {
  emit('react', subjectId, content)
}

// Auto-focus editor when pendingNote appears
const onEditorMounted = () => {
  nextTick(() => editorRef.value?.focus())
}
</script>

<template>
  <Teleport to="body">
    <Transition name="sidebar-slide">
      <div v-if="open" class="annotation-sidebar-wrapper">
        <!-- Transparent backdrop for click-to-close -->
        <div class="sidebar-backdrop" @click="emit('close')" />

        <div class="annotation-sidebar" @click.stop>
          <!-- Header -->
          <div class="sidebar-header">
            <span class="sidebar-title">笔记</span>
            <button class="sidebar-close" @click="emit('close')" title="关闭">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>

          <div class="sidebar-body">
            <!-- Pending new note: inline editor card -->
            <div v-if="pendingNote && user" class="sidebar-card sidebar-card-new">
              <div class="card-quote">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="card-quote-icon">
                  <path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V21z"/>
                  <path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3c0 .001 0 1.003 1 1.003z"/>
                </svg>
                <span class="card-quote-text">{{ pendingNote.text }}</span>
              </div>
              <MarkdownEditor
                ref="editorRef"
                placeholder="写下你的笔记... 支持 Markdown 语法"
                submit-label="提交"
                @submit="onSubmitNote"
                @cancel="onCancelNote"
                @vue:mounted="onEditorMounted"
              />
            </div>

            <!-- Pending new note: login prompt -->
            <div v-else-if="pendingNote && !user" class="sidebar-card sidebar-card-login">
              <div class="card-quote">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="card-quote-icon">
                  <path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V21z"/>
                  <path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3c0 .001 0 1.003 1 1.003z"/>
                </svg>
                <span class="card-quote-text">{{ pendingNote.text }}</span>
              </div>
              <button class="card-login-btn" @click="login()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>
                </svg>
                登录 GitHub 后添加笔记
              </button>
            </div>

            <!-- Existing thread cards -->
            <div v-for="thread in threads" :key="thread.id" class="sidebar-card">
              <div class="card-quote">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="card-quote-icon">
                  <path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V21z"/>
                  <path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3c0 .001 0 1.003 1 1.003z"/>
                </svg>
                <span class="card-quote-text">{{ threadQuote(thread) }}</span>
              </div>
              <CommentItem
                :id="thread.id"
                :body="thread.note"
                :author="thread.author"
                :author-avatar="thread.authorAvatar"
                :created-at="thread.createdAt"
                :reactions="thread.reactions"
                :replies="thread.replies"
                compact
                @reply="onReply"
                @react="onReact"
              />
            </div>

            <div v-if="!threads.length && !pendingNote" class="sidebar-empty">
              暂无笔记
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.annotation-sidebar-wrapper {
  position: fixed;
  inset: 0;
  z-index: 250;
  pointer-events: none;
}

.sidebar-backdrop {
  position: absolute;
  inset: 0;
  pointer-events: auto;
}

.annotation-sidebar {
  position: absolute;
  top: 0;
  right: 0;
  width: 400px;
  max-width: 90vw;
  height: 100vh;
  height: 100dvh;
  background: var(--vp-c-bg);
  border-left: 1px solid var(--vp-c-divider);
  display: flex;
  flex-direction: column;
  box-shadow: -4px 0 24px rgba(0, 0, 0, 0.08);
  pointer-events: auto;
}

.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--vp-c-divider);
  flex-shrink: 0;
}

.sidebar-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.sidebar-close {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 6px;
  background: none;
  color: var(--vp-c-text-3);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
}
.sidebar-close:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.sidebar-body {
  flex: 1;
  overflow-y: auto;
  padding: 8px 12px;
}

/* --- Card --- */
.sidebar-card {
  padding: 10px 12px;
  border-bottom: 1px solid var(--vp-c-divider);
}
.sidebar-card:last-child {
  border-bottom: none;
}

.sidebar-card-new {
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  border-bottom: none;
  margin-bottom: 8px;
}

.sidebar-card-login {
  background: var(--vp-c-bg-soft);
  border-radius: 8px;
  border-bottom: none;
  margin-bottom: 8px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* --- Card quote --- */
.card-quote {
  display: flex;
  gap: 6px;
  align-items: flex-start;
  margin-bottom: 8px;
  padding: 6px 8px;
  border-left: 2px solid var(--vp-c-warning-1);
  border-radius: 0 4px 4px 0;
  background: color-mix(in srgb, var(--vp-c-warning-1) 5%, transparent);
}

.card-quote-icon {
  flex-shrink: 0;
  color: var(--vp-c-text-3);
  margin-top: 1px;
}

.card-quote-text {
  font-size: 12px;
  line-height: 1.5;
  color: var(--vp-c-text-3);
  font-style: italic;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* --- Login button --- */
.card-login-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 10px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}
.card-login-btn:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}

/* --- Empty state --- */
.sidebar-empty {
  text-align: center;
  color: var(--vp-c-text-3);
  font-size: 14px;
  padding: 32px 0;
}

/* --- Slide animation --- */
.sidebar-slide-enter-active,
.sidebar-slide-leave-active {
  transition: opacity 0.2s ease;
}
.sidebar-slide-enter-active .annotation-sidebar,
.sidebar-slide-leave-active .annotation-sidebar {
  transition: transform 0.2s ease;
}
.sidebar-slide-enter-from,
.sidebar-slide-leave-to {
  opacity: 0;
}
.sidebar-slide-enter-from .annotation-sidebar,
.sidebar-slide-leave-to .annotation-sidebar {
  transform: translateX(100%);
}

@media (max-width: 768px) {
  .annotation-sidebar {
    width: 100vw;
    max-width: 100vw;
  }
}
</style>
