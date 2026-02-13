<script setup lang="ts">
import { computed } from 'vue'
import { useAuth } from '../composables/useAuth'
import type { AnnotationThread } from '../composables/useAnnotations'
import CommentItem from './CommentItem.vue'
import MarkdownEditor from './MarkdownEditor.vue'

const props = defineProps<{
  open: boolean
  threads: AnnotationThread[]
}>()

const emit = defineEmits<{
  close: []
  reply: [threadId: string, body: string]
  react: [subjectId: string, content: string]
  'add-note': [text: string]
}>()

const { user, login } = useAuth()

const quoteText = computed(() => {
  if (!props.threads.length) return ''
  const thread = props.threads[0]
  if (thread.segments && thread.segments.length > 1) {
    return thread.segments.map(s => s.selectedText).join(' … ')
  }
  return thread.anchor.selectedText
})

function onReply(threadId: string, body: string) {
  emit('reply', threadId, body)
}

function onReact(subjectId: string, content: string) {
  emit('react', subjectId, content)
}

function onAddNote(text: string) {
  emit('add-note', text)
}
</script>

<template>
  <Teleport to="body">
    <Transition name="drawer-slide">
      <div v-if="open" class="annotation-drawer-overlay" @click.self="emit('close')">
        <div class="annotation-drawer" @click.stop>
          <!-- Header -->
          <div class="drawer-header">
            <div class="drawer-quote">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="quote-icon">
                <path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V21z"/>
                <path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3c0 .001 0 1.003 1 1.003z"/>
              </svg>
              <span class="quote-text">{{ quoteText }}</span>
            </div>
            <button class="drawer-close" @click="emit('close')" title="关闭">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>

          <!-- Thread list -->
          <div class="drawer-body">
            <div v-for="thread in threads" :key="thread.id" class="drawer-thread">
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

            <div v-if="!threads.length" class="drawer-empty">
              暂无笔记
            </div>
          </div>

          <!-- Footer: add new note -->
          <div v-if="user" class="drawer-footer">
            <MarkdownEditor
              placeholder="添加笔记..."
              submit-label="发送"
              @submit="onAddNote"
              @cancel="emit('close')"
            />
          </div>
          <div v-else class="drawer-footer drawer-login-footer">
            <button class="drawer-login-btn" @click="login()">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>
              </svg>
              登录 GitHub 后参与讨论
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.annotation-drawer-overlay {
  position: fixed;
  inset: 0;
  z-index: 250;
  background: rgba(0, 0, 0, 0.15);
}

.annotation-drawer {
  position: fixed;
  top: 0;
  right: 0;
  width: 420px;
  max-width: 90vw;
  height: 100vh;
  height: 100dvh;
  background: var(--vp-c-bg);
  border-left: 1px solid var(--vp-c-divider);
  display: flex;
  flex-direction: column;
  box-shadow: -4px 0 24px rgba(0, 0, 0, 0.1);
}

.drawer-header {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 16px;
  border-bottom: 1px solid var(--vp-c-divider);
  flex-shrink: 0;
}

.drawer-quote {
  flex: 1;
  display: flex;
  gap: 8px;
  align-items: flex-start;
  min-width: 0;
}

.quote-icon {
  flex-shrink: 0;
  color: var(--vp-c-text-3);
  margin-top: 2px;
}

.quote-text {
  font-size: 13px;
  line-height: 1.6;
  color: var(--vp-c-text-2);
  font-style: italic;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.drawer-close {
  flex-shrink: 0;
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
.drawer-close:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.drawer-body {
  flex: 1;
  overflow-y: auto;
  padding: 8px 16px;
}

.drawer-thread {
  border-bottom: 1px solid var(--vp-c-divider);
  padding-bottom: 8px;
}
.drawer-thread:last-child {
  border-bottom: none;
}

.drawer-empty {
  text-align: center;
  color: var(--vp-c-text-3);
  font-size: 14px;
  padding: 32px 0;
}

.drawer-footer {
  flex-shrink: 0;
  padding: 12px 16px;
  border-top: 1px solid var(--vp-c-divider);
}

.drawer-login-footer {
  display: flex;
  justify-content: center;
}

.drawer-login-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
}

.drawer-login-btn:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
}

/* Slide animation */
.drawer-slide-enter-active,
.drawer-slide-leave-active {
  transition: opacity 0.25s ease;
}
.drawer-slide-enter-active .annotation-drawer,
.drawer-slide-leave-active .annotation-drawer {
  transition: transform 0.25s ease;
}
.drawer-slide-enter-from,
.drawer-slide-leave-to {
  opacity: 0;
}
.drawer-slide-enter-from .annotation-drawer,
.drawer-slide-leave-to .annotation-drawer {
  transform: translateX(100%);
}

@media (max-width: 768px) {
  .annotation-drawer {
    width: 100vw;
    max-width: 100vw;
  }
}
</style>
