<script setup lang="ts">
import MarkdownEditor from './MarkdownEditor.vue'

defineProps<{
  selectedText: string
}>()

const emit = defineEmits<{
  submit: [note: string]
  cancel: []
}>()

function onOverlayClick(e: MouseEvent) {
  if ((e.target as HTMLElement).classList.contains('note-editor-overlay')) {
    emit('cancel')
  }
}
</script>

<template>
  <Teleport to="body">
    <div class="note-editor-overlay" @click="onOverlayClick">
      <div class="note-editor-modal">
        <div class="note-editor-quote">「{{ selectedText }}」</div>
        <MarkdownEditor
          placeholder="写下你的笔记... 支持 Markdown 语法"
          submit-label="提交"
          @submit="(text) => emit('submit', text)"
          @cancel="emit('cancel')"
        />
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.note-editor-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.3);
  z-index: 300;
  display: flex;
  align-items: center;
  justify-content: center;
}

.note-editor-modal {
  width: min(90vw, 680px);
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.note-editor-quote {
  padding: 12px 16px;
  font-size: 12px;
  color: var(--vp-c-text-3);
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-bottom: none;
  border-radius: 12px 12px 0 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.note-editor-modal :deep(.md-editor) {
  border-radius: 0 0 12px 12px;
  border-top: none;
}

@media (max-width: 768px) {
  .note-editor-modal {
    width: 100vw;
    height: 100vh;
    max-height: 100vh;
    border-radius: 0;
  }
  .note-editor-quote { border-radius: 0; }
  .note-editor-modal :deep(.md-editor) { border-radius: 0; flex: 1; }
  .note-editor-modal :deep(.md-editor-body) { flex: 1; max-height: none; }
  .note-editor-modal :deep(.md-textarea) { min-height: 0; height: 100%; }
  .note-editor-modal :deep(.md-preview) { min-height: 0; height: 100%; }
}
</style>
