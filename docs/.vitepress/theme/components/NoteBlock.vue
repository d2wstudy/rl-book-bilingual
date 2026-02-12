<script setup lang="ts">
import { ref } from 'vue'

const expanded = ref(false)
</script>

<template>
  <div class="note-block" :class="{ expanded }">
    <button class="note-trigger" @click="expanded = !expanded">
      <span class="note-icon">📝</span>
      <span class="note-label">译注</span>
      <span class="note-arrow">{{ expanded ? '▾' : '▸' }}</span>
    </button>
    <div v-show="expanded" class="note-content">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.note-block {
  margin: 8px 0;
  border-left: 3px solid var(--vp-c-brand-1);
  border-radius: 0 8px 8px 0;
  background: var(--vp-c-bg-soft);
}

.note-trigger {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 6px 12px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 13px;
  color: var(--vp-c-brand-1);
}

.note-trigger:hover {
  background: var(--vp-c-bg-mute);
}

.note-icon {
  font-size: 14px;
}

.note-label {
  font-weight: 500;
}

.note-arrow {
  font-size: 11px;
  margin-left: auto;
}

.note-content {
  padding: 4px 16px 12px;
  font-size: 14px;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.note-content :deep(ul) {
  margin: 0;
  padding-left: 1.2em;
}

.note-content :deep(li + li) {
  margin-top: 4px;
}
</style>
