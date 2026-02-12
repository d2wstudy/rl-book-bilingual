<script setup lang="ts">
import { useLang } from '../composables/useLang'

const { defaultLang, toggleDefault } = useLang()
</script>

<template>
  <button class="lang-switch" @click="toggleDefault" :title="defaultLang === 'zh' ? '切换默认语言为英文' : '切换默认语言为中文'">
    <svg class="lang-icon" width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="8" cy="8" r="6.5"/>
      <ellipse cx="8" cy="8" rx="3" ry="6.5"/>
      <line x1="1.5" y1="8" x2="14.5" y2="8"/>
    </svg>
    <span>{{ defaultLang === 'zh' ? '中文' : 'EN' }}</span>
  </button>
</template>

<style scoped>
.lang-switch {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  min-width: 68px;
  padding: 2px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}

.lang-icon {
  flex-shrink: 0;
}

.lang-switch:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
</style>
