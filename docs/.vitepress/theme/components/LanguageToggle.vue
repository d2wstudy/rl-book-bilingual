<script setup lang="ts">
import { useLang } from '../composables/useLang'

const { defaultLang, toggleDefault } = useLang()
</script>

<template>
  <button class="lang-switch" @click="toggleDefault" :title="defaultLang === 'zh' ? '切换默认语言为英文' : '切换默认语言为中文'">
    {{ defaultLang === 'zh' ? '中文' : 'EN' }}
  </button>
</template>

<style scoped>
.lang-switch {
  padding: 2px 10px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
}

.lang-switch:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
</style>
