<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const props = defineProps<{ t: string }>()

const show = ref(false)
const wrapper = ref<HTMLElement | null>(null)

function toggle(e: Event) {
  e.stopPropagation()
  show.value = !show.value
}

function onClickOutside(e: Event) {
  if (wrapper.value && !wrapper.value.contains(e.target as Node)) {
    show.value = false
  }
}

onMounted(() => document.addEventListener('click', onClickOutside))
onUnmounted(() => document.removeEventListener('click', onClickOutside))
</script>

<template>
  <span class="anno" ref="wrapper">
    <span class="anno-trigger" :class="{ active: show }" @click="toggle"><slot /></span>
    <Transition name="anno-fade">
      <span v-if="show" class="anno-popup">{{ t }}</span>
    </Transition>
  </span>
</template>

<style scoped>
.anno {
  position: relative;
  display: inline;
}

.anno-trigger {
  text-decoration: underline;
  text-decoration-color: var(--vp-c-brand-2);
  text-decoration-thickness: 2px;
  text-underline-offset: 3px;
  cursor: pointer;
  transition: background 0.15s;
  border-radius: 2px;
}

.anno-trigger:hover,
.anno-trigger.active {
  background: var(--vp-c-brand-soft);
}

.anno-popup {
  position: absolute;
  bottom: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--vp-c-text-2);
  width: max-content;
  max-width: 320px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
  z-index: 100;
  white-space: normal;
}

.anno-popup::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: var(--vp-c-bg-elv);
}

.anno-fade-enter-active,
.anno-fade-leave-active {
  transition: opacity 0.15s, transform 0.15s;
}

.anno-fade-enter-from,
.anno-fade-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(4px);
}
</style>
