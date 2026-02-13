<script setup lang="ts">
defineProps<{
  visible: boolean
  x: number
  y: number
  loggedIn: boolean
}>()

defineEmits<{
  'open-editor': []
  'login': []
}>()
</script>

<template>
  <Teleport to="body">
    <Transition name="bubble-fade">
      <div
        v-if="visible"
        class="note-bubble"
        :style="{ left: x + 'px', top: y + 'px' }"
      >
        <button v-if="loggedIn" class="note-bubble-btn" @click="$emit('open-editor')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 20h9"/>
            <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
          </svg>
          笔记
        </button>
        <button v-else class="note-bubble-btn login-bubble-btn" @click="$emit('login')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>
          </svg>
          登录后可添加笔记
        </button>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.note-bubble {
  position: fixed;
  transform: translateX(-50%) translateY(-100%);
  z-index: 200;
  pointer-events: auto;
}

.note-bubble-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-elv);
  color: var(--vp-c-text-1);
  font-size: 13px;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
  transition: all 0.15s;
  white-space: nowrap;
}

.note-bubble-btn:hover {
  color: var(--vp-c-brand-1);
}

.login-bubble-btn {
  color: var(--vp-c-text-2);
  font-size: 12px;
}

.bubble-fade-enter-active,
.bubble-fade-leave-active {
  transition: opacity 0.15s, transform 0.15s;
}

.bubble-fade-enter-from,
.bubble-fade-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(calc(-100% + 4px));
}
</style>
