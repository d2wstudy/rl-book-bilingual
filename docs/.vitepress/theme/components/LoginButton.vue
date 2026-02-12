<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { useAuth } from '../composables/useAuth'

const { user, loading, login, logout } = useAuth()
const showMenu = ref(false)
const btnRef = ref<HTMLElement | null>(null)
const menuStyle = ref({ top: '0px', right: '0px' })
let hideTimer: ReturnType<typeof setTimeout> | null = null

function updateMenuPos() {
  if (!btnRef.value) return
  const rect = btnRef.value.getBoundingClientRect()
  menuStyle.value = {
    top: rect.bottom + 4 + 'px',
    right: window.innerWidth - rect.right + 'px',
  }
}

function onEnter() {
  if (!user.value) return
  if (hideTimer) { clearTimeout(hideTimer); hideTimer = null }
  showMenu.value = true
  nextTick(updateMenuPos)
}

function onLeave() {
  hideTimer = setTimeout(() => { showMenu.value = false }, 150)
}

function onMenuEnter() {
  if (hideTimer) { clearTimeout(hideTimer); hideTimer = null }
}

function onMenuLeave() {
  showMenu.value = false
}

function handleClick() {
  if (!user.value) {
    login()
  }
}

function handleLogout() {
  showMenu.value = false
  logout()
}

function onClickOutside(e: MouseEvent) {
  if (!showMenu.value) return
  if (btnRef.value?.contains(e.target as Node)) return
  const menu = document.querySelector('.login-menu-portal')
  if (menu?.contains(e.target as Node)) return
  showMenu.value = false
}

onMounted(() => document.addEventListener('click', onClickOutside, true))
onUnmounted(() => {
  document.removeEventListener('click', onClickOutside, true)
  if (hideTimer) clearTimeout(hideTimer)
})
</script>

<template>
  <div class="login-btn-wrapper">
    <span v-if="loading" class="avatar-placeholder" />
    <button
      v-else
      ref="btnRef"
      class="avatar-btn"
      @click="handleClick"
      @mouseenter="onEnter"
      @mouseleave="onLeave"
    >
      <img v-if="user" :src="user.avatar_url" :alt="user.login" class="user-avatar" />
      <svg v-else width="32" height="32" viewBox="0 0 20 20" class="default-avatar">
        <circle cx="10" cy="10" r="10" fill="#D1D5DB"/>
        <clipPath id="avatar-clip"><circle cx="10" cy="10" r="10"/></clipPath>
        <g clip-path="url(#avatar-clip)">
          <circle cx="10" cy="8" r="3.2" fill="#9CA3AF"/>
          <ellipse cx="10" cy="19" rx="7" ry="5.5" fill="#9CA3AF"/>
        </g>
      </svg>
    </button>

    <Teleport to="body">
      <div
        v-if="showMenu && user"
        class="login-menu-portal"
        :style="menuStyle"
        @mouseenter="onMenuEnter"
        @mouseleave="onMenuLeave"
      >
        <div class="menu-header">
          <img :src="user.avatar_url" class="menu-avatar" />
          <span class="menu-name">{{ user.login }}</span>
        </div>
        <div class="menu-divider" />
        <button class="menu-item danger" @click="handleLogout">退出登录</button>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.login-btn-wrapper {
  display: flex;
  align-items: center;
}

.avatar-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  border: none;
  background: none;
  cursor: pointer;
  border-radius: 50%;
  transition: opacity 0.2s;
}

.avatar-btn:hover {
  opacity: 0.8;
}

.user-avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
}

.default-avatar {
  border-radius: 50%;
  display: block;
}

.avatar-placeholder {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: var(--vp-c-bg-soft);
}
</style>

<style>
.login-menu-portal {
  position: fixed;
  min-width: 160px;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
  z-index: 9999;
  padding: 6px 0;
}

.login-menu-portal .menu-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
}

.login-menu-portal .menu-avatar {
  width: 24px;
  height: 24px;
  border-radius: 50%;
}

.login-menu-portal .menu-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--vp-c-text-1);
}

.login-menu-portal .menu-divider {
  height: 1px;
  background: var(--vp-c-divider);
  margin: 4px 0;
}

.login-menu-portal .menu-item {
  display: block;
  width: 100%;
  padding: 8px 12px;
  border: none;
  background: none;
  color: var(--vp-c-text-2);
  font-size: 13px;
  text-align: left;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}

.login-menu-portal .menu-item:hover {
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-1);
}

.login-menu-portal .menu-item.danger:hover {
  color: var(--vp-c-danger-1, #e5484d);
}
</style>
