<script setup lang="ts">
import { ref, onMounted } from 'vue'

const props = defineProps<{
  repo: string
  repoId?: string
  category?: string
  categoryId?: string
}>()

const container = ref<HTMLElement>()

onMounted(() => {
  if (!container.value) return
  const script = document.createElement('script')
  script.src = 'https://giscus.app/client.js'
  script.setAttribute('data-repo', props.repo)
  script.setAttribute('data-repo-id', props.repoId || '')
  script.setAttribute('data-category', props.category || 'Announcements')
  script.setAttribute('data-category-id', props.categoryId || '')
  script.setAttribute('data-mapping', 'pathname')
  script.setAttribute('data-strict', '0')
  script.setAttribute('data-reactions-enabled', '1')
  script.setAttribute('data-emit-metadata', '0')
  script.setAttribute('data-input-position', 'top')
  script.setAttribute('data-theme', 'preferred_color_scheme')
  script.setAttribute('data-lang', 'zh-CN')
  script.crossOrigin = 'anonymous'
  script.async = true
  container.value.appendChild(script)
})
</script>

<template>
  <div ref="container" class="giscus-wrapper" />
</template>

<style scoped>
.giscus-wrapper {
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid var(--vp-c-divider);
}
</style>
