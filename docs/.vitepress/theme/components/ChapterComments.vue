<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { useRoute } from 'vitepress'
import { useAuth } from '../composables/useAuth'
import { useComments } from '../composables/useComments'
import { purgeWorkerCache } from '../composables/useGithubGql'
import { formatRelativeTime } from '../composables/useDiscussionThread'
import MarkdownEditor from './MarkdownEditor.vue'
import CommentItem from './CommentItem.vue'

const { user, token, login } = useAuth()
const { comments, loaded, loadComments, addComment, replyToComment, toggleReaction } = useComments()
const route = useRoute()

const showEditor = ref(false)
const submitting = ref(false)

const totalCount = computed(() =>
  comments.value.reduce((sum, c) => sum + 1 + c.replies.length, 0)
)

// Single watcher with immediate: true replaces both onMounted + watch(route.path)
watch(() => route.path, (path) => loadComments(path), { immediate: true })

// Re-fetch with user's own token after login (or fall back to Worker after logout)
watch(token, () => {
  loadComments(route.path, true)
})

async function onSubmit(text: string) {
  submitting.value = true
  try {
    await addComment(route.path, text)
    showEditor.value = false
  } finally {
    submitting.value = false
  }
}

async function onReply(commentId: string, body: string) {
  await replyToComment(route.path, commentId, body)
}

async function onReact(subjectId: string, content: string) {
  await toggleReaction(subjectId, content)
  await purgeWorkerCache(route.path, 'Announcements', true)
}
</script>

<template>
  <div class="chapter-comments">
    <div class="comments-header">
      <h3 class="comments-title">讨论 ({{ totalCount }})</h3>
    </div>

    <!-- Comment list -->
    <div v-if="loaded && comments.length" class="comments-list">
      <div v-for="c in comments" :key="c.id" class="comment-thread">
        <CommentItem
          :id="c.id"
          :body="c.body"
          :author="c.author"
          :author-avatar="c.authorAvatar"
          :created-at="c.createdAt"
          :reactions="c.reactions"
          :replies="c.replies"
          @reply="onReply"
          @react="onReact"
        />
      </div>
    </div>

    <div v-else-if="loaded && !comments.length && user" class="comments-empty">
      还没有评论，来发表第一条吧
    </div>

    <!-- Editor / Login -->
    <div class="comments-footer">
      <template v-if="user">
        <div v-if="!showEditor" class="comment-input-placeholder" @click="showEditor = true">
          <img :src="user.avatar_url" class="comment-avatar-sm" />
          <span>写下你的评论...</span>
        </div>
        <div v-else class="comment-editor-wrap">
          <MarkdownEditor
            placeholder="写下你的评论... 支持 Markdown 语法"
            :submit-label="submitting ? '提交中...' : '评论'"
            @submit="onSubmit"
            @cancel="showEditor = false"
          />
        </div>
      </template>
      <div v-else class="comments-login">
        <button class="login-btn" @click="login">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
          登录 GitHub 参与讨论
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chapter-comments {
  margin-top: 48px;
  padding-top: 24px;
  border-top: 1px solid var(--vp-c-divider);
}

.comments-title {
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 16px;
  color: var(--vp-c-text-1);
}

.comments-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 20px;
}

.comment-thread {
  border-bottom: 1px solid var(--vp-c-divider);
  padding-bottom: 12px;
}
.comment-thread:last-child {
  border-bottom: none;
}

.comments-empty {
  text-align: center;
  color: var(--vp-c-text-3);
  font-size: 14px;
  padding: 24px 0;
}

.comments-footer {
  margin-top: 16px;
}

.comment-input-placeholder {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  cursor: text;
  color: var(--vp-c-text-3);
  font-size: 14px;
  transition: border-color 0.2s;
}
.comment-input-placeholder:hover {
  border-color: var(--vp-c-brand-1);
}

.comment-avatar-sm {
  width: 28px;
  height: 28px;
  border-radius: 50%;
}

.comments-login {
  text-align: center;
  padding: 16px 0;
}

.login-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-elv);
  color: var(--vp-c-text-1);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}
.login-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
</style>
