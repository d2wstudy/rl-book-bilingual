<script setup lang="ts">
import { ref, reactive, onMounted, watch, computed } from 'vue'
import { useRoute } from 'vitepress'
import { useAuth } from '../composables/useAuth'
import { useComments, type ReactionGroup } from '../composables/useComments'
import { useMarkdown } from '../composables/useMarkdown'
import MarkdownEditor from './MarkdownEditor.vue'

const { user, token, login } = useAuth()
const { comments, loaded, loadComments, addComment, replyToComment, toggleReaction } = useComments()
const { renderMarkdown } = useMarkdown()
const route = useRoute()

const showEditor = ref(false)
const submitting = ref(false)

const replyingTo = ref<string | null>(null)
const replyTarget = ref<{ author: string } | null>(null)
const replySubmitting = ref(false)

const expandedReplies = reactive<Record<string, boolean>>({})

// Reaction emoji picker state
const pickerOpenFor = ref<string | null>(null)

const REACTION_EMOJI: Record<string, string> = {
  THUMBS_UP: '\u{1F44D}',
  THUMBS_DOWN: '\u{1F44E}',
  LAUGH: '\u{1F604}',
  HOORAY: '\u{1F389}',
  HEART: '\u2764\uFE0F',
  ROCKET: '\u{1F680}',
  EYES: '\u{1F440}',
}

const PICKER_REACTIONS = ['THUMBS_UP', 'HEART', 'LAUGH', 'HOORAY', 'ROCKET', 'EYES']

const totalCount = computed(() =>
  comments.value.reduce((sum, c) => sum + 1 + c.replies.length, 0)
)

onMounted(() => loadComments(route.path))
watch(() => route.path, (path) => loadComments(path))
watch(token, () => loadComments(route.path))

function toggleReplies(commentId: string) {
  expandedReplies[commentId] = !expandedReplies[commentId]
}

function startReply(commentId: string, mentionAuthor?: string) {
  if (replyingTo.value === commentId && replyTarget.value?.author === mentionAuthor) {
    replyingTo.value = null
    replyTarget.value = null
    return
  }
  replyingTo.value = commentId
  replyTarget.value = mentionAuthor ? { author: mentionAuthor } : null
  expandedReplies[commentId] = true
}

async function onSubmit(text: string) {
  submitting.value = true
  try {
    await addComment(route.path, text)
    showEditor.value = false
  } finally {
    submitting.value = false
  }
}

async function onReplySubmit(commentId: string, text: string) {
  replySubmitting.value = true
  try {
    const body = replyTarget.value ? `@${replyTarget.value.author} ${text}` : text
    await replyToComment(route.path, commentId, body)
    replyingTo.value = null
    replyTarget.value = null
    expandedReplies[commentId] = true
  } finally {
    replySubmitting.value = false
  }
}

function togglePicker(subjectId: string) {
  pickerOpenFor.value = pickerOpenFor.value === subjectId ? null : subjectId
}

async function onReact(subjectId: string, content: string) {
  pickerOpenFor.value = null
  await toggleReaction(subjectId, content)
}

function formatTime(iso: string) {
  const d = new Date(iso)
  const now = Date.now()
  const diff = now - d.getTime()
  if (diff < 60000) return '刚刚'
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`
  if (diff < 2592000000) return `${Math.floor(diff / 86400000)} 天前`
  return d.toLocaleDateString('zh-CN')
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
        <!-- Top-level comment -->
        <div class="comment-item">
          <img :src="c.authorAvatar" class="comment-avatar" :alt="c.author" />
          <div class="comment-content">
            <div class="comment-meta">
              <span class="comment-author">{{ c.author }}</span>
              <span class="comment-time">{{ formatTime(c.createdAt) }}</span>
            </div>
            <div class="comment-body" v-html="renderMarkdown(c.body)" />

            <!-- Reactions display -->
            <div class="reactions-bar">
              <button
                v-for="r in c.reactions"
                :key="r.content"
                class="reaction-pill"
                :class="{ 'reaction-active': r.viewerHasReacted }"
                :disabled="!user"
                @click="onReact(c.id, r.content)"
              >
                {{ REACTION_EMOJI[r.content] }} {{ r.count }}
              </button>
              <div v-if="user" class="reaction-add-wrap">
                <button class="reaction-add-btn" @click="togglePicker(c.id)">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
                </button>
                <div v-if="pickerOpenFor === c.id" class="reaction-picker">
                  <button
                    v-for="key in PICKER_REACTIONS"
                    :key="key"
                    class="picker-emoji"
                    @click="onReact(c.id, key)"
                  >{{ REACTION_EMOJI[key] }}</button>
                </div>
              </div>
            </div>

            <div class="comment-actions">
              <button v-if="user" class="action-btn" @click="startReply(c.id)">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 17 4 12 9 7"/><path d="M20 18v-2a4 4 0 0 0-4-4H4"/></svg>
                回复
              </button>
              <button
                v-if="c.replies.length"
                class="action-btn expand-btn"
                @click="toggleReplies(c.id)"
              >
                <svg
                  width="12" height="12" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
                  :class="{ 'chevron-open': expandedReplies[c.id] }"
                  class="chevron-icon"
                >
                  <polyline points="6 9 12 15 18 9"/>
                </svg>
                {{ c.replies.length }} 条回复
              </button>
            </div>
          </div>
        </div>

        <!-- Replies (flat, collapsible) -->
        <div v-if="c.replies.length && expandedReplies[c.id]" class="replies-section">
          <div v-for="r in c.replies" :key="r.id" class="reply-item">
            <img :src="r.authorAvatar" class="reply-avatar" :alt="r.author" />
            <div class="reply-content">
              <div class="comment-meta">
                <span class="comment-author">{{ r.author }}</span>
                <span class="comment-time">{{ formatTime(r.createdAt) }}</span>
              </div>
              <div class="comment-body" v-html="renderMarkdown(r.body)" />

              <!-- Reply reactions -->
              <div class="reactions-bar">
                <button
                  v-for="rg in r.reactions"
                  :key="rg.content"
                  class="reaction-pill"
                  :class="{ 'reaction-active': rg.viewerHasReacted }"
                  :disabled="!user"
                  @click="onReact(r.id, rg.content)"
                >
                  {{ REACTION_EMOJI[rg.content] }} {{ rg.count }}
                </button>
                <div v-if="user" class="reaction-add-wrap">
                  <button class="reaction-add-btn" @click="togglePicker(r.id)">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
                  </button>
                  <div v-if="pickerOpenFor === r.id" class="reaction-picker">
                    <button
                      v-for="key in PICKER_REACTIONS"
                      :key="key"
                      class="picker-emoji"
                      @click="onReact(r.id, key)"
                    >{{ REACTION_EMOJI[key] }}</button>
                  </div>
                </div>
              </div>

              <div class="comment-actions">
                <button v-if="user" class="action-btn" @click="startReply(c.id, r.author)">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 17 4 12 9 7"/><path d="M20 18v-2a4 4 0 0 0-4-4H4"/></svg>
                  回复
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Reply editor -->
        <div v-if="replyingTo === c.id" class="reply-editor-section">
          <div v-if="replyTarget" class="reply-indicator">
            回复 <span class="reply-indicator-author">@{{ replyTarget.author }}</span>
            <button class="reply-indicator-clear" @click="replyTarget = null" title="改为回复主楼">&times;</button>
          </div>
          <MarkdownEditor
            :placeholder="replyTarget ? `回复 @${replyTarget.author}...` : '写下你的回复... 支持 Markdown 语法'"
            :submit-label="replySubmitting ? '提交中...' : '回复'"
            @submit="(text: string) => onReplySubmit(c.id, text)"
            @cancel="replyingTo = null; replyTarget = null"
          />
        </div>
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

.comment-item {
  display: flex;
  gap: 12px;
  padding: 12px 0 4px;
}

.comment-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  flex-shrink: 0;
}

.comment-content {
  flex: 1;
  min-width: 0;
}

.comment-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.comment-author {
  font-size: 13px;
  font-weight: 600;
  color: var(--vp-c-text-1);
}

.comment-time {
  font-size: 12px;
  color: var(--vp-c-text-3);
}

.comment-body {
  font-size: 14px;
  line-height: 1.7;
  color: var(--vp-c-text-1);
}
.comment-body :deep(p) { margin: 4px 0; }
.comment-body :deep(code) { background: var(--vp-c-bg-soft); padding: 2px 4px; border-radius: 3px; font-size: 13px; }
.comment-body :deep(pre) { background: var(--vp-c-bg-soft); padding: 8px; border-radius: 6px; overflow-x: auto; margin: 6px 0; }
.comment-body :deep(pre code) { background: none; padding: 0; }
.comment-body :deep(blockquote) { border-left: 3px solid var(--vp-c-divider); padding-left: 8px; color: var(--vp-c-text-2); margin: 4px 0; }
.comment-body :deep(a) { color: var(--vp-c-brand-1); }

/* Reactions */
.reactions-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 6px;
  flex-wrap: wrap;
}

.reaction-pill {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 2px 8px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  background: var(--vp-c-bg);
  color: var(--vp-c-text-2);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
  line-height: 1.4;
}
.reaction-pill:hover:not(:disabled) {
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-bg-soft);
}
.reaction-pill:disabled {
  cursor: default;
}
.reaction-active {
  border-color: var(--vp-c-brand-1);
  background: color-mix(in srgb, var(--vp-c-brand-1) 10%, transparent);
  color: var(--vp-c-brand-1);
}

.reaction-add-wrap {
  position: relative;
}

.reaction-add-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 24px;
  border: 1px dashed var(--vp-c-divider);
  border-radius: 12px;
  background: none;
  color: var(--vp-c-text-3);
  cursor: pointer;
  transition: all 0.15s;
}
.reaction-add-btn:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  background: var(--vp-c-bg-soft);
}

.reaction-picker {
  position: absolute;
  bottom: calc(100% + 6px);
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 2px;
  padding: 4px 6px;
  background: var(--vp-c-bg-elv);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  z-index: 10;
  white-space: nowrap;
}

.picker-emoji {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 6px;
  background: none;
  font-size: 18px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.1s;
}
.picker-emoji:hover {
  background: var(--vp-c-bg-soft);
}

.comment-actions {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 4px;
}

.action-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border: none;
  border-radius: 4px;
  background: none;
  color: var(--vp-c-text-3);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}
.action-btn:hover {
  color: var(--vp-c-brand-1);
  background: var(--vp-c-bg-soft);
}

.expand-btn {
  color: var(--vp-c-brand-1);
  font-weight: 500;
}

.chevron-icon {
  transition: transform 0.2s;
}
.chevron-open {
  transform: rotate(180deg);
}

/* Replies section */
.replies-section {
  margin-left: 48px;
  border-left: 2px solid var(--vp-c-divider);
}

.reply-item {
  display: flex;
  gap: 10px;
  padding: 10px 0 4px 14px;
}

.reply-avatar {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  flex-shrink: 0;
}

.reply-content {
  flex: 1;
  min-width: 0;
}

/* Reply editor */
.reply-editor-section {
  margin-left: 48px;
  margin-top: 8px;
}

.reply-indicator {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--vp-c-text-3);
  margin-bottom: 6px;
  padding: 4px 8px;
  background: var(--vp-c-bg-soft);
  border-radius: 4px;
  width: fit-content;
}

.reply-indicator-author {
  color: var(--vp-c-brand-1);
  font-weight: 500;
}

.reply-indicator-clear {
  border: none;
  background: none;
  color: var(--vp-c-text-3);
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
  padding: 0 2px;
  margin-left: 4px;
}
.reply-indicator-clear:hover {
  color: var(--vp-c-text-1);
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
