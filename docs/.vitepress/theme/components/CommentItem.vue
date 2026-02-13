<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useAuth } from '../composables/useAuth'
import { useMarkdown } from '../composables/useMarkdown'
import { formatRelativeTime, type ReactionGroup, type ThreadReply } from '../composables/useDiscussionThread'
import MarkdownEditor from './MarkdownEditor.vue'

const props = withDefaults(defineProps<{
  id: string
  body: string
  author: string
  authorAvatar: string
  createdAt: string
  reactions: ReactionGroup[]
  replies: ThreadReply[]
  compact?: boolean
}>(), {
  compact: false,
})

const emit = defineEmits<{
  reply: [commentId: string, body: string]
  react: [subjectId: string, content: string]
}>()

const { user } = useAuth()
const { renderMarkdown } = useMarkdown()

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

const pickerOpenFor = ref<string | null>(null)
const expandedReplies = ref(false)
const replyingTo = ref(false)
const replyMention = ref<string | null>(null)
const replySubmitting = ref(false)

function togglePicker(subjectId: string) {
  pickerOpenFor.value = pickerOpenFor.value === subjectId ? null : subjectId
}

function onReact(subjectId: string, content: string) {
  pickerOpenFor.value = null
  emit('react', subjectId, content)
}

function startReply(mentionAuthor?: string) {
  if (replyingTo.value && replyMention.value === (mentionAuthor ?? null)) {
    replyingTo.value = false
    replyMention.value = null
    return
  }
  replyingTo.value = true
  replyMention.value = mentionAuthor ?? null
  expandedReplies.value = true
}

async function onReplySubmit(text: string) {
  replySubmitting.value = true
  try {
    const body = replyMention.value ? `@${replyMention.value} ${text}` : text
    emit('reply', props.id, body)
    replyingTo.value = false
    replyMention.value = null
    expandedReplies.value = true
  } finally {
    replySubmitting.value = false
  }
}
</script>

<template>
  <div class="comment-item" :class="{ compact }">
    <img :src="authorAvatar" class="comment-avatar" :alt="author" />
    <div class="comment-content">
      <div class="comment-meta">
        <span class="comment-author">{{ author }}</span>
        <span class="comment-time">{{ formatRelativeTime(createdAt) }}</span>
      </div>

      <!-- Slot for extra content before body (e.g. annotation quote) -->
      <slot name="before-body" />

      <div class="comment-body" v-html="renderMarkdown(body)" />

      <!-- Reactions -->
      <div class="reactions-bar">
        <button
          v-for="r in reactions"
          :key="r.content"
          class="reaction-pill"
          :class="{ 'reaction-active': r.viewerHasReacted }"
          :disabled="!user"
          @click="onReact(id, r.content)"
        >
          {{ REACTION_EMOJI[r.content] }} {{ r.count }}
        </button>
        <div v-if="user" class="reaction-add-wrap">
          <button class="reaction-add-btn" @click="togglePicker(id)">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
          </button>
          <div v-if="pickerOpenFor === id" class="reaction-picker">
            <button
              v-for="key in PICKER_REACTIONS"
              :key="key"
              class="picker-emoji"
              @click="onReact(id, key)"
            >{{ REACTION_EMOJI[key] }}</button>
          </div>
        </div>
      </div>

      <!-- Actions -->
      <div class="comment-actions">
        <button v-if="user" class="action-btn" @click="startReply()">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 17 4 12 9 7"/><path d="M20 18v-2a4 4 0 0 0-4-4H4"/></svg>
          回复
        </button>
        <button
          v-if="replies.length"
          class="action-btn expand-btn"
          @click="expandedReplies = !expandedReplies"
        >
          <svg
            width="12" height="12" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
            :class="{ 'chevron-open': expandedReplies }"
            class="chevron-icon"
          >
            <polyline points="6 9 12 15 18 9"/>
          </svg>
          {{ replies.length }} 条回复
        </button>
      </div>
    </div>
  </div>

  <!-- Replies -->
  <div v-if="replies.length && expandedReplies" class="replies-section" :class="{ compact }">
    <div v-for="r in replies" :key="r.id" class="reply-item">
      <img :src="r.authorAvatar" class="reply-avatar" :alt="r.author" />
      <div class="reply-content">
        <div class="comment-meta">
          <span class="comment-author">{{ r.author }}</span>
          <span class="comment-time">{{ formatRelativeTime(r.createdAt) }}</span>
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
          <button v-if="user" class="action-btn" @click="startReply(r.author)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 17 4 12 9 7"/><path d="M20 18v-2a4 4 0 0 0-4-4H4"/></svg>
            回复
          </button>
        </div>
      </div>
    </div>
  </div>

  <!-- Reply editor -->
  <div v-if="replyingTo" class="reply-editor-section" :class="{ compact }">
    <div v-if="replyMention" class="reply-indicator">
      回复 <span class="reply-indicator-author">@{{ replyMention }}</span>
      <button class="reply-indicator-clear" @click="replyMention = null" title="改为回复主楼">&times;</button>
    </div>
    <MarkdownEditor
      :placeholder="replyMention ? `回复 @${replyMention}...` : '写下你的回复... 支持 Markdown 语法'"
      :submit-label="replySubmitting ? '提交中...' : '回复'"
      @submit="onReplySubmit"
      @cancel="replyingTo = false; replyMention = null"
    />
  </div>
</template>

<style scoped>
.comment-item {
  display: flex;
  gap: 12px;
  padding: 12px 0 4px;
}
.comment-item.compact {
  gap: 10px;
  padding: 10px 0 4px;
}

.comment-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  flex-shrink: 0;
}
.compact .comment-avatar {
  width: 30px;
  height: 30px;
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
.comment-body :deep(pre code) { background: none; padding: 0; font-size: 13px; }
.comment-body :deep(blockquote) { border-left: 3px solid var(--vp-c-divider); padding-left: 8px; color: var(--vp-c-text-2); margin: 4px 0; }
.comment-body :deep(a) { color: var(--vp-c-brand-1); text-decoration: none; }
.comment-body :deep(a:hover) { text-decoration: underline; }
.comment-body :deep(h1),
.comment-body :deep(h2),
.comment-body :deep(h3),
.comment-body :deep(h4) {
  margin: 8px 0 4px; font-size: 14px; font-weight: 600;
  border: none; padding: 0; line-height: 1.5;
}
.comment-body :deep(h1) { font-size: 17px; }
.comment-body :deep(h2) { font-size: 16px; }
.comment-body :deep(h3) { font-size: 15px; }
.comment-body :deep(ul),
.comment-body :deep(ol) { margin: 4px 0; padding-left: 20px; }
.comment-body :deep(li) { margin: 2px 0; }
.comment-body :deep(li + li) { margin-top: 2px; }
.comment-body :deep(hr) { margin: 8px 0; border: none; border-top: 1px solid var(--vp-c-divider); }
.comment-body :deep(img) { max-width: 100%; border-radius: 6px; margin: 4px 0; }
.comment-body :deep(table) { border-collapse: collapse; margin: 6px 0; font-size: 13px; width: auto; }
.comment-body :deep(th),
.comment-body :deep(td) { border: 1px solid var(--vp-c-divider); padding: 4px 8px; }
.comment-body :deep(th) { background: var(--vp-c-bg-soft); font-weight: 600; }
.comment-body :deep(del) { color: var(--vp-c-text-3); }
.comment-body :deep(.mention) {
  color: var(--vp-c-brand-1); font-weight: 500;
  background: color-mix(in srgb, var(--vp-c-brand-1) 8%, transparent);
  padding: 1px 4px; border-radius: 3px;
}

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
.reaction-pill:disabled { cursor: default; }
.reaction-active {
  border-color: var(--vp-c-brand-1);
  background: color-mix(in srgb, var(--vp-c-brand-1) 10%, transparent);
  color: var(--vp-c-brand-1);
}

.reaction-add-wrap { position: relative; }

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
.picker-emoji:hover { background: var(--vp-c-bg-soft); }

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

.chevron-icon { transition: transform 0.2s; }
.chevron-open { transform: rotate(180deg); }

/* Replies */
.replies-section {
  margin-left: 48px;
  border-left: 2px solid var(--vp-c-divider);
}
.replies-section.compact {
  margin-left: 40px;
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
.compact .reply-avatar {
  width: 24px;
  height: 24px;
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
.reply-editor-section.compact {
  margin-left: 40px;
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
.reply-indicator-clear:hover { color: var(--vp-c-text-1); }
</style>
