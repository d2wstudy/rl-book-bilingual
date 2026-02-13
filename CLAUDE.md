# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bilingual (Chinese/English) online reading platform for "Reinforcement Learning: An Introduction" (2nd edition), built with VitePress. It supports paragraph-level language switching, reader annotations via GitHub Discussions, and GitHub OAuth authentication.

## Commands

- `npm run dev` — Start dev server on port 15689
- `npm run build` — Build static site to `docs/.vitepress/dist`
- `npm run preview` — Preview production build

Worker (in `worker/` directory):
- `npx wrangler dev` — Run Cloudflare Worker locally
- `npx wrangler deploy` — Deploy Worker to Cloudflare

No test runner or linter is configured in this project.

## Architecture

### Bilingual System

Markdown files in `docs/chapters/` use custom containers `::: en` and `::: zh` to mark English/Chinese paragraphs. A third container `::: notes` renders author notes (`<div class="author-notes">`). The rendering pipeline:

1. `docs/.vitepress/config.ts` registers markdown-it-container renderers for `en`, `zh`, and `notes` that output `<div class="bilingual-en">` / `<div class="bilingual-zh">` / `<div class="author-notes">`
2. `docs/.vitepress/theme/index.ts` — `pairBilingualBlocks()` runs at runtime on mount and route change, finds adjacent `.bilingual-en` + `.bilingual-zh` divs, wraps them in `.bilingual-pair`, and injects a per-pair toggle button. It is idempotent (skips if `.bilingual-pair` already exists) and resets its counter on route change. Each pair gets two IDs: a content-stable `data-pair-id` (heading slug + djb2 hash of English text) and a legacy counter-based `data-pair-id-legacy` (`{headingSlug}-p{counter}`) for backward compatibility with existing annotations.
3. `docs/.vitepress/theme/composables/useLang.ts` — manages global default language (localStorage), `applyDefaultLang()` shows/hides blocks while respecting per-paragraph manual overrides (`.flipped-manual` class)

### Theme Layout

`docs/.vitepress/theme/index.ts` extends VitePress DefaultTheme and uses layout slots:
- `nav-bar-content-after` → renders `LanguageToggle` + `LoginButton` in the navbar
- `doc-after` → renders `AnnotationLayer` below document content

Five global components are registered: `LanguageToggle`, `LoginButton`, `Anno`, `AnnotationLayer`, `ChapterComments`.

### Authentication Flow

GitHub OAuth via Cloudflare Worker proxy:

1. `LoginButton.vue` redirects to GitHub OAuth
2. GitHub callbacks with `code` parameter
3. `useAuth.ts` → `exchangeCode()` calls Worker `/api/auth` endpoint
4. Worker exchanges code for access_token with GitHub
5. Token stored in sessionStorage

Environment-specific OAuth apps configured in `docs/.env.development` and `docs/.env.production`.

### Discussion-Backed Features

Both annotations and chapter comments use GitHub Discussions as storage (repo: `d2wstudy/rl-book-bilingual`), one discussion per page, separated by Discussion category:
- **Annotations** (category: `Notes`) — `useAnnotations.ts` stores text-selection notes as JSON in comment bodies (`{ type: "annotation", paragraphId, startOffset, endOffset, selectedText, prefix, suffix, note }`)
- **Chapter comments** (category: `Announcements`) — `useComments.ts` stores plain-text discussion comments

Shared logic lives in `useDiscussionThread.ts` (types `ReactionGroup`/`ThreadReply`, reaction toggling, reply mapping, time formatting). Both composables follow the same pattern: module-level singleton state, deduplication of in-flight loads, optimistic local updates after writes.

`useGithubGql.ts` is the data layer: reads go through the Worker proxy (`GET /api/discussions`), mutations go directly to GitHub GraphQL API with the user's OAuth token. Worker URL is hardcoded: `https://rl-book-auth.d2w.workers.dev`.

### Cloudflare Worker (`worker/index.js`)

OAuth proxy and Discussions read proxy with four routes:
- `GET /api/discussions?path=xxx&category=Notes[&id=xxx]` — read discussions with two-tier caching
- `POST /api/cache/purge?path=xxx&category=Notes[&id=xxx]` — purge/refill cache after writes (requires `Authorization: Bearer ...` or `X-Purge-Key`)
- `POST /api/auth` — exchange OAuth code for token
- `POST /api/revoke` — revoke OAuth authorization

Two-tier caching via Cloudflare Cache API:
- **Shared cache** (5 min TTL) — discussion content with `viewerHasReacted` stripped. Populated by first authenticated user's token or PAT as fallback. Purge triggers immediate refill.
- **Per-user reaction cache** (7 day TTL, keyed by token hash) — lightweight `viewerHasReacted` overlay fetched via batch `nodes(ids:)` query, merged onto shared cache at response time.
- Reaction toggles use in-place `totalCount` patching on shared cache (no full refill).

Uses separate client ID/secret pairs for dev vs production (env vars with `_DEV` suffix). Worker secrets are set via `wrangler secret`.

## Key Tech

- Vue 3 + VitePress 1.6.x
- Built-in VitePress math rendering (`markdown.math: true`)
- markdown-it-container for bilingual blocks
- marked + DOMPurify for client-side Markdown rendering in annotations
- Cloudflare Workers for serverless OAuth/Discussions proxy

## Deployment

- Push to `main` triggers GitHub Actions (`.github/workflows/deploy.yml`): `npm ci` → `npm run build` → deploy to GitHub Pages
- Worker deployed separately via `npx wrangler deploy` from `worker/` directory

## Conventions

- Always respond in Chinese-simplified (per user preference).
- Chapter files follow naming: `docs/chapters/ch{NN}-{slug}.md`
- Base path is `/rl-book-bilingual/` (GitHub Pages deployment)
