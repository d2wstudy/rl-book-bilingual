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
2. `docs/.vitepress/theme/index.ts` — `pairBilingualBlocks()` runs at runtime on mount and route change, finds adjacent `.bilingual-en` + `.bilingual-zh` divs, wraps them in `.bilingual-pair`, generates stable IDs (`{headingSlug}-p{counter}`), and injects a per-pair toggle button. It is idempotent (skips if `.bilingual-pair` already exists) and resets its counter on route change.
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

### Annotation System

- Storage: GitHub Discussions (one discussion per page, repo: `d2wstudy/rl-book-bilingual`)
- `AnnotationLayer.vue` captures text selection, shows annotation popup
- `useAnnotations.ts` manages CRUD via GitHub GraphQL API through `useGithubGql.ts`
- Notes serialized as JSON in discussion comments with `paragraphId`, offsets, and note text

### Cloudflare Worker (`worker/index.js`)

OAuth proxy and Discussions read proxy with three routes:
- `GET /api/discussions?path=xxx&category=Notes` — read discussions (cached 5 min via Cloudflare Cache API, uses `GITHUB_PAT`)
- `POST /api/auth` — exchange OAuth code for token
- `POST /api/revoke` — revoke OAuth authorization

Uses separate client ID/secret pairs for dev vs production (env vars with `_DEV` suffix). Worker secrets are set via `wrangler secret`.

## Key Tech

- Vue 3 + VitePress 1.6.x
- Built-in VitePress math rendering (`markdown.math: true`)
- markdown-it-container for bilingual blocks
- marked + DOMPurify for client-side Markdown rendering in annotations
- Cloudflare Workers for serverless OAuth/Discussions proxy

## Conventions

- Always respond in Chinese-simplified (per user preference).
- Chapter files follow naming: `docs/chapters/ch{NN}-{slug}.md`
- Base path is `/rl-book-bilingual/` (GitHub Pages deployment)
