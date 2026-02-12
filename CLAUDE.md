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

## Architecture

### Bilingual System

Markdown files in `docs/chapters/` use custom containers `::: en` and `::: zh` to mark English/Chinese paragraphs. The rendering pipeline:

1. `docs/.vitepress/config.ts` registers custom markdown-it-container renderers that output `<div class="bilingual-en">` / `<div class="bilingual-zh">`
2. `docs/.vitepress/theme/index.ts` — `pairBilingualBlocks()` runs at runtime, finds adjacent en/zh divs, wraps them in `.bilingual-pair`, generates stable IDs (based on preceding heading), and injects a toggle button per pair
3. `docs/.vitepress/theme/composables/useLang.ts` — manages global default language (localStorage), `applyDefaultLang()` shows/hides blocks while respecting per-paragraph manual overrides

### Authentication Flow

GitHub OAuth via Cloudflare Worker proxy:

1. `LoginButton.vue` redirects to GitHub OAuth
2. GitHub callbacks with `code` parameter
3. `useAuth.ts` → `exchangeCode()` calls Worker `/api/auth` endpoint
4. Worker exchanges code for access_token with GitHub
5. Token stored in sessionStorage

Environment-specific OAuth apps configured in `docs/.env.development` and `docs/.env.production`.

### Annotation System

- Storage: GitHub Discussions (one discussion per page)
- `AnnotationLayer.vue` captures text selection, shows annotation popup
- `useAnnotations.ts` manages CRUD via GitHub GraphQL API
- Notes serialized as JSON in discussion comments with `paragraphId`, offsets, and note text

### Cloudflare Worker (`worker/index.js`)

OAuth proxy with two routes:
- `POST /api/auth` — exchange OAuth code for token
- `POST /api/revoke` — revoke OAuth authorization

Uses separate client ID/secret pairs for dev vs production (env vars with `_DEV` suffix).

## Key Tech

- Vue 3 + VitePress 1.6.x
- markdown-it-mathjax3 for math rendering
- markdown-it-container for bilingual blocks
- Giscus for chapter-level comments
- Cloudflare Workers for serverless OAuth proxy

## Conventions

- Always respond in Chinese-simplified (per user preference).
- Chapter files follow naming: `docs/chapters/ch{NN}-{slug}.md`
- Base path is `/rl-book-bilingual/` (GitHub Pages deployment)
