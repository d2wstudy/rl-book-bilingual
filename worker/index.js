// Cloudflare Worker: GitHub OAuth proxy + Discussions read proxy with shared cache
// Deploy: npx wrangler deploy
//
// Environment variables (set via wrangler secret):
//   GITHUB_CLIENT_ID        — production OAuth App client ID
//   GITHUB_CLIENT_SECRET    — production OAuth App client secret
//   GITHUB_CLIENT_ID_DEV    — (optional) dev OAuth App client ID
//   GITHUB_CLIENT_SECRET_DEV — (optional) dev OAuth App client secret
//   GITHUB_PAT              — fine-grained PAT with Discussions read-only access

const REPO_OWNER = 'd2wstudy'
const REPO_NAME = 'rl-book-bilingual'
const CACHE_TTL = 300 // 5 minutes (shared cache)
const USER_REACTION_TTL = 604800 // 7 days (per-user reaction cache, invalidated on toggle)

const COMMENT_FIELDS = `
  id body createdAt author { login avatarUrl }
  reactionGroups { content viewerHasReacted reactors { totalCount } }
  replies(first: 50) {
    nodes {
      id body createdAt author { login avatarUrl }
      reactionGroups { content viewerHasReacted reactors { totalCount } }
    }
  }
`

export default {
  async fetch(request, env) {
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    }

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders })
    }

    const url = new URL(request.url)

    // GET /api/discussions?path=xxx&category=Notes[&id=xxx]
    if (url.pathname === '/api/discussions' && request.method === 'GET') {
      return handleDiscussions(request, url, env, corsHeaders)
    }

    // POST /api/cache/purge?path=xxx&category=Notes
    if (url.pathname === '/api/cache/purge' && request.method === 'POST') {
      return handleCachePurge(request, url, corsHeaders)
    }

    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405, headers: corsHeaders })
    }

    // POST /api/auth — exchange code for token
    if (url.pathname === '/api/auth') {
      const { code, client_id, redirect_uri } = await request.json()
      if (!code) {
        return Response.json({ error: 'Missing code' }, { status: 400, headers: corsHeaders })
      }

      let id = env.GITHUB_CLIENT_ID
      let secret = env.GITHUB_CLIENT_SECRET
      if (client_id && client_id === env.GITHUB_CLIENT_ID_DEV) {
        id = env.GITHUB_CLIENT_ID_DEV
        secret = env.GITHUB_CLIENT_SECRET_DEV
      }

      const resp = await fetch('https://github.com/login/oauth/access_token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({
          client_id: id,
          client_secret: secret,
          code,
          redirect_uri,
        }),
      })

      const data = await resp.json()
      return Response.json(data, { headers: corsHeaders })
    }

    // POST /api/revoke — revoke OAuth grant
    if (url.pathname === '/api/revoke') {
      const { access_token, client_id } = await request.json()
      if (!access_token) {
        return Response.json({ error: 'Missing access_token' }, { status: 400, headers: corsHeaders })
      }

      let id = env.GITHUB_CLIENT_ID
      let secret = env.GITHUB_CLIENT_SECRET
      if (client_id && client_id === env.GITHUB_CLIENT_ID_DEV) {
        id = env.GITHUB_CLIENT_ID_DEV
        secret = env.GITHUB_CLIENT_SECRET_DEV
      }

      const revokeResp = await fetch(`https://api.github.com/applications/${id}/grant`, {
        method: 'DELETE',
        headers: {
          Authorization: 'Basic ' + btoa(`${id}:${secret}`),
          Accept: 'application/json',
          'Content-Type': 'application/json',
          'User-Agent': 'rl-book-worker',
        },
        body: JSON.stringify({ access_token }),
      })

      // 204 = success, 422 = already revoked — both are fine
      if (revokeResp.status === 204 || revokeResp.status === 422) {
        return Response.json({ ok: true }, { headers: corsHeaders })
      }
      return Response.json(
        { ok: false, status: revokeResp.status },
        { status: revokeResp.status, headers: corsHeaders },
      )
    }

    return new Response('Not found', { status: 404, headers: corsHeaders })
  },
}

/** Build a normalized shared cache key (path + category only) */
function buildCacheKey(url) {
  const pagePath = url.searchParams.get('path')
  const categoryName = url.searchParams.get('category')
  const normalized = new URL(url.origin + '/api/discussions')
  normalized.searchParams.set('path', pagePath)
  normalized.searchParams.set('category', categoryName)
  return new Request(normalized.toString(), { method: 'GET' })
}

/** Build a per-user reaction cache key (path + category + token hash) */
function buildUserCacheKey(url, tokenHash) {
  const pagePath = url.searchParams.get('path')
  const categoryName = url.searchParams.get('category')
  const normalized = new URL(url.origin + '/api/reactions')
  normalized.searchParams.set('path', pagePath)
  normalized.searchParams.set('category', categoryName)
  normalized.searchParams.set('u', tokenHash)
  return new Request(normalized.toString(), { method: 'GET' })
}

/** SHA-256 hash of token (first 16 hex chars) for per-user cache key */
async function hashToken(token) {
  const data = new TextEncoder().encode(token)
  const hash = await crypto.subtle.digest('SHA-256', data)
  return Array.from(new Uint8Array(hash).slice(0, 8))
    .map(b => b.toString(16).padStart(2, '0')).join('')
}

/** Extract viewerHasReacted map from comments: { subjectId: { CONTENT: true } } */
function extractReactions(comments) {
  const map = {}
  for (const c of comments) {
    for (const g of c.reactionGroups || []) {
      if (g.viewerHasReacted) {
        if (!map[c.id]) map[c.id] = {}
        map[c.id][g.content] = true
      }
    }
    for (const r of c.replies?.nodes || []) {
      for (const g of r.reactionGroups || []) {
        if (g.viewerHasReacted) {
          if (!map[r.id]) map[r.id] = {}
          map[r.id][g.content] = true
        }
      }
    }
  }
  return map
}

/** Overlay per-user reaction state onto shared cache comments (mutates in place) */
function overlayReactions(comments, reactionMap) {
  for (const c of comments) {
    const cr = reactionMap[c.id] || {}
    for (const g of c.reactionGroups || []) {
      g.viewerHasReacted = !!cr[g.content]
    }
    for (const r of c.replies?.nodes || []) {
      const rr = reactionMap[r.id] || {}
      for (const g of r.reactionGroups || []) {
        g.viewerHasReacted = !!rr[g.content]
      }
    }
  }
}

/** Fetch discussion from GitHub (reusable for both shared and per-user queries) */
async function fetchDiscussion(token, pagePath, categoryName, knownId) {
  if (knownId) {
    const data = await githubGql(token, `query($id: ID!) {
      node(id: $id) {
        ... on Discussion {
          id
          comments(first: 100) { nodes { ${COMMENT_FIELDS} } }
        }
      }
    }`, { id: knownId })
    const node = data?.node
    return node
      ? { discussionId: node.id, comments: node.comments.nodes }
      : { discussionId: null, comments: [] }
  }
  const searchQuery = `repo:${REPO_OWNER}/${REPO_NAME} in:title ${JSON.stringify(pagePath)} category:${JSON.stringify(categoryName)}`
  const data = await githubGql(token, `query($q: String!) {
    search(query: $q, type: DISCUSSION, first: 3) {
      nodes {
        ... on Discussion {
          id title
          comments(first: 100) { nodes { ${COMMENT_FIELDS} } }
        }
      }
    }
  }`, { q: searchQuery })
  const nodes = data?.search?.nodes || []
  const match = nodes.find(n => n.title === pagePath)
  return match
    ? { discussionId: match.id, comments: match.comments.nodes }
    : { discussionId: null, comments: [] }
}

async function handleDiscussions(request, url, env, corsHeaders) {
  const pagePath = url.searchParams.get('path')
  const categoryName = url.searchParams.get('category')
  const knownId = url.searchParams.get('id')
  if (!pagePath || !categoryName) {
    return Response.json({ error: 'Missing path or category' }, { status: 400, headers: corsHeaders })
  }

  const authHeader = request.headers.get('Authorization')
  const userToken = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null

  const cache = caches.default
  const cacheKey = buildCacheKey(url)
  const cached = await cache.match(cacheKey)

  let result
  let fetchedWithUserToken = false

  if (cached) {
    console.log(`[CACHE HIT] ${categoryName} | ${pagePath}`)
    result = await cached.json()
  } else if (userToken) {
    // Authenticated user populates shared cache (implicit token pool)
    console.log(`[CACHE MISS] ${categoryName} | ${pagePath} — fetching with user token`)
    result = await fetchDiscussion(userToken, pagePath, categoryName, knownId)
    fetchedWithUserToken = true
    console.log(`[CACHE MISS] ${categoryName} | ${pagePath} — found: ${!!result.discussionId}`)

    // Populate shared cache (strip viewerHasReacted before caching)
    const sharedResult = JSON.parse(JSON.stringify(result))
    if (sharedResult.comments?.length) overlayReactions(sharedResult.comments, {})
    const sharedResp = Response.json(sharedResult, {
      headers: { 'Cache-Control': `public, max-age=${CACHE_TTL}` },
    })
    await cache.put(cacheKey, sharedResp)
  } else {
    // Unauthenticated user, cache miss — return empty, next auth user will populate
    console.log(`[CACHE MISS] ${categoryName} | ${pagePath} — no token, returning empty`)
    result = { discussionId: null, comments: [] }
  }

  // Per-user reaction overlay for authenticated users;
  // for unauthenticated users, strip leaked viewerHasReacted from shared cache
  if (!userToken && result.comments?.length) {
    overlayReactions(result.comments, {})
  } else if (userToken && result.comments?.length) {
    const tHash = await hashToken(userToken)
    const userCacheKey = buildUserCacheKey(url, tHash)
    const userCached = await cache.match(userCacheKey)

    if (userCached) {
      // Per-user cache hit — overlay onto shared result
      console.log(`[USER CACHE HIT] ${categoryName} | ${pagePath}`)
      overlayReactions(result.comments, await userCached.json())
    } else if (fetchedWithUserToken) {
      // Just fetched with this user's token — reactions already correct, cache them
      const reactionMap = extractReactions(result.comments)
      await cache.put(userCacheKey, Response.json(reactionMap, {
        headers: { 'Cache-Control': `public, max-age=${USER_REACTION_TTL}` },
      }))
      console.log(`[USER CACHE SET] ${categoryName} | ${pagePath}`)
    } else {
      // Shared cache hit but no per-user data — fetch with user's token (~51 pts)
      const discId = result.discussionId
      if (discId) {
        console.log(`[USER CACHE MISS] ${categoryName} | ${pagePath} — fetching reactions`)
        const userData = await fetchDiscussion(userToken, pagePath, categoryName, discId)
        const reactionMap = extractReactions(userData.comments || [])
        overlayReactions(result.comments, reactionMap)
        await cache.put(userCacheKey, Response.json(reactionMap, {
          headers: { 'Cache-Control': `public, max-age=${USER_REACTION_TTL}` },
        }))
        console.log(`[USER CACHE SET] ${categoryName} | ${pagePath}`)
      }
    }
  }

  return Response.json(result, {
    headers: { ...corsHeaders, 'X-Cache': cached ? 'HIT' : 'MISS' },
  })
}

async function handleCachePurge(request, url, corsHeaders) {
  const pagePath = url.searchParams.get('path')
  const categoryName = url.searchParams.get('category')
  const userOnly = url.searchParams.get('user_only') === '1'
  if (!pagePath || !categoryName) {
    return Response.json({ error: 'Missing path or category' }, { status: 400, headers: corsHeaders })
  }

  const cache = caches.default
  let deleted = false

  // Only purge shared cache for structural changes (new comments/replies), not reactions
  if (!userOnly) {
    const cacheKey = buildCacheKey(url)
    deleted = await cache.delete(cacheKey)
    console.log(`[CACHE PURGE] ${categoryName} | ${pagePath} — shared: ${deleted}`)
  }

  // Always purge per-user reaction cache if token provided
  const authHeader = request.headers.get('Authorization')
  const userToken = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null
  let userDeleted = false
  if (userToken) {
    const tHash = await hashToken(userToken)
    const userCacheKey = buildUserCacheKey(url, tHash)
    userDeleted = await cache.delete(userCacheKey)
    console.log(`[CACHE PURGE] ${categoryName} | ${pagePath} — user: ${userDeleted}`)
  }

  return Response.json({ ok: true, deleted, userDeleted }, { headers: corsHeaders })
}

async function githubGql(token, query, variables) {
  const resp = await fetch('https://api.github.com/graphql', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      'User-Agent': 'rl-book-worker',
    },
    body: JSON.stringify({ query, variables }),
  })
  const json = await resp.json()
  return json.data
}
