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
const CACHE_TTL = 300 // 5 minutes

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
      return handleCachePurge(url, corsHeaders)
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

/** Build a normalized cache key using only path + category (ignoring id param and auth) */
function buildCacheKey(url) {
  const pagePath = url.searchParams.get('path')
  const categoryName = url.searchParams.get('category')
  const normalized = new URL(url.origin + '/api/discussions')
  normalized.searchParams.set('path', pagePath)
  normalized.searchParams.set('category', categoryName)
  return new Request(normalized.toString(), { method: 'GET' })
}

async function handleDiscussions(request, url, env, corsHeaders) {
  const pagePath = url.searchParams.get('path')
  const categoryName = url.searchParams.get('category')
  const knownId = url.searchParams.get('id') // optional: cheaper node() query
  if (!pagePath || !categoryName) {
    return Response.json({ error: 'Missing path or category' }, { status: 400, headers: corsHeaders })
  }

  // Use user's token if provided, fall back to PAT
  const authHeader = request.headers.get('Authorization')
  const token = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null
  const effectiveToken = token || env.GITHUB_PAT
  if (!effectiveToken) {
    return Response.json({ error: 'Server not configured' }, { status: 500, headers: corsHeaders })
  }

  // Check cache (normalized key: path + category only)
  const cache = caches.default
  const cacheKey = buildCacheKey(url)
  const cached = await cache.match(cacheKey)
  if (cached) {
    console.log(`[CACHE HIT] ${categoryName} | ${pagePath}`)
    const resp = new Response(cached.body, cached)
    resp.headers.set('Access-Control-Allow-Origin', '*')
    resp.headers.set('X-Cache', 'HIT')
    return resp
  }

  console.log(`[CACHE MISS] ${categoryName} | ${pagePath} — fetching from GitHub`)

  let result
  if (knownId) {
    // Cheaper node() query when discussion ID is known (~51 points vs ~153)
    const data = await githubGql(effectiveToken, `query($id: ID!) {
      node(id: $id) {
        ... on Discussion {
          id
          comments(first: 100) { nodes { ${COMMENT_FIELDS} } }
        }
      }
    }`, { id: knownId })
    const node = data?.node
    result = node
      ? { discussionId: node.id, comments: node.comments.nodes }
      : { discussionId: null, comments: [] }
  } else {
    // Search query when discussion ID is unknown
    const searchQuery = `repo:${REPO_OWNER}/${REPO_NAME} in:title ${JSON.stringify(pagePath)} category:${JSON.stringify(categoryName)}`
    const data = await githubGql(effectiveToken, `query($q: String!) {
      search(query: $q, type: DISCUSSION, first: 3) {
        nodes {
          ... on Discussion {
            id
            title
            comments(first: 100) { nodes { ${COMMENT_FIELDS} } }
          }
        }
      }
    }`, { q: searchQuery })
    const nodes = data?.search?.nodes || []
    const match = nodes.find(n => n.title === pagePath)
    result = match
      ? { discussionId: match.id, comments: match.comments.nodes }
      : { discussionId: null, comments: [] }
  }

  console.log(`[CACHE MISS] ${categoryName} | ${pagePath} — found: ${!!result.discussionId}`)

  // Cache the response
  const response = Response.json(result, {
    headers: {
      ...corsHeaders,
      'Cache-Control': `public, max-age=${CACHE_TTL}`,
      'X-Cache': 'MISS',
    },
  })
  await cache.put(cacheKey, response.clone())
  return response
}

async function handleCachePurge(url, corsHeaders) {
  const pagePath = url.searchParams.get('path')
  const categoryName = url.searchParams.get('category')
  if (!pagePath || !categoryName) {
    return Response.json({ error: 'Missing path or category' }, { status: 400, headers: corsHeaders })
  }

  const cache = caches.default
  const cacheKey = buildCacheKey(url)
  const deleted = await cache.delete(cacheKey)
  console.log(`[CACHE PURGE] ${categoryName} | ${pagePath} — deleted: ${deleted}`)
  return Response.json({ ok: true, deleted }, { headers: corsHeaders })
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
