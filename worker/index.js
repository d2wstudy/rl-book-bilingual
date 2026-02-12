// Cloudflare Worker: GitHub OAuth proxy + Discussions read proxy
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

export default {
  async fetch(request, env) {
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    }

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders })
    }

    const url = new URL(request.url)

    // GET /api/discussions?path=xxx&category=Notes
    if (url.pathname === '/api/discussions' && request.method === 'GET') {
      return handleDiscussions(url, env, corsHeaders)
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

async function handleDiscussions(url, env, corsHeaders) {
  const pagePath = url.searchParams.get('path')
  const categoryName = url.searchParams.get('category')
  if (!pagePath || !categoryName) {
    return Response.json({ error: 'Missing path or category' }, { status: 400, headers: corsHeaders })
  }

  if (!env.GITHUB_PAT) {
    return Response.json({ error: 'Server not configured' }, { status: 500, headers: corsHeaders })
  }

  // Check cache
  const cache = caches.default
  const cacheKey = new Request(url.toString(), { method: 'GET' })
  const cached = await cache.match(cacheKey)
  if (cached) {
    const resp = new Response(cached.body, cached)
    resp.headers.set('Access-Control-Allow-Origin', '*')
    return resp
  }

  // Fetch categories to get category ID
  const catData = await githubGql(env.GITHUB_PAT, `query($owner: String!, $name: String!) {
    repository(owner: $owner, name: $name) {
      discussionCategories(first: 20) { nodes { id name } }
    }
  }`, { owner: REPO_OWNER, name: REPO_NAME })

  const categories = catData?.repository?.discussionCategories?.nodes || []
  const category = categories.find(c => c.name === categoryName)
  if (!category) {
    return Response.json({ discussionId: null, comments: [] }, { headers: corsHeaders })
  }

  // Fetch discussions with comments
  const data = await githubGql(env.GITHUB_PAT, `query($owner: String!, $name: String!, $categoryId: ID) {
    repository(owner: $owner, name: $name) {
      discussions(first: 50, categoryId: $categoryId, orderBy: {field: CREATED_AT, direction: DESC}) {
        nodes {
          id
          title
          comments(first: 100) {
            nodes {
              id body createdAt author { login avatarUrl }
              replies(first: 50) {
                nodes { id body createdAt author { login avatarUrl } }
              }
            }
          }
        }
      }
    }
  }`, { owner: REPO_OWNER, name: REPO_NAME, categoryId: category.id })

  const nodes = data?.repository?.discussions?.nodes || []
  const match = nodes.find(n => n.title === pagePath)

  const result = match
    ? { discussionId: match.id, comments: match.comments.nodes }
    : { discussionId: null, comments: [] }

  // Cache the response
  const response = Response.json(result, {
    headers: {
      ...corsHeaders,
      'Cache-Control': `public, max-age=${CACHE_TTL}`,
    },
  })
  await cache.put(cacheKey, response.clone())
  return response
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
