import { useAuth } from './useAuth'

const REPO_OWNER = 'd2wstudy'
const REPO_NAME = 'rl-book-bilingual'
const GRAPHQL_URL = 'https://api.github.com/graphql'
const WORKER_URL = 'https://rl-book-auth.d2w.workers.dev'

// --- Deduplication layer (no TTL cache — always fetch fresh data) ---
const _inflightDiscussions = new Map<string, Promise<any>>()

// --- Microtask batch queue (combines multiple category fetches into one GraphQL call) ---
interface BatchRequest {
  pagePath: string
  categoryName: string
  knownId: string | null
  resolve: (result: { discussionId: string | null; comments: any[] } | null) => void
  reject: (err: any) => void
}
let _batchQueue: BatchRequest[] = []
let _batchScheduled = false

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

function scheduleBatch() {
  if (!_batchScheduled) {
    _batchScheduled = true
    Promise.resolve().then(processBatch)
  }
}

async function processBatch() {
  const queue = _batchQueue
  _batchQueue = []
  _batchScheduled = false
  if (queue.length === 0) return

  // Single request — skip dynamic query building
  if (queue.length === 1) {
    const req = queue[0]
    try {
      const result = req.knownId
        ? await fetchDiscussionById(req.knownId)
        : await fetchDirectFromGitHub(req.pagePath, req.categoryName)
      req.resolve(result)
    } catch (err) { req.reject(err) }
    return
  }

  // Build a single batched GraphQL query with aliases
  const varDefs: string[] = []
  const queryParts: string[] = []
  const variables: Record<string, any> = {}

  for (let i = 0; i < queue.length; i++) {
    const req = queue[i]
    if (req.knownId) {
      varDefs.push(`$id${i}: ID!`)
      variables[`id${i}`] = req.knownId
      queryParts.push(`d${i}: node(id: $id${i}) { ... on Discussion { id comments(first: 100) { nodes { ${COMMENT_FIELDS} } } } }`)
    } else {
      const searchQuery = `repo:${REPO_OWNER}/${REPO_NAME} in:title ${JSON.stringify(req.pagePath)} category:${JSON.stringify(req.categoryName)}`
      varDefs.push(`$q${i}: String!`)
      variables[`q${i}`] = searchQuery
      queryParts.push(`d${i}: search(query: $q${i}, type: DISCUSSION, first: 3) { nodes { ... on Discussion { id title comments(first: 100) { nodes { ${COMMENT_FIELDS} } } } } }`)
    }
  }

  try {
    const data = await gql(`query(${varDefs.join(', ')}) {\n${queryParts.join('\n')}\n}`, variables)

    for (let i = 0; i < queue.length; i++) {
      const req = queue[i]
      const raw = data?.[`d${i}`]
      if (req.knownId) {
        req.resolve(raw ? { discussionId: raw.id, comments: raw.comments.nodes } : null)
      } else {
        const nodes = raw?.nodes || []
        const match = nodes.find((n: any) => n.title === req.pagePath)
        req.resolve(match
          ? { discussionId: match.id, comments: match.comments.nodes }
          : { discussionId: null, comments: [] })
      }
    }
  } catch (err) {
    for (const req of queue) req.reject(err)
  }
}

// Shared category ID cache (Promise-based to deduplicate concurrent calls)
let _categoriesPromise: Promise<Map<string, string>> | null = null

async function fetchCategories(): Promise<Map<string, string>> {
  const data = await gql(`query($owner: String!, $name: String!) {
    repository(owner: $owner, name: $name) {
      discussionCategories(first: 20) { nodes { id name } }
    }
  }`, { owner: REPO_OWNER, name: REPO_NAME })
  const map = new Map<string, string>()
  for (const c of data?.repository?.discussionCategories?.nodes || []) {
    map.set(c.name, c.id)
  }
  return map
}

export async function getCategoryId(categoryName: string): Promise<string | null> {
  const { token } = useAuth()
  if (!token.value) return null

  if (!_categoriesPromise) {
    _categoriesPromise = fetchCategories()
  }
  const categories = await _categoriesPromise
  return categories.get(categoryName) || null
}

/** Find a discussion by title in a category AND fetch its comments.
 *  Authenticated users: batched via microtask queue (multiple categories merged into one API call).
 *  Unauthenticated users: fetched via Worker proxy. */
export async function findDiscussionWithComments(
  pagePath: string,
  categoryName: string,
  knownDiscussionId?: string | null,
): Promise<{ discussionId: string | null; comments: any[] } | null> {
  const key = `${categoryName}::${pagePath}`

  // Deduplicate concurrent in-flight requests for the same category+page
  const inflight = _inflightDiscussions.get(key)
  if (inflight) return inflight

  const { token } = useAuth()

  let promise: Promise<{ discussionId: string | null; comments: any[] } | null>

  if (!token.value) {
    // Unauthenticated: use Worker proxy
    promise = fetchViaProxy(pagePath, categoryName)
  } else {
    // Authenticated: enqueue for microtask batching
    promise = new Promise((resolve, reject) => {
      _batchQueue.push({
        pagePath,
        categoryName,
        knownId: knownDiscussionId ?? null,
        resolve,
        reject,
      })
      scheduleBatch()
    })
  }

  _inflightDiscussions.set(key, promise)
  promise.finally(() => _inflightDiscussions.delete(key))
  return promise
}

/** Fetch a known discussion by its node ID (cheaper than search) */
async function fetchDiscussionById(
  discussionId: string,
): Promise<{ discussionId: string; comments: any[] } | null> {
  const data = await gql(`query($id: ID!) {
    node(id: $id) {
      ... on Discussion {
        id
        comments(first: 100) {
          nodes { ${COMMENT_FIELDS} }
        }
      }
    }
  }`, { id: discussionId })

  const node = data?.node
  return node
    ? { discussionId: node.id, comments: node.comments.nodes }
    : null
}

/** Direct GitHub GraphQL fetch using the user's own token */
async function fetchDirectFromGitHub(
  pagePath: string,
  categoryName: string,
): Promise<{ discussionId: string | null; comments: any[] } | null> {
  const searchQuery = `repo:${REPO_OWNER}/${REPO_NAME} in:title ${JSON.stringify(pagePath)} category:${JSON.stringify(categoryName)}`
  const data = await gql(`query($q: String!) {
    search(query: $q, type: DISCUSSION, first: 3) {
      nodes {
        ... on Discussion {
          id
          title
          comments(first: 100) {
            nodes { ${COMMENT_FIELDS} }
          }
        }
      }
    }
  }`, { q: searchQuery })

  const nodes = data?.search?.nodes || []
  const match = nodes.find((n: any) => n.title === pagePath)
  return match
    ? { discussionId: match.id, comments: match.comments.nodes }
    : { discussionId: null, comments: [] }
}

/** Read-only proxy via Cloudflare Worker (for unauthenticated users) */
async function fetchViaProxy(
  pagePath: string,
  categoryName: string,
): Promise<{ discussionId: string | null; comments: any[] } | null> {
  try {
    const params = new URLSearchParams({ path: pagePath, category: categoryName })
    const resp = await fetch(`${WORKER_URL}/api/discussions?${params}`)
    if (!resp.ok) return null
    return await resp.json()
  } catch {
    console.warn('[GQL] Worker proxy request failed')
    return null
  }
}

/** Create a new discussion in a category */
export async function createDiscussion(pagePath: string, categoryName: string, bodyText: string): Promise<string | null> {
  const categoryId = await getCategoryId(categoryName)
  if (!categoryId) return null

  const repoData = await gql(`query($owner: String!, $name: String!) {
    repository(owner: $owner, name: $name) { id }
  }`, { owner: REPO_OWNER, name: REPO_NAME })
  const repoId = repoData?.repository?.id
  if (!repoId) return null

  const result = await gql(`mutation($repoId: ID!, $categoryId: ID!, $title: String!, $body: String!) {
    createDiscussion(input: { repositoryId: $repoId, categoryId: $categoryId, title: $title, body: $body }) {
      discussion { id }
    }
  }`, { repoId, categoryId, title: pagePath, body: bodyText })
  return result?.createDiscussion?.discussion?.id || null
}

/** Add a comment to a discussion — returns the new comment data */
export async function addDiscussionComment(discussionId: string, body: string) {
  const data = await gql(`mutation($discussionId: ID!, $body: String!) {
    addDiscussionComment(input: { discussionId: $discussionId, body: $body }) {
      comment {
        id body createdAt
        author { login avatarUrl }
        reactionGroups { content viewerHasReacted reactors { totalCount } }
      }
    }
  }`, { discussionId, body })
  return data?.addDiscussionComment?.comment || null
}

/** Add a reaction to a subject (comment or reply) */
export async function addReaction(subjectId: string, content: string) {
  return gql(`mutation($subjectId: ID!, $content: ReactionContent!) {
    addReaction(input: { subjectId: $subjectId, content: $content }) {
      reaction { content }
    }
  }`, { subjectId, content })
}

/** Remove a reaction from a subject */
export async function removeReaction(subjectId: string, content: string) {
  return gql(`mutation($subjectId: ID!, $content: ReactionContent!) {
    removeReaction(input: { subjectId: $subjectId, content: $content }) {
      reaction { content }
    }
  }`, { subjectId, content })
}

/** Add a reply to a discussion comment — returns the new reply data */
export async function addDiscussionReply(discussionId: string, replyToId: string, body: string) {
  const data = await gql(`mutation($discussionId: ID!, $replyToId: ID!, $body: String!) {
    addDiscussionComment(input: { discussionId: $discussionId, replyToId: $replyToId, body: $body }) {
      comment {
        id body createdAt
        author { login avatarUrl }
        reactionGroups { content viewerHasReacted reactors { totalCount } }
      }
    }
  }`, { discussionId, replyToId, body })
  return data?.addDiscussionComment?.comment || null
}

/** Shared GraphQL helper (requires auth) */
export async function gql(query: string, variables: Record<string, any>) {
  const { token } = useAuth()
  const t = token.value
  if (!t) return null

  // Dev: inject rateLimit field to monitor API point consumption
  const actualQuery = import.meta.env.DEV
    ? query.replace(/\{/, '{ rateLimit { cost remaining resetAt }')
    : query

  const resp = await fetch(GRAPHQL_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${t}`,
    },
    body: JSON.stringify({ query: actualQuery, variables }),
  })
  if (resp.status === 403 || resp.status === 429) {
    console.warn('[GQL] GitHub API rate limit exceeded')
    return null
  }
  const json = await resp.json()
  if (json.errors) {
    console.warn('[GQL] GraphQL errors:', json.errors)
  }
  if (import.meta.env.DEV && json.data?.rateLimit) {
    const rl = json.data.rateLimit
    console.log(`[GQL] cost=${rl.cost} remaining=${rl.remaining} reset=${rl.resetAt}`)
  }
  return json.data
}
