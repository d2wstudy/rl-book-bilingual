import { ref, readonly } from 'vue'

// ---- Configuration ----
const GITHUB_CLIENT_ID = import.meta.env.VITE_GITHUB_CLIENT_ID || ''
const WORKER_URL = 'https://rl-book-auth.d2w.workers.dev'

interface GitHubUser {
  login: string
  avatar_url: string
  html_url: string
}

const token = ref<string | null>(null)
const user = ref<GitHubUser | null>(null)
const loading = ref(false)
let revokePromise: Promise<void> | null = null

export function useAuth() {
  function init() {
    // Check for OAuth callback code in URL
    const params = new URLSearchParams(window.location.search)
    const code = params.get('code')
    if (code) {
      // Clean URL
      window.history.replaceState({}, '', window.location.pathname + window.location.hash)
      exchangeCode(code)
      return
    }

    // Restore from localStorage (persists across browser sessions)
    const saved = localStorage.getItem('gh-token')
    if (saved) {
      token.value = saved
      loading.value = true
      fetchUser().finally(() => { loading.value = false })
    }
  }

  async function login() {
    // Wait for any pending revoke to complete before starting new OAuth flow,
    // otherwise GitHub may auto-approve with the previous account's grant
    if (revokePromise) {
      await revokePromise
    }
    // Remember current page so we can return after OAuth
    sessionStorage.setItem('gh-redirect', window.location.href)
    // Always use site root as redirect_uri to match OAuth App config
    // dev: http://localhost:5173/  prod: https://xxx.github.io/rl-book-bilingual/
    const callbackUrl = window.location.origin + import.meta.env.BASE_URL
    const url = `https://github.com/login/oauth/authorize?client_id=${GITHUB_CLIENT_ID}&scope=public_repo&redirect_uri=${encodeURIComponent(callbackUrl)}`
    window.location.href = url
  }

  function logout() {
    const savedToken = token.value
    token.value = null
    user.value = null
    localStorage.removeItem('gh-token')
    // Revoke GitHub OAuth grant so next login shows the authorization page,
    // allowing the user to switch accounts
    if (savedToken) {
      revokePromise = fetch(`${WORKER_URL}/api/revoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ access_token: savedToken, client_id: GITHUB_CLIENT_ID }),
      })
        .then(() => { revokePromise = null })
        .catch(() => { revokePromise = null })
    }
  }

  async function exchangeCode(code: string) {
    loading.value = true
    const redirect_uri = window.location.origin + import.meta.env.BASE_URL
    try {
      const resp = await fetch(`${WORKER_URL}/api/auth`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, client_id: GITHUB_CLIENT_ID, redirect_uri }),
      })
      const data = await resp.json()
      if (data.access_token) {
        token.value = data.access_token
        localStorage.setItem('gh-token', data.access_token)
        await fetchUser()
        // Redirect back to the page where user clicked login
        const redirect = sessionStorage.getItem('gh-redirect')
        if (redirect) {
          sessionStorage.removeItem('gh-redirect')
          window.location.href = redirect
        }
      } else {
        console.error('[auth] token exchange failed:', data)
      }
    } catch (e) {
      console.error('[auth] exchangeCode error:', e)
    } finally {
      loading.value = false
    }
  }

  async function fetchUser() {
    if (!token.value) return
    try {
      const resp = await fetch('https://api.github.com/user', {
        headers: { Authorization: `Bearer ${token.value}` },
      })
      if (resp.ok) {
        user.value = await resp.json()
        console.log('[auth] user loaded:', user.value?.login)
      } else if (resp.status === 401) {
        console.warn('[auth] token invalid (401), clearing session')
        logout()
      } else {
        console.warn('[auth] fetchUser failed:', resp.status)
      }
    } catch (e) {
      console.warn('[auth] fetchUser network error:', e)
    }
  }

  return {
    token: readonly(token),
    user: readonly(user),
    loading: readonly(loading),
    init,
    login,
    logout,
  }
}
