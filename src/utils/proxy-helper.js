const config = require('../config/index.js')
const { HttpsProxyAgent } = require('https-proxy-agent')

// Per-account agent cache keyed by `${proxyUrl}::${email}`.
// LRU eviction when cache exceeds MAX_AGENT_CACHE_SIZE.
const proxyAgents = new Map()
const MAX_AGENT_CACHE_SIZE = 50

// Accept http/https/socks5 protocols; regex intentionally loose to catch common typos only
const PROXY_URL_REGEX = /^(https?|socks5):\/\/[^\s]+$/i

/**
 * Validate proxy URL format.
 * Null/undefined/empty are valid (means "no account-level proxy").
 * @param {string|null|undefined} url
 * @returns {boolean}
 */
const isValidProxyUrl = (url) => {
    if (url === null || url === undefined || url === '') return true
    if (typeof url !== 'string') return false
    const trimmed = url.trim()
    if (!trimmed) return true
    return PROXY_URL_REGEX.test(trimmed)
}

/**
 * Resolve the effective proxy URL for an account.
 * Priority: account.proxy > global PROXY_URL > null
 * @param {Object} [account]
 * @returns {string|null}
 */
const resolveProxyUrl = (account) => {
    if (account && typeof account.proxy === 'string' && account.proxy.trim()) {
        return account.proxy.trim()
    }
    return config.proxyUrl || null
}

/**
 * Evict oldest entry from agent cache when over limit.
 * Map iteration order is insertion order, so first key is oldest.
 */
const evictOldestAgent = () => {
    if (proxyAgents.size <= MAX_AGENT_CACHE_SIZE) return
    const oldestKey = proxyAgents.keys().next().value
    if (oldestKey !== undefined) {
        const agent = proxyAgents.get(oldestKey)
        try {
            if (agent && typeof agent.destroy === 'function') agent.destroy()
        } catch (_) {}
        proxyAgents.delete(oldestKey)
    }
}

/**
 * Build cache key from proxy URL and account email.
 * @param {string|null} url
 * @param {Object} [account]
 * @returns {string}
 */
const buildAgentCacheKey = (url, account) => {
    const email = account && account.email ? account.email : '__global__'
    return `${url || ''}::${email}`
}

/**
 * Get or create a proxy agent, keyed by proxy URL + account email.
 * Separate TCP pools per account even when sharing the same proxy.
 * @param {string|null} url
 * @param {Object} [account]
 * @returns {HttpsProxyAgent|undefined}
 */
const getOrCreateAgent = (url, account) => {
    if (!url) return undefined
    const key = buildAgentCacheKey(url, account)
    let agent = proxyAgents.get(key)
    if (!agent) {
        agent = new HttpsProxyAgent(url)
        proxyAgents.set(key, agent)
        evictOldestAgent()
    } else {
        // Move to end (most recently used) by deleting and re-inserting
        proxyAgents.delete(key)
        proxyAgents.set(key, agent)
    }
    return agent
}

/**
 * Get proxy agent for an account.
 * @param {Object} [account] - Account object (optional). Falls back to global PROXY_URL
 * @returns {HttpsProxyAgent|undefined}
 */
const getProxyAgent = (account) => {
    return getOrCreateAgent(resolveProxyUrl(account), account)
}

/**
 * Invalidate cached agent for a specific proxy URL.
 * Called when an account's proxy is changed or removed.
 * @param {string|null} url
 */
const invalidateProxyAgent = (url) => {
    if (!url) return
    // Delete all entries matching this proxy URL (any account)
    for (const [key, agent] of proxyAgents.entries()) {
        if (key.startsWith(`${url}::`)) {
            try {
                if (typeof agent.destroy === 'function') agent.destroy()
            } catch (_) {}
            proxyAgents.delete(key)
        }
    }
}

/**
 * Get Chat API base URL.
 * @returns {string}
 */
const getChatBaseUrl = () => config.qwenChatProxyUrl

/**
 * Get CLI API base URL.
 * @returns {string}
 */
const getCliBaseUrl = () => config.qwenCliProxyUrl

/**
 * Apply proxy settings to axios request config.
 * Note: account as second optional param for backward compatibility.
 * @param {Object} [requestConfig]
 * @param {Object} [account]
 * @returns {Object}
 */
const applyProxyToAxiosConfig = (requestConfig = {}, account) => {
    const proxyAgent = getProxyAgent(account)
    if (proxyAgent) {
        requestConfig.httpsAgent = proxyAgent
        requestConfig.proxy = false
    }
    return requestConfig
}

/**
 * Apply proxy settings to fetch options.
 * @param {Object} [fetchOptions]
 * @param {Object} [account]
 * @returns {Object}
 */
const applyProxyToFetchOptions = (fetchOptions = {}, account) => {
    const proxyAgent = getProxyAgent(account)
    if (proxyAgent) {
        fetchOptions.agent = proxyAgent
    }
    return fetchOptions
}

module.exports = {
    resolveProxyUrl,
    getProxyAgent,
    invalidateProxyAgent,
    getChatBaseUrl,
    getCliBaseUrl,
    applyProxyToAxiosConfig,
    applyProxyToFetchOptions,
    isValidProxyUrl
}
