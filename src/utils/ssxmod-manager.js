/**
 * SSXMOD Cookie Manager — per-account cache with TTL.
 * Eliminates global cookie correlation by keying on account email.
 * Lazy regeneration on cache miss; no global timer.
 */

const { generateCookies } = require('./cookie-generator');
const { logger } = require('./logger');

// Per-account cookie cache: Map<email, {ssxmod_itna, ssxmod_itna2, timestamp}>
const accountCache = new Map();

// Cache TTL: 15 minutes (matches previous refresh interval)
const CACHE_TTL_MS = 15 * 60 * 1000;

// Maximum number of cached accounts to prevent unbounded memory growth
const MAX_CACHE_SIZE = 100;

/**
 * Get or lazily generate SSXMOD cookies for a specific account.
 * @param {Object} account - Account object (must have .email)
 * @returns {{ssxmod_itna: string, ssxmod_itna2: string}}
 */
function getSsxmodForAccount(account) {
    const email = account && account.email;
    if (!email) {
        // Fallback for calls without account context — use a synthetic key
        return getOrGenerate('__global__');
    }
    return getOrGenerate(email);
}

/**
 * Internal: get from cache or generate fresh cookies.
 * @param {string} key
 * @returns {{ssxmod_itna: string, ssxmod_itna2: string}}
 */
function evictOldestSsxmod() {
    if (accountCache.size <= MAX_CACHE_SIZE) return;
    const oldestKey = accountCache.keys().next().value;
    if (oldestKey !== undefined) {
        accountCache.delete(oldestKey);
    }
}

function getOrGenerate(key) {
    const cached = accountCache.get(key);
    const now = Date.now();
    if (cached && (now - cached.timestamp) < CACHE_TTL_MS) {
        return { ssxmod_itna: cached.ssxmod_itna, ssxmod_itna2: cached.ssxmod_itna2 };
    }
    try {
        const result = generateCookies();
        accountCache.set(key, {
            ssxmod_itna: result.ssxmod_itna,
            ssxmod_itna2: result.ssxmod_itna2,
            timestamp: now
        });
        evictOldestSsxmod();
        return { ssxmod_itna: result.ssxmod_itna, ssxmod_itna2: result.ssxmod_itna2 };
    } catch (err) {
        logger.error(`SSXMOD generation failed for ${key}: ${err.message}`, 'SSXMOD');
        // Return empty strings rather than crashing the request
        return { ssxmod_itna: '', ssxmod_itna2: '' };
    }
}

// Legacy API compatibility — returns global-fallback values
function getSsxmodItna() {
    return getOrGenerate('__global__').ssxmod_itna;
}

function getSsxmodItna2() {
    return getOrGenerate('__global__').ssxmod_itna2;
}

function getCookies() {
    const g = getOrGenerate('__global__');
    return { ssxmod_itna: g.ssxmod_itna, ssxmod_itna2: g.ssxmod_itna2, timestamp: Date.now() };
}

/**
 * No-op: initialization is now lazy. Kept for backward compatibility.
 */
function initSsxmodManager() {
    // Intentionally empty — lazy init replaces startup side effect
}

function refreshCookies() {
    // Refresh global fallback entry
    getOrGenerate('__global__');
}

function stopRefresh() {
    // No timer to stop in lazy model
}

module.exports = {
    initSsxmodManager,
    getSsxmodForAccount,
    getSsxmodItna,
    getSsxmodItna2,
    getCookies,
    refreshCookies,
    stopRefresh
};
