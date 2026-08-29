const { describe, it } = require('node:test')
const assert = require('node:assert/strict')

// --- Unit under test imports ---
const {
  generateDeterministicFingerprint,
  parseFingerprint
} = require('../src/utils/fingerprint.js')

const {
  buildRequestHeaders,
  buildUserAgent,
  platformToSecChUa,
  LEGACY_HEADERS,
  LEGACY_UA
} = require('../src/utils/header-profile.js')

const {
  getSsxmodForAccount
} = require('../src/utils/ssxmod-manager.js')

const { jitter } = require('../src/utils/tools.js')

// ---------------------------------------------------------------------------
// AC1: Deterministic fingerprint generation & persistence shape
// ---------------------------------------------------------------------------
describe('AC1 - deterministic fingerprint from email', () => {
  it('produces identical fingerprint for same email', () => {
    const a = generateDeterministicFingerprint('alice@example.com')
    const b = generateDeterministicFingerprint('alice@example.com')
    assert.strictEqual(a.deviceId, b.deviceId)
    assert.strictEqual(a.platform, b.platform)
    assert.strictEqual(a.language, b.language)
    assert.strictEqual(a.hash, b.hash)
    assert.strictEqual(a.raw, b.raw)
  })

  it('produces different fingerprints for different emails', () => {
    const a = generateDeterministicFingerprint('alice@example.com')
    const b = generateDeterministicFingerprint('bob@example.com')
    assert.notStrictEqual(a.deviceId, b.deviceId)
  })

  it('returns required fields for header derivation', () => {
    const fp = generateDeterministicFingerprint('test@example.com')
    assert.ok(fp.deviceId && fp.deviceId.length === 20)
    assert.ok(fp.platform)
    assert.ok(fp.language)
    assert.ok(fp.timezoneOffset !== undefined)
    assert.ok(fp.webglRenderer)
    assert.ok(fp.vendor)
    assert.ok(fp.screenInfo)
    assert.ok(fp.hash && fp.hash.length === 32)
    assert.ok(fp.raw)
  })
})

// ---------------------------------------------------------------------------
// AC2: Header consistency — same account produces identical headers
// ---------------------------------------------------------------------------
describe('AC2 - header consistency for same account', () => {
  it('buildRequestHeaders returns identical UA/platform/sec-ch-ua for same account', () => {
    const account = {
      email: 'consistency@example.com',
      fingerprint: generateDeterministicFingerprint('consistency@example.com')
    }
    const opts = { chatBaseUrl: 'https://chat.qwen.ai', token: 'tok', ssxmodItna: 'itna1', ssxmodItna2: 'itna2' }
    const h1 = buildRequestHeaders(account, opts)
    const h2 = buildRequestHeaders(account, opts)
    assert.strictEqual(h1['user-agent'], h2['user-agent'])
    assert.strictEqual(h1['sec-ch-ua-platform'], h2['sec-ch-ua-platform'])
    assert.strictEqual(h1['sec-ch-ua'], h2['sec-ch-ua'])
  })

  it('platform in headers matches fingerprint platform', () => {
    const account = {
      email: 'plat@example.com',
      fingerprint: generateDeterministicFingerprint('plat@example.com')
    }
    const h = buildRequestHeaders(account, {})
    const expectedPlatform = platformToSecChUa(account.fingerprint.platform)
    assert.strictEqual(h['sec-ch-ua-platform'], expectedPlatform)
    // UA must also be consistent with platform
    if (account.fingerprint.platform === 'MacIntel') {
      assert.ok(h['user-agent'].includes('Macintosh'))
    } else if (account.fingerprint.platform === 'Win32') {
      assert.ok(h['user-agent'].includes('Windows'))
    } else if (account.fingerprint.platform.startsWith('Linux')) {
      assert.ok(h['user-agent'].includes('Linux'))
    }
  })
})

// ---------------------------------------------------------------------------
// AC3: Diversity — 5 distinct accounts produce distinct UAs
// ---------------------------------------------------------------------------
describe('AC3 - header diversity across accounts', () => {
  it('no two of 5 accounts share identical User-Agent', () => {
    const emails = [
      'u1@example.com', 'u2@example.com', 'u3@example.com',
      'u4@example.com', 'u5@example.com'
    ]
    const uas = emails.map(email => {
      const fp = generateDeterministicFingerprint(email)
      return buildUserAgent(fp)
    })
    const unique = new Set(uas)
    assert.strictEqual(unique.size, uas.length, `Expected ${uas.length} unique UAs, got ${unique.size}: ${[...unique].join(' | ')}`)
  })
})

// ---------------------------------------------------------------------------
// AC4: Per-account SSXMOD cookies differ
// ---------------------------------------------------------------------------
describe('AC4 - per-account SSXMOD isolation', () => {
  it('two concurrent accounts get different ssxmod_itna values', () => {
    const a = { email: 'ssx-a@example.com' }
    const b = { email: 'ssx-b@example.com' }
    const ca = getSsxmodForAccount(a)
    const cb = getSsxmodForAccount(b)
    assert.ok(ca.ssxmod_itna, 'account A should have ssxmod_itna')
    assert.ok(cb.ssxmod_itna, 'account B should have ssxmod_itna')
    assert.notStrictEqual(ca.ssxmod_itna, cb.ssxmod_itna)
  })
})

// ---------------------------------------------------------------------------
// AC6: Jitter varies ±25% around base
// ---------------------------------------------------------------------------
describe('AC6 - timing jitter', () => {
  it('jitter(400, 0.25) stays within [300, 500] over 200 samples', () => {
    const base = 400
    const pct = 0.25
    const lo = base * (1 - pct)
    const hi = base * (1 + pct)
    for (let i = 0; i < 200; i++) {
      const v = jitter(base, pct)
      assert.ok(v >= lo && v <= hi, `jitter value ${v} out of range [${lo}, ${hi}]`)
    }
    // Also verify we actually see variation (not always exactly base)
    const samples = Array.from({ length: 50 }, () => jitter(base, pct))
    const distinct = new Set(samples).size
    assert.ok(distinct > 1, 'jitter should produce varying values')
  })
})

// ---------------------------------------------------------------------------
// AC7: Feature flag rollback
// ---------------------------------------------------------------------------
describe('AC7 - ANTIDETECT_TIER1_ENABLED=false fallback', () => {
  it('returns legacy static headers when flag is disabled', () => {
    // Temporarily override config
    const config = require('../src/config/index.js')
    const original = config.antidetectTier1Enabled
    config.antidetectTier1Enabled = false
    try {
      const account = {
        email: 'flag-test@example.com',
        fingerprint: generateDeterministicFingerprint('flag-test@example.com')
      }
      const h = buildRequestHeaders(account, {})
      assert.strictEqual(h['user-agent'], LEGACY_UA)
      assert.strictEqual(h['sec-ch-ua-platform'], LEGACY_HEADERS['sec-ch-ua-platform'])
      assert.strictEqual(h['sec-ch-ua'], LEGACY_HEADERS['sec-ch-ua'])
    } finally {
      config.antidetectTier1Enabled = original
    }
  })
})
