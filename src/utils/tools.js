const crypto = require('crypto')
const { jwtDecode } = require('jwt-decode')
const { logger } = require('./logger')


const isJson = (str) => {
  try {
    JSON.parse(str)
    return true
  } catch (error) {
    return false
  }
}

const sleep = async (ms) => {
  return await new Promise(resolve => setTimeout(resolve, ms))
}

const sha256Encrypt = (text) => {
  if (typeof text !== 'string') {
    logger.error('输入必须是字符串类型', 'TOOLS')
    throw new Error('输入必须是字符串类型')
  }
  const hash = crypto.createHash('sha256')
  hash.update(text, 'utf-8')
  return hash.digest('hex')
}

const JwtDecode = (token) => {
  try {
    const decoded = jwtDecode(token, { complete: true })
    return decoded
  } catch (error) {
    logger.error('解析JWT失败', 'JWT', '', error)
    return null
  }
}

/**
 * 生成UUID v4
 * 使用Node.js内置的crypto.randomUUID()
 * @returns {string} UUID v4字符串
 */
const generateUUID = () => {
  return crypto.randomUUID()
}

/**
 * 浏览器风格 timezone 请求头（仅 ASCII）
 * 中文 Windows 的 Date#toString 含“中国标准时间”，会破坏 HTTP 头。
 * @returns {string}
 */
const getTimezoneHeader = () => {
  // 去掉非 ASCII（如“中国标准时间”），并压缩空白
  return new Date().toString().replace(/[^\x20-\x7E]/g, '').replace(/\s+/g, ' ').trim()
}

/**
 * Add timing jitter to break machine-perfect patterns.
 * @param {number} baseMs - Base delay in milliseconds
 * @param {number} pct - Jitter percentage (0-1). Default 0.25 = ±25%
 * @returns {number} Jittered delay in ms
 */
const jitter = (baseMs, pct = 0.25) => {
  const range = baseMs * pct
  return Math.max(0, Math.round(baseMs + (Math.random() * 2 - 1) * range))
}

module.exports = {
  isJson,
  sleep,
  sha256Encrypt,
  JwtDecode,
  generateUUID,
  getTimezoneHeader,
  jitter
}
