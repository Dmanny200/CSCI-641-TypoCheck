/**
 * background.ts
 *
 * Intercepts every main-frame HTTP/HTTPS navigation, redirects to an
 * interstitial page that calls the local API and shows the verdict.
 * The user must confirm before the original page loads.
 */

const API_BASE        = "http://127.0.0.1:8000"
const INTERSTITIAL    = chrome.runtime.getURL("tabs/interstitial.html")

// Tracks URLs that the user explicitly chose to proceed through (tabId -> url)
const allowedNavigations = new Map<number, string>()

const SKIP_HOSTS = new Set(["localhost", "127.0.0.1", "newtab", ""])

function shouldSkip(url: string): boolean {
  try {
    const { protocol, hostname } = new URL(url)
    if (protocol !== "http:" && protocol !== "https:") return true
    if (SKIP_HOSTS.has(hostname)) return true
    if (/^\d+\.\d+\.\d+\.\d+$/.test(hostname)) return true
    return false
  } catch {
    return true
  }
}

// ── intercept navigations ─────────────────────────────────────────────────────

chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  if (details.frameId !== 0) return                      // main frame only
  if (!details.url.startsWith("http")) return            // http/https only
  if (details.url.startsWith(INTERSTITIAL)) return       // never intercept our own page
  if (shouldSkip(details.url)) return

  // If the user already clicked "Proceed" for this exact URL, let it through
  const allowed = allowedNavigations.get(details.tabId)
  if (allowed === details.url) {
    allowedNavigations.delete(details.tabId)
    return
  }

  const interstitialUrl =
    `${INTERSTITIAL}?target=${encodeURIComponent(details.url)}&tabId=${details.tabId}`
  chrome.tabs.update(details.tabId, { url: interstitialUrl })
})

// ── messages from the interstitial page ───────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "PROCEED") {
    // Mark this URL as allowed for this tab, then navigate
    allowedNavigations.set(msg.tabId, msg.target)
    chrome.tabs.update(msg.tabId, { url: msg.target })
    sendResponse({ ok: true })
  }
  return true
})

// ── clean up on tab close ─────────────────────────────────────────────────────

chrome.tabs.onRemoved.addListener((tabId) => {
  allowedNavigations.delete(tabId)
})
