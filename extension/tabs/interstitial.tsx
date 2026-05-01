/**
 * interstitial.tsx
 *
 * Full-page verdict screen shown before every HTTP/HTTPS navigation.
 *
 * ALLOW   → auto-redirects after 2 s (user can skip the wait)
 * WARNING → shows result, user picks Proceed or Go Back
 * BLOCK   → shows result, Go Back is the primary action
 * ERROR   → server offline, user can proceed unverified or go back
 */

import { useEffect, useRef, useState } from "react"
import "./interstitial.css"

// ── types ─────────────────────────────────────────────────────────────────────

interface Feature {
  name: string
  value: number
  avg_imp: number
}

interface WhoisResult {
  available: boolean
  domain_age_days?: number | null
  has_privacy_protection?: boolean | null
  registrar_known?: boolean | null
  expires_soon?: boolean | null
  original_tier?: string
  final_tier?: string
  reasons?: string[]
}

interface Result {
  domain: string
  rf_conf: number
  rf_label: string
  rf_threshold: number
  xgb_conf: number
  xgb_label: string
  xgb_threshold: number
  max_conf: number
  ml_tier: string
  final_tier: string
  features: Feature[]
  whois_eligible: boolean
}

type Phase = "loading" | "allow" | "warning" | "block" | "error"

// ── tier config ───────────────────────────────────────────────────────────────

const TIER_CFG = {
  allow:   { bg: "#d4edda", fg: "#155724", border: "#28a745", icon: "✓", label: "ALLOW" },
  warning: { bg: "#fff3cd", fg: "#856404", border: "#ffc107", icon: "⚠", label: "WARNING" },
  block:   { bg: "#f8d7da", fg: "#721c24", border: "#dc3545", icon: "✕", label: "BLOCK" },
  error:   { bg: "#f0f0f0", fg: "#333",    border: "#999",    icon: "?", label: "UNKNOWN" },
  loading: { bg: "#e8f0fe", fg: "#1a73e8", border: "#1a73e8", icon: "…", label: "CHECKING" },
}

function pct(v: number) { return (v * 100).toFixed(1) + "%" }

// ── main component ────────────────────────────────────────────────────────────

export default function Interstitial() {
  const params   = new URLSearchParams(window.location.search)
  const target   = params.get("target") ?? ""
  const tabId    = parseInt(params.get("tabId") ?? "0", 10)

  let displayDomain = ""
  try { displayDomain = new URL(target).hostname.replace(/^www\./, "") } catch {}

  const [phase,        setPhase]        = useState<Phase>("loading")
  const [result,       setResult]       = useState<Result | null>(null)
  const [errMsg,       setErrMsg]       = useState("")
  const [countdown,    setCd]           = useState(4)
  const [whois,        setWhois]        = useState<WhoisResult | null>(null)
  const [whoisLoading, setWhoisLoading] = useState(false)
  const [whoisError,   setWhoisError]   = useState("")
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── fetch verdict ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!displayDomain) { setPhase("error"); setErrMsg("Invalid URL"); return }

    fetch(`http://127.0.0.1:8000/predict?url=${encodeURIComponent(target)}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then((data: Result) => {
        setResult(data)
        const tier = data.final_tier
        if (tier === "BLOCK")        { setPhase("block") }
        else if (tier === "WARNING") { setPhase("warning") }
        else                         { setPhase("allow") }
      })
      .catch(err => {
        setPhase("error")
        setErrMsg(err.message?.includes("fetch") ? "Server offline — is uvicorn running?" : String(err))
      })
  }, [])

  // ── auto-redirect countdown for ALLOW ────────────────────────────────────────
  useEffect(() => {
    if (phase !== "allow") return
    timerRef.current = setInterval(() => {
      setCd(prev => {
        if (prev <= 1) { clearInterval(timerRef.current!); proceed(); return 0 }
        return prev - 1
      })
    }, 1000)
    return () => clearInterval(timerRef.current!)
  }, [phase])

  // ── actions ──────────────────────────────────────────────────────────────────
  function proceed() {
    clearInterval(timerRef.current!)
    chrome.runtime.sendMessage({ type: "PROCEED", tabId, target })
  }

  function goBack() {
    clearInterval(timerRef.current!)
    if (window.history.length > 1) window.history.back()
    else window.close()
  }

  function runWhois() {
    if (!result) return
    setWhoisLoading(true)
    setWhoisError("")
    fetch(`http://127.0.0.1:8000/whois?url=${encodeURIComponent(target)}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(setWhois)
      .catch(e => setWhoisError(e.message?.includes("fetch") ? "Server offline" : String(e)))
      .finally(() => setWhoisLoading(false))
  }

  // ── render ───────────────────────────────────────────────────────────────────
  const cfg = TIER_CFG[phase]

  return (
    <div className="page" style={{ borderTop: `5px solid ${cfg.border}` }}>

      {/* ── header ── */}
      <div className="header">
        <span className="logo">Domain Threat Detector</span>
      </div>

      {/* ── verdict card ── */}
      <div className="card" style={{ background: cfg.bg, color: cfg.fg, borderColor: cfg.border }}>
        <div className="verdict-icon" style={{ color: cfg.border }}>{cfg.icon}</div>
        <div className="verdict-label">{cfg.label}</div>
        <div className="domain-name">{displayDomain || "—"}</div>

        {phase === "loading" && (
          <div className="sub">Contacting detection server…</div>
        )}

        {result && phase !== "loading" && (
          <div className="conf-row">
            Max confidence: <strong>{pct(result.max_conf)}</strong> malicious
          </div>
        )}

        {phase === "allow" && (
          <div className="countdown">
            Redirecting in <strong>{countdown}s</strong>
            <button className="btn-ghost" onClick={proceed}>Go now</button>
          </div>
        )}
      </div>

      {/* ── model detail ── */}
      {result && (
        <div className="detail-grid">
          <div className="detail-box">
            <div className="box-title">Model predictions</div>
            <table className="mini-table">
              <tbody>
                <tr>
                  <td>Random Forest</td>
                  <td className={result.rf_label === "malicious" ? "mal" : "ben"}>
                    {result.rf_label}
                  </td>
                  <td>{pct(result.rf_conf)}</td>
                  <td className="thr">thr {result.rf_threshold.toFixed(2)}</td>
                </tr>
                <tr>
                  <td>XGBoost</td>
                  <td className={result.xgb_label === "malicious" ? "mal" : "ben"}>
                    {result.xgb_label}
                  </td>
                  <td>{pct(result.xgb_conf)}</td>
                  <td className="thr">thr {result.xgb_threshold.toFixed(2)}</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="detail-box">
            <div className="box-title">Top features</div>
            <table className="mini-table">
              <tbody>
                {result.features.slice(0, 5).map(f => (
                  <tr key={f.name}>
                    <td>{f.name}</td>
                    <td>{Number.isInteger(f.value) ? f.value : f.value.toFixed(3)}</td>
                    <td className="imp">imp {f.avg_imp.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* WHOIS section — available for malicious or uncertain domains */}
          {result.whois_eligible && (
            <div className="detail-box whois-box">
              <div className="box-title">
                WHOIS Registration
                {whois?.final_tier && whois.final_tier !== whois.original_tier && (
                  <span className="tier-change">
                    {" "}· tier adjusted: {whois.original_tier} → {whois.final_tier}
                  </span>
                )}
              </div>

              {!whois && (
                <div className="whois-prompt">
                  <div className="whois-hint">
                    Look up registration data for additional context.
                  </div>
                  <button
                    className="btn-whois"
                    onClick={runWhois}
                    disabled={whoisLoading}
                  >
                    {whoisLoading ? "Looking up…" : "Run WHOIS Lookup"}
                  </button>
                  {whoisError && <div className="whois-err">{whoisError}</div>}
                </div>
              )}

              {whois && (
                <>
                  {!whois.available ? (
                    <div className="whois-na">Lookup timed out or unavailable</div>
                  ) : (
                    <table className="mini-table">
                      <tbody>
                        <tr>
                          <td>Domain age</td>
                          <td>{whois.domain_age_days != null
                            ? `${whois.domain_age_days} days` : "unknown"}</td>
                        </tr>
                        <tr>
                          <td>Privacy protection</td>
                          <td>{whois.has_privacy_protection ? "Yes" : "No"}</td>
                        </tr>
                        <tr>
                          <td>Known registrar</td>
                          <td>{whois.registrar_known ? "Yes" : "No"}</td>
                        </tr>
                        <tr>
                          <td>Expires within 30d</td>
                          <td>{whois.expires_soon ? "Yes" : "No"}</td>
                        </tr>
                      </tbody>
                    </table>
                  )}
                  {whois.reasons?.map((r, i) => (
                    <div key={i} className="reason">• {r}</div>
                  ))}
                </>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── error detail ── */}
      {phase === "error" && errMsg && (
        <div className="error-detail">{errMsg}</div>
      )}

      {/* ── action buttons ── */}
      <div className="actions">
        {phase === "block" && (
          <>
            <button className="btn-primary" onClick={goBack}>← Go Back (Recommended)</button>
            <button className="btn-danger"  onClick={proceed}>Proceed anyway</button>
          </>
        )}
        {phase === "warning" && (
          <>
            <button className="btn-ghost"   onClick={goBack}>← Go Back</button>
            <button className="btn-primary" onClick={proceed}>Proceed →</button>
          </>
        )}
        {phase === "allow" && (
          <>
            <button className="btn-ghost" onClick={goBack}>← Go Back</button>
            <button className="btn-primary" onClick={proceed}>Proceed now →</button>
          </>
        )}
        {phase === "error" && (
          <>
            <button className="btn-ghost"   onClick={goBack}>← Go Back</button>
            <button className="btn-warning" onClick={proceed}>Proceed unverified →</button>
          </>
        )}
      </div>

      <div className="footer">
        Destination: <span className="target-url">{target}</span>
      </div>
    </div>
  )
}
