/// <reference types="chrome" />
import { useEffect, useState } from "react"
import "./popup.css"

interface Feature { name: string; value: number; avg_imp: number }
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
  domain: string; rf_conf: number; rf_label: string; rf_threshold: number
  xgb_conf: number; xgb_label: string; xgb_threshold: number
  max_conf: number; ml_tier: string; final_tier: string
  features: Feature[]; whois_eligible: boolean
}

const TIER: Record<string, { bg: string; fg: string }> = {
  ALLOW:   { bg: "#d4edda", fg: "#155724" },
  WARNING: { bg: "#fff3cd", fg: "#856404" },
  BLOCK:   { bg: "#f8d7da", fg: "#721c24" },
}

function pct(v: number) { return (v * 100).toFixed(1) + "%" }

export default function Popup() {
  const [domain,       setDomain]       = useState("")
  const [result,       setResult]       = useState<Result | null>(null)
  const [loading,      setLoading]      = useState(true)
  const [error,        setError]        = useState("")
  const [whois,        setWhois]        = useState<WhoisResult | null>(null)
  const [whoisLoading, setWhoisLoading] = useState(false)
  const [whoisError,   setWhoisError]   = useState("")

  useEffect(() => {
    chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
      if (!tab?.url) { setLoading(false); return }
      let host = ""
      try { host = new URL(tab.url).hostname.replace(/^www\./, "") } catch {}
      if (!host || host === "localhost" || host === "127.0.0.1") { setLoading(false); return }
      setDomain(host)
      fetch(`http://127.0.0.1:8000/predict?domain=${encodeURIComponent(host)}`)
        .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
        .then(setResult)
        .catch(e => setError(e.message?.includes("fetch") ? "Server offline" : String(e)))
        .finally(() => setLoading(false))
    })
  }, [])

  function runWhois() {
    if (!result) return
    setWhoisLoading(true)
    setWhoisError("")
    fetch(`http://127.0.0.1:8000/whois?domain=${encodeURIComponent(result.domain)}`)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(setWhois)
      .catch(e => setWhoisError(e.message?.includes("fetch") ? "Server offline" : String(e)))
      .finally(() => setWhoisLoading(false))
  }

  const tier = result?.final_tier ?? ""
  const ts = TIER[tier]

  return (
    <div className="popup-root" style={ts ? { borderTop: `4px solid ${ts.fg}` } : {}}>
      <div className="header">Domain Threat Detector</div>

      {loading && <div className="state-msg">Checking {domain || "domain"}…</div>}

      {!loading && error && (
        <div className="state-msg error">
          {error}
          <div className="hint">Is uvicorn running on port 8000?</div>
        </div>
      )}

      {!loading && !error && !result && (
        <div className="state-msg">Open a web page to see threat analysis.</div>
      )}

      {result && (
        <>
          <div className="banner" style={{ background: ts?.bg, color: ts?.fg }}>
            <div className="tier-label">{tier}</div>
            <div className="domain-text">{result.domain}</div>
            <div className="conf-text">max confidence: {pct(result.max_conf)}</div>
          </div>

          <div className="section-title">Models</div>
          <div className="models">
            {[
              { name: "Random Forest", label: result.rf_label, conf: result.rf_conf, thr: result.rf_threshold },
              { name: "XGBoost",       label: result.xgb_label, conf: result.xgb_conf, thr: result.xgb_threshold },
            ].map(m => (
              <div key={m.name} className="model-row">
                <span className="model-name">{m.name}</span>
                <span className={`model-label ${m.label === "malicious" ? "mal" : "ben"}`}>{m.label}</span>
                <span className="model-conf">{pct(m.conf)}</span>
                <span className="model-thr">thr {m.thr.toFixed(2)}</span>
              </div>
            ))}
          </div>

          <div className="section-title">Top features</div>
          <table className="feat-table">
            <tbody>
              {result.features.slice(0, 7).map(f => (
                <tr key={f.name}>
                  <td className="feat-name">{f.name}</td>
                  <td className="feat-val">{Number.isInteger(f.value) ? f.value : f.value.toFixed(3)}</td>
                  <td className="feat-imp">{f.avg_imp.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* WHOIS section — shown when domain is malicious or confidence is uncertain */}
          {result.whois_eligible && !whois && (
            <div className="whois-prompt">
              <button
                className="whois-btn"
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
              <div className="section-title">
                WHOIS
                {whois.final_tier && whois.final_tier !== whois.original_tier && (
                  <span className="tier-change">
                    {" "}tier adjusted: {whois.original_tier} → {whois.final_tier}
                  </span>
                )}
              </div>
              <div className="whois-block">
                {!whois.available ? (
                  <div className="whois-na">Lookup timed out or unavailable</div>
                ) : (
                  <table className="feat-table">
                    <tbody>
                      <tr>
                        <td>Domain age</td>
                        <td className="feat-val">
                          {whois.domain_age_days != null ? `${whois.domain_age_days} days` : "unknown"}
                        </td>
                      </tr>
                      <tr>
                        <td>Privacy protection</td>
                        <td className="feat-val">{whois.has_privacy_protection ? "Yes" : "No"}</td>
                      </tr>
                      <tr>
                        <td>Known registrar</td>
                        <td className="feat-val">{whois.registrar_known ? "Yes" : "No"}</td>
                      </tr>
                      <tr>
                        <td>Expires within 30d</td>
                        <td className="feat-val">{whois.expires_soon ? "Yes" : "No"}</td>
                      </tr>
                    </tbody>
                  </table>
                )}
                {whois.reasons?.map((r, i) => <div key={i} className="reason">• {r}</div>)}
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}
