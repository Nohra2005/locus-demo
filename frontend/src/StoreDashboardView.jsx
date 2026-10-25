import { useState, useRef, useEffect, useCallback } from "react";

const API = "";

const T = {
  bg:          "#080808",
  bgDeep:      "#040404",
  surface:     "#111111",
  surfaceHov:  "#181818",
  surfaceEl:   "#151515",
  border:      "#222222",
  borderFaint: "#141414",
  accent:      "#c9a96e",
  accentDeep:  "#a07840",
  accentBg:    "rgba(201,169,110,0.06)",
  accentRing:  "rgba(201,169,110,0.25)",
  text:        "#f0ede8",
  textMuted:   "#6b6458",
  textFaint:   "#3a3530",
  green:       "#7aab8a",
  yellow:      "#c9a96e",
  red:         "#c97070",
  blue:        "#6e9ecf",
};

// ── Auth helpers ───────────────────────────────────────────────────────────────

const AUTH_KEY = "locus_store_auth";

function getAuth() {
  try { return JSON.parse(localStorage.getItem(AUTH_KEY) || "null"); } catch { return null; }
}
function saveAuth(data) { localStorage.setItem(AUTH_KEY, JSON.stringify(data)); }
function clearAuth()    { localStorage.removeItem(AUTH_KEY); }

function authHeaders(token) {
  return { "Content-Type": "application/json", Authorization: `Bearer ${token}` };
}

// ── Batch indexing (shared, auth-aware) ───────────────────────────────────────

async function runBatchIndex(items, storeName, mallName, token, setProgress, setErrors, setStatus) {
  setStatus("indexing");
  setErrors([]);
  const CHUNK = 5;
  const total = items.length;
  let success = 0, failed = 0;
  const errs = [];

  for (let start = 0; start < total; start += CHUNK) {
    const chunk    = items.slice(start, start + CHUNK);
    const batchNum = Math.floor(start / CHUNK) + 1;
    const totalB   = Math.ceil(total / CHUNK);

    setProgress({ done: start, total, success, failed,
      current: `Batch ${batchNum}/${totalB} (${start + 1}–${Math.min(start + CHUNK, total)})` });

    try {
      const headers = token
        ? { "Content-Type": "application/json", Authorization: `Bearer ${token}` }
        : { "Content-Type": "application/json" };

      const resp = await fetch(`${API}/add-bulk-batch`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          items: chunk.map(row => ({
            name:      row.name      || "Product",
            store:     storeName,
            mall:      mallName,
            image_url: row.image_url || row.image || "",
            price:     row.price     || "",
            category:  row.category  || "",
          })),
        }),
      });

      if (resp.ok) {
        const d = await resp.json();
        success += d.success || 0;
        failed  += (d.failed || []).length;
        (d.failed || []).forEach(f => errs.push(`${f.item}: ${f.error}`));
      } else {
        failed += chunk.length;
        errs.push(`Batch ${batchNum}: HTTP ${resp.status}`);
      }
    } catch (e) {
      failed += chunk.length;
      errs.push(`Batch ${batchNum}: ${e.message}`);
    }
  }
  setProgress({ done: total, total, success, failed, current: "" });
  setErrors(errs);
  setStatus("done");
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROOT COMPONENT — decides auth vs portal
// ═══════════════════════════════════════════════════════════════════════════════

export default function StoreDashboardView() {
  const [auth, setAuth]   = useState(getAuth);
  const [page, setPage]   = useState("dashboard");

  const handleLogin  = (data) => { saveAuth(data); setAuth(data); setPage("dashboard"); };
  const handleLogout = ()     => { clearAuth(); setAuth(null); setPage("dashboard"); };
  const handleProfileUpdate = (updates) => {
    const next = { ...auth, ...updates };
    saveAuth(next);
    setAuth(next);
  };

  if (!auth) return <AuthPage onAuth={handleLogin} />;

  return (
    <PortalLayout
      auth={auth}
      page={page}
      onPage={setPage}
      onLogout={handleLogout}
      onProfileUpdate={handleProfileUpdate}
    />
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTH PAGE — login + register
// ═══════════════════════════════════════════════════════════════════════════════

function AuthPage({ onAuth }) {
  // mode: "login" | "register"
  const [mode,      setMode]      = useState("login");
  const [email,     setEmail]     = useState("");
  const [pass,      setPass]      = useState("");
  const [name,      setName]      = useState("");
  const [mall,      setMall]      = useState("");
  const [phone,     setPhone]     = useState("");
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState("");
  const [success,   setSuccess]   = useState("");
  const [showPass,  setShowPass]  = useState(false);

  const goTo = (m) => { setMode(m); setError(""); setSuccess(""); };

  const apiPost = async (endpoint, body) => {
    const resp = await fetch(`${API}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let data = {};
    try { data = await resp.json(); } catch {}
    if (!resp.ok) throw new Error(data.detail || `Server error (${resp.status}). Please try again.`);
    return data;
  };

  const submit = async (e) => {
    e.preventDefault();
    setError(""); setSuccess(""); setLoading(true);
    try {
      const endpoint = mode === "login" ? "/auth/login" : "/auth/register";
      const body = mode === "login"
        ? { email: email.trim(), password: pass }
        : { email: email.trim(), password: pass, store_name: name.trim(), mall: mall.trim(), phone: phone.trim() };
      const data = await apiPost(endpoint, body);
      onAuth(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const headingMap = {
    login:   "Welcome back",
    register:"Join Locus",
  };
  const subMap = {
    login:   "Sign in to manage your store's catalogue.",
    register:"Create a retailer account to list your products.",
  };
  const btnMap = {
    login:   { idle: "Sign in",        busy: "Signing in…" },
    register:{ idle: "Create account", busy: "Creating account…" },
  };

  return (
    <div style={{
      minHeight: "calc(100dvh - 61px)",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: "32px 20px",
      background: T.bg,
    }}>
      <div style={{ width: "100%", maxWidth: 440 }}>

        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14, marginBottom: 20 }}>
            <svg width="52" height="52" viewBox="0 0 48 48" fill="none">
              <circle cx="24" cy="24" r="15" stroke={T.accent} strokeWidth="1.2"/>
              <circle cx="24" cy="24" r="3" fill={T.accent}/>
              <line x1="24" y1="4"  x2="24" y2="10" stroke={T.accent} strokeWidth="1.2" strokeLinecap="round"/>
              <line x1="24" y1="38" x2="24" y2="44" stroke={T.accent} strokeWidth="1.2" strokeLinecap="round"/>
              <line x1="4"  y1="24" x2="10" y2="24" stroke={T.accent} strokeWidth="1.2" strokeLinecap="round"/>
              <line x1="38" y1="24" x2="44" y2="24" stroke={T.accent} strokeWidth="1.2" strokeLinecap="round"/>
            </svg>
            <span style={{
              fontFamily: "'Josefin Sans', sans-serif",
              fontSize: "0.75rem", fontWeight: 300, color: T.accent,
              letterSpacing: "0.5em", textTransform: "uppercase", paddingRight: "0.5em",
            }}>locus</span>
          </div>
          <h1 style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "2.2rem",
            fontWeight: 500, color: T.text, margin: 0, lineHeight: 1.1 }}>
            {headingMap[mode]}
          </h1>
          <p style={{ marginTop: 10, fontSize: "0.8rem", color: T.textMuted }}>
            {subMap[mode]}
          </p>
        </div>

        {/* Card */}
        <form onSubmit={submit} style={{
          background: T.surface, border: `1px solid ${T.border}`,
          borderRadius: 16, padding: "32px 28px",
        }}>

          {error && (
            <div style={{ background: "rgba(201,112,112,0.07)", border: `1px solid rgba(201,112,112,0.22)`,
              borderRadius: 8, padding: "10px 14px", fontSize: "0.78rem",
              color: T.red, marginBottom: 22 }}>
              {error}
            </div>
          )}

          {success && (
            <div style={{ background: "rgba(122,171,138,0.08)", border: `1px solid rgba(122,171,138,0.28)`,
              borderRadius: 8, padding: "10px 14px", fontSize: "0.78rem",
              color: T.green, marginBottom: 22 }}>
              {success}
            </div>
          )}

          <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>

            {/* ── register fields ── */}
            {mode === "register" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                <AuthInput label="Store name" value={name} onChange={setName}
                  placeholder="e.g. Zara" required />
                <AuthInput label="Mall / Location" value={mall} onChange={setMall}
                  placeholder="e.g. ABC Achrafieh" required />
              </div>
            )}

            {/* ── email ── */}
            {(mode === "login" || mode === "register") && (
              <AuthInput label="Email address" type="email" value={email}
                onChange={setEmail} placeholder="you@store.com" required />
            )}

            {/* ── password (login + register) ── */}
            {(mode === "login" || mode === "register") && (
              <div>
                <label style={labelStyle}>Password</label>
                <div style={{ position: "relative" }}>
                  <input
                    type={showPass ? "text" : "password"}
                    value={pass} onChange={e => setPass(e.target.value)}
                    placeholder={mode === "register" ? "8–128 characters" : "Your password"}
                    minLength={mode === "register" ? 8 : undefined}
                    maxLength={128}
                    required
                    style={{ ...inputStyle, paddingRight: 42 }}
                  />
                  <button type="button" onClick={() => setShowPass(s => !s)} style={{
                    position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)",
                    background: "none", border: "none", cursor: "pointer",
                    color: T.textMuted, fontSize: "0.8rem", padding: 0,
                  }}>
                    {showPass ? "hide" : "show"}
                  </button>
                </div>
              </div>
            )}

            {/* ── phone (register only) ── */}
            {mode === "register" && (
              <AuthInput label="Phone (optional)" type="tel" value={phone}
                onChange={setPhone} placeholder="+961 70 000 000" />
            )}

          </div>

          <button type="submit" disabled={loading} style={{
            width: "100%", marginTop: 28, padding: "13px 20px",
            background: loading ? T.surfaceHov : T.accent,
            border: "none", borderRadius: 10, cursor: loading ? "not-allowed" : "pointer",
            fontSize: "0.85rem", fontWeight: 600, color: T.bgDeep,
            fontFamily: "'DM Sans', sans-serif", letterSpacing: "0.04em",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
            transition: "opacity 0.2s", opacity: loading ? 0.6 : 1,
          }}>
            {loading
              ? <><Spinner small /> {btnMap[mode].busy}</>
              : btnMap[mode].idle}
          </button>

          {/* ── bottom links ── */}
          <div style={{ textAlign: "center", marginTop: 22, fontSize: "0.78rem", color: T.textMuted }}>
            {(mode === "login" || mode === "register") && (
              <>
                {mode === "login" ? "Don't have an account? " : "Already have an account? "}
                <button type="button" onClick={() => goTo(mode === "login" ? "register" : "login")}
                  style={{ background: "none", border: "none", cursor: "pointer",
                    color: T.accent, fontSize: "0.78rem", padding: 0, fontFamily: "'DM Sans', sans-serif" }}>
                  {mode === "login" ? "Sign up" : "Sign in"}
                </button>
              </>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// PORTAL LAYOUT — sidebar + content
// ═══════════════════════════════════════════════════════════════════════════════

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard",        icon: "⊞" },
  { id: "inventory", label: "Inventory",         icon: "◫" },
  { id: "upload",    label: "Upload",            icon: "↑" },
  { id: "settings",  label: "Settings",          icon: "⚙" },
];

function PortalLayout({ auth, page, onPage, onLogout, onProfileUpdate }) {
  return (
    <div style={{
      minHeight: "calc(100dvh - 61px)",
      display: "flex",
      background: T.bg,
    }}>
      {/* Sidebar — desktop only */}
      <aside style={{
        width: 220, flexShrink: 0,
        borderRight: `1px solid ${T.border}`,
        display: "flex", flexDirection: "column",
        padding: "32px 0",
        position: "sticky", top: 0,
        height: "calc(100dvh - 61px)",
        overflowY: "auto",
      }}
        className="portal-sidebar"
      >
        <div style={{ padding: "0 20px", marginBottom: 32 }}>
          <div style={{ fontSize: "0.6rem", color: T.accent, letterSpacing: "0.2em",
            textTransform: "uppercase", marginBottom: 6,
            display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: "0.5rem" }}>✦</span> Store Portal
          </div>
          <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "1.1rem",
            color: T.text, fontWeight: 500, lineHeight: 1.2,
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {auth.store_name}
          </div>
          <div style={{ fontSize: "0.68rem", color: T.textMuted, marginTop: 2,
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {auth.mall || auth.email}
          </div>
        </div>

        <nav style={{ flex: 1 }}>
          {NAV_ITEMS.map(item => (
            <NavBtn key={item.id} item={item} active={page === item.id}
              onClick={() => onPage(item.id)} />
          ))}
        </nav>

        <div style={{ padding: "0 20px", paddingTop: 24, borderTop: `1px solid ${T.borderFaint}` }}>
          <button onClick={onLogout} style={{
            width: "100%", padding: "10px 14px",
            background: "none", border: `1px solid ${T.border}`,
            borderRadius: 8, cursor: "pointer",
            fontSize: "0.75rem", color: T.textMuted,
            fontFamily: "'DM Sans', sans-serif",
            textAlign: "left", display: "flex", alignItems: "center", gap: 8,
            transition: "all 0.18s",
          }}
            onMouseEnter={e => { e.currentTarget.style.color = T.red; e.currentTarget.style.borderColor = T.red; }}
            onMouseLeave={e => { e.currentTarget.style.color = T.textMuted; e.currentTarget.style.borderColor = T.border; }}
          >
            ↩ Sign out
          </button>
        </div>
      </aside>

      {/* Content */}
      <main style={{ flex: 1, overflowY: "auto", minWidth: 0 }}>
        <div style={{ maxWidth: 900, margin: "0 auto", padding: "40px 28px 80px" }}>
          {page === "dashboard" && <DashboardPage auth={auth} onPage={onPage} />}
          {page === "inventory" && <InventoryPage auth={auth} />}
          {page === "upload"    && <UploadPage    auth={auth} />}
          {page === "settings"  && <SettingsPage  auth={auth} onProfileUpdate={onProfileUpdate} onLogout={onLogout} />}
        </div>
      </main>

      {/* Bottom nav — mobile only */}
      <MobileNav page={page} onPage={onPage} />

      <style>{`
        @media (max-width: 640px) {
          .portal-sidebar { display: none !important; }
        }
        @media (min-width: 641px) {
          .portal-mobile-nav { display: none !important; }
        }
      `}</style>
    </div>
  );
}

function NavBtn({ item, active, onClick }) {
  const [hov, setHov] = useState(false);
  return (
    <button onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        width: "100%", padding: "10px 20px",
        background: active ? T.accentBg : hov ? T.surfaceHov : "none",
        border: "none", borderLeft: `2px solid ${active ? T.accent : "transparent"}`,
        cursor: "pointer", textAlign: "left",
        display: "flex", alignItems: "center", gap: 10,
        fontSize: "0.82rem",
        color: active ? T.accent : hov ? T.text : T.textMuted,
        fontFamily: "'DM Sans', sans-serif",
        fontWeight: active ? 600 : 400,
        transition: "all 0.18s",
      }}>
      <span style={{ fontSize: "0.9rem", opacity: active ? 1 : 0.7 }}>{item.icon}</span>
      {item.label}
    </button>
  );
}

function MobileNav({ page, onPage }) {
  return (
    <div className="portal-mobile-nav" style={{
      position: "fixed", bottom: 61, left: 0, right: 0, zIndex: 50,
      background: T.surface, borderTop: `1px solid ${T.border}`,
      display: "flex",
    }}>
      {NAV_ITEMS.map(item => (
        <button key={item.id} onClick={() => onPage(item.id)} style={{
          flex: 1, padding: "10px 4px",
          background: "none", border: "none",
          cursor: "pointer",
          display: "flex", flexDirection: "column", alignItems: "center", gap: 3,
          fontSize: "0.62rem",
          color: page === item.id ? T.accent : T.textMuted,
          fontFamily: "'DM Sans', sans-serif",
          fontWeight: page === item.id ? 600 : 400,
        }}>
          <span style={{ fontSize: "1rem" }}>{item.icon}</span>
          {item.label}
        </button>
      ))}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DASHBOARD PAGE
// ═══════════════════════════════════════════════════════════════════════════════

function DashboardPage({ auth, onPage }) {
  const [stats,   setStats]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState("");

  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch(`${API}/store-stats`, {
          headers: { Authorization: `Bearer ${auth.access_token}` },
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        setStats(await resp.json());
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    })();
  }, [auth.access_token]);

  const catColor = useCategoryColors();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
      {/* Header */}
      <div>
        <div style={{ fontSize: "0.65rem", color: T.accent, letterSpacing: "0.18em",
          textTransform: "uppercase", marginBottom: 10,
          display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: "0.55rem" }}>✦</span> Overview
        </div>
        <h2 style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "2rem",
          fontWeight: 500, color: T.text, margin: 0 }}>
          {auth.store_name}
        </h2>
        <p style={{ marginTop: 6, fontSize: "0.78rem", color: T.textMuted }}>
          {auth.mall && <>{auth.mall} · </>}{auth.email}
        </p>
      </div>

      {loading && <LoadingState label="Loading store stats…" />}
      {error   && <ErrorBanner msg={error} />}

      {stats && (
        <>
          {/* Stat cards */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>
            <StatCard icon="◫" label="Total Products" value={stats.total_products} color={T.accent} />
            <StatCard icon="★" label="Avg Rating"
              value={stats.avg_rating ? `${stats.avg_rating} / 5` : "No ratings yet"}
              sub={stats.rating_count ? `${stats.rating_count} reviews` : null}
              color={T.yellow} />
            <StatCard icon="▦" label="Categories"
              value={Object.keys(stats.categories).length}
              sub="product types"
              color={T.blue} />
          </div>

          {/* Category breakdown */}
          {Object.keys(stats.categories).length > 0 && (
            <Section title="Category breakdown">
              <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
                {Object.entries(stats.categories)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cat, count]) => (
                    <div key={cat} style={{
                      display: "flex", alignItems: "center", gap: 8,
                      background: T.surfaceEl, border: `1px solid ${T.border}`,
                      borderRadius: 8, padding: "8px 14px",
                    }}>
                      <span style={{ width: 8, height: 8, borderRadius: "50%",
                        background: catColor[cat] || T.textMuted, display: "inline-block" }} />
                      <span style={{ fontSize: "0.75rem", color: T.text, textTransform: "capitalize" }}>{cat}</span>
                      <span style={{ fontSize: "0.72rem", color: T.textMuted }}>{count}</span>
                    </div>
                  ))}
              </div>
            </Section>
          )}

          {/* Recent products */}
          {stats.recent_products.length > 0 && (
            <Section title="Recently indexed" action={{ label: "View all →", onClick: () => onPage("inventory") }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(130px, 1fr))", gap: 12 }}>
                {stats.recent_products.map((p, i) => (
                  <MiniProductCard key={i} product={p} catColor={catColor} />
                ))}
              </div>
            </Section>
          )}

          {stats.total_products === 0 && (
            <EmptyState
              icon="◫"
              title="No products yet"
              body="Upload a CSV or scrape your website to start indexing products into Locus."
              action={{ label: "Upload products →", onClick: () => onPage("upload") }}
            />
          )}
        </>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// INVENTORY PAGE
// ═══════════════════════════════════════════════════════════════════════════════

function InventoryPage({ auth }) {
  const [products, setProducts] = useState([]);
  const [total,    setTotal]    = useState(0);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState("");
  const [offset,   setOffset]   = useState(0);
  const [search,   setSearch]   = useState("");
  const [deleting, setDeleting] = useState({});
  const LIMIT = 24;

  const fetchProducts = useCallback(async (off = 0) => {
    setLoading(true); setError("");
    try {
      const resp = await fetch(
        `${API}/store-catalogue?store_name=${encodeURIComponent(auth.store_name)}&limit=${LIMIT}&offset=${off}`,
        { headers: { Authorization: `Bearer ${auth.access_token}` } }
      );
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setProducts(data.products || []);
      setTotal(data.total || 0);
      setOffset(off);
    } catch (e) {
      setError(`Failed to load: ${e.message}`);
    } finally {
      setLoading(false);
    }
  }, [auth.store_name, auth.access_token]);

  useEffect(() => { fetchProducts(0); }, [fetchProducts]);

  const handleDelete = async (id, name) => {
    if (!window.confirm(`Remove "${name}" from your catalogue?`)) return;
    setDeleting(d => ({ ...d, [id]: true }));
    try {
      const resp = await fetch(`${API}/store-catalogue/item/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${auth.access_token}` },
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setProducts(ps => ps.filter(p => p.id !== id));
      setTotal(t => t - 1);
    } catch (e) {
      alert(`Delete failed: ${e.message}`);
    } finally {
      setDeleting(d => ({ ...d, [id]: false }));
    }
  };

  const catColor = useCategoryColors();
  const filtered = products.filter(p =>
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    (p.category_tag || "").toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between",
        flexWrap: "wrap", gap: 16 }}>
        <div>
          <PageTitle label="Catalogue" sub="Manage your indexed products" />
        </div>
        <button className="btn-ghost" onClick={() => fetchProducts(offset)}
          disabled={loading} style={{ fontSize: "0.75rem", marginTop: 6 }}>
          {loading ? "Loading…" : "↻ Refresh"}
        </button>
      </div>

      {/* Search bar */}
      <div style={{ position: "relative" }}>
        <span style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)",
          color: T.textMuted, fontSize: "0.8rem" }}>⌕</span>
        <input
          value={search} onChange={e => setSearch(e.target.value)}
          placeholder="Search by name or category…"
          style={{ ...inputStyle, paddingLeft: 34, width: "100%", boxSizing: "border-box" }}
        />
      </div>

      {/* Summary */}
      <div style={{ fontSize: "0.75rem", color: T.textMuted }}>
        <span style={{ color: T.accent, fontWeight: 600 }}>{total}</span> products ·{" "}
        <span style={{ color: T.text }}>{auth.store_name}</span>
        {search && <> · <span style={{ color: T.textMuted }}>{filtered.length} match</span></>}
      </div>

      {error   && <ErrorBanner msg={error} />}

      {loading ? (
        <LoadingState label="Loading products…" />
      ) : filtered.length === 0 ? (
        <EmptyState
          icon="◫"
          title={total === 0 ? "No products indexed" : "No matches"}
          body={total === 0 ? "Use the Upload tab to add products." : "Try a different search term."}
        />
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))", gap: 14 }}>
          {filtered.map(p => (
            <CatalogueCard
              key={p.id} product={p} catColor={catColor}
              deleting={!!deleting[p.id]}
              onDelete={() => handleDelete(p.id, p.name)}
            />
          ))}
        </div>
      )}

      {total > LIMIT && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 12, marginTop: 8 }}>
          <button className="btn-ghost"
            onClick={() => fetchProducts(Math.max(0, offset - LIMIT))}
            disabled={offset === 0 || loading}
            style={{ fontSize: "0.78rem" }}>← Prev</button>
          <span style={{ fontSize: "0.72rem", color: T.textMuted }}>
            {offset + 1}–{Math.min(offset + LIMIT, total)} of {total}
          </span>
          <button className="btn-ghost"
            onClick={() => fetchProducts(offset + LIMIT)}
            disabled={offset + LIMIT >= total || loading}
            style={{ fontSize: "0.78rem" }}>Next →</button>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPLOAD PAGE
// ═══════════════════════════════════════════════════════════════════════════════

function UploadPage({ auth }) {
  const [tab, setTab] = useState("csv");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <PageTitle label="Upload Products" sub="Index new products into your catalogue" />

      {/* Tabs */}
      <div style={{
        display: "flex", gap: 4, background: T.surface,
        border: `1px solid ${T.border}`, borderRadius: 10,
        padding: 4, width: "fit-content",
      }}>
        {[
          { id: "csv",    label: "CSV / Excel",    icon: "📋" },
          { id: "scrape", label: "Scrape Website", icon: "🌐" },
        ].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: "8px 20px", borderRadius: 7, border: "none", cursor: "pointer",
            fontSize: "0.8rem", fontFamily: "'DM Sans', sans-serif",
            fontWeight: tab === t.id ? 600 : 400,
            color:      tab === t.id ? T.text : T.textMuted,
            background: tab === t.id ? T.surfaceHov : "transparent",
            transition: "all 0.2s", display: "flex", alignItems: "center", gap: 6,
          }}>
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {tab === "csv"    && <CsvUploadPanel     auth={auth} />}
      {tab === "scrape" && <ScrapeWebsitePanel auth={auth} />}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SETTINGS PAGE
// ═══════════════════════════════════════════════════════════════════════════════

function SettingsPage({ auth, onProfileUpdate, onLogout }) {
  const [name,      setName]      = useState(auth.store_name || "");
  const [mall,      setMall]      = useState(auth.mall || "");
  const [phone,     setPhone]     = useState(auth.phone || "");
  const [mapsUrl,   setMapsUrl]   = useState(auth.maps_url || "");
  const [latitude,  setLatitude]  = useState(auth.latitude != null ? String(auth.latitude) : "");
  const [longitude, setLongitude] = useState(auth.longitude != null ? String(auth.longitude) : "");
  const [locating,  setLocating]  = useState(false);
  const [saving,    setSaving]    = useState(false);
  const [saved,     setSaved]     = useState(false);
  const [error,     setError]     = useState("");

  const handleUseMyLocation = () => {
    if (!navigator.geolocation) { setError("Geolocation is not supported by your browser."); return; }
    setLocating(true);
    navigator.geolocation.getCurrentPosition(
      pos => {
        setLatitude(pos.coords.latitude.toFixed(6));
        setLongitude(pos.coords.longitude.toFixed(6));
        setLocating(false);
      },
      () => { setError("Could not get your location. Please allow location access and try again."); setLocating(false); },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  };

  const handleSave = async (e) => {
    e.preventDefault();
    setSaving(true); setError(""); setSaved(false);
    const lat = latitude !== "" ? parseFloat(latitude) : null;
    const lng = longitude !== "" ? parseFloat(longitude) : null;
    if ((latitude !== "" && isNaN(lat)) || (longitude !== "" && isNaN(lng))) {
      setError("Latitude and longitude must be valid numbers."); setSaving(false); return;
    }
    try {
      const resp = await fetch(`${API}/auth/profile`, {
        method: "PUT",
        headers: authHeaders(auth.access_token),
        body: JSON.stringify({
          store_name: name.trim(), mall: mall.trim(), phone: phone.trim(),
          maps_url: mapsUrl.trim(), latitude: lat, longitude: lng,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Update failed");
      onProfileUpdate({
        store_name: data.store_name, mall: data.mall, phone: data.phone,
        maps_url: data.maps_url, latitude: data.latitude, longitude: data.longitude,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 32, maxWidth: 520 }}>
      <PageTitle label="Settings" sub="Manage your store profile" />

      {/* Profile form */}
      <form onSubmit={handleSave} style={{
        background: T.surface, border: `1px solid ${T.border}`,
        borderRadius: 14, padding: "28px 28px",
      }}>
        <div style={{ fontSize: "0.65rem", color: T.accent, letterSpacing: "0.14em",
          textTransform: "uppercase", marginBottom: 22 }}>
          Store Profile
        </div>

        {error  && <ErrorBanner msg={error} />}
        {saved  && (
          <div style={{ background: "rgba(122,171,138,0.07)", border: `1px solid rgba(122,171,138,0.22)`,
            borderRadius: 8, padding: "10px 14px", fontSize: "0.78rem",
            color: T.green, marginBottom: 18 }}>
            Profile updated successfully.
          </div>
        )}

        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          <AuthInput label="Store name" value={name} onChange={setName}
            placeholder="e.g. Zara" required />
          <AuthInput label="Mall / Location" value={mall} onChange={setMall}
            placeholder="e.g. ABC Achrafieh" required />
          <AuthInput label="Phone" type="tel" value={phone} onChange={setPhone}
            placeholder="+961 70 000 000" />

          {/* Google Maps location */}
          <div>
            <label style={labelStyle}>Google Maps link</label>
            <input
              value={mapsUrl}
              onChange={e => setMapsUrl(e.target.value)}
              placeholder="https://maps.google.com/?q=..."
              type="url"
              style={{ ...inputStyle, width: "100%", boxSizing: "border-box" }}
            />
            <div style={{ fontSize: "0.68rem", color: T.textMuted, marginTop: 6, lineHeight: 1.5 }}>
              Open Google Maps, find your store, tap Share → Copy link, then paste it here.
              Customers will see a "Get directions" button on every product.
            </div>
            {mapsUrl && (
              <a href={mapsUrl} target="_blank" rel="noopener noreferrer"
                style={{ display: "inline-flex", alignItems: "center", gap: 5, marginTop: 8,
                  fontSize: "0.72rem", color: T.accent, textDecoration: "none" }}>
                ◎ Preview link →
              </a>
            )}
          </div>

          {/* Store coordinates for map display */}
          <div>
            <label style={labelStyle}>Store coordinates</label>
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <input
                value={latitude}
                onChange={e => setLatitude(e.target.value)}
                placeholder="Latitude (e.g. 33.8869)"
                style={{ ...inputStyle, flex: 1, minWidth: 130, boxSizing: "border-box" }}
              />
              <input
                value={longitude}
                onChange={e => setLongitude(e.target.value)}
                placeholder="Longitude (e.g. 35.5131)"
                style={{ ...inputStyle, flex: 1, minWidth: 130, boxSizing: "border-box" }}
              />
              <button
                type="button"
                onClick={handleUseMyLocation}
                disabled={locating}
                style={{
                  padding: "9px 14px", borderRadius: 9, border: `1px solid ${T.accentRing}`,
                  background: T.accentBg, color: T.accent, cursor: locating ? "not-allowed" : "pointer",
                  fontSize: "0.78rem", fontFamily: "inherit", whiteSpace: "nowrap",
                  display: "flex", alignItems: "center", gap: 6, opacity: locating ? 0.65 : 1,
                }}
              >
                {locating ? <><Spinner small /> Locating…</> : "⊕ Use my location"}
              </button>
            </div>
            <div style={{ fontSize: "0.68rem", color: T.textMuted, marginTop: 6, lineHeight: 1.5 }}>
              Tap "Use my location" while standing in your store — your pin will appear on the map
              when customers search for nearby items.
            </div>
            {latitude && longitude && (
              <div style={{ marginTop: 6, fontSize: "0.72rem", color: T.green }}>
                ✓ Pinned at {parseFloat(latitude).toFixed(5)}, {parseFloat(longitude).toFixed(5)}
              </div>
            )}
          </div>

          <div>
            <label style={labelStyle}>Email</label>
            <input value={auth.email} disabled
              style={{ ...inputStyle, opacity: 0.45, cursor: "not-allowed" }} />
            <div style={{ fontSize: "0.68rem", color: T.textMuted, marginTop: 6 }}>
              Email cannot be changed.
            </div>
          </div>
        </div>

        <button type="submit" disabled={saving} style={{
          marginTop: 24, padding: "11px 28px",
          background: saving ? T.surfaceHov : T.accentBg,
          border: `1px solid ${T.accentRing}`,
          borderRadius: 9, cursor: saving ? "not-allowed" : "pointer",
          fontSize: "0.82rem", color: T.accent,
          fontFamily: "'DM Sans', sans-serif",
          display: "flex", alignItems: "center", gap: 8,
          opacity: saving ? 0.7 : 1,
        }}>
          {saving ? <><Spinner small /> Saving…</> : "Save changes"}
        </button>
      </form>

      {/* Account info */}
      <div style={{ background: T.surface, border: `1px solid ${T.border}`,
        borderRadius: 14, padding: "24px 28px" }}>
        <div style={{ fontSize: "0.65rem", color: T.textMuted, letterSpacing: "0.14em",
          textTransform: "uppercase", marginBottom: 18 }}>
          Account
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <InfoRow label="Email"    value={auth.email} />
          <InfoRow label="Store ID" value={auth.store_id} mono />
        </div>
      </div>

      {/* Danger zone */}
      <div style={{ background: "rgba(201,112,112,0.04)", border: `1px solid rgba(201,112,112,0.15)`,
        borderRadius: 14, padding: "24px 28px" }}>
        <div style={{ fontSize: "0.65rem", color: T.red, letterSpacing: "0.14em",
          textTransform: "uppercase", marginBottom: 12 }}>
          Danger Zone
        </div>
        <p style={{ fontSize: "0.78rem", color: T.textMuted, marginBottom: 16, lineHeight: 1.6 }}>
          Sign out of the store portal on this device.
        </p>
        <button onClick={onLogout} style={{
          padding: "9px 20px",
          background: "none", border: `1px solid rgba(201,112,112,0.3)`,
          borderRadius: 8, cursor: "pointer",
          fontSize: "0.78rem", color: T.red,
          fontFamily: "'DM Sans', sans-serif",
        }}>
          Sign out
        </button>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// CSV UPLOAD PANEL
// ═══════════════════════════════════════════════════════════════════════════════

function CsvUploadPanel({ auth }) {
  const [file,     setFile]     = useState(null);
  const [rows,     setRows]     = useState([]);
  const [headers,  setHeaders]  = useState([]);
  const [status,   setStatus]   = useState("idle");
  const [progress, setProgress] = useState({ done: 0, total: 0, success: 0, failed: 0, current: "" });
  const [errors,   setErrors]   = useState([]);
  const inputRef = useRef();

  const handleFile = async (f) => {
    if (!f) return;
    setFile(f); setStatus("idle"); setErrors([]);
    const text  = await f.text();
    const lines = text.trim().split("\n").filter(Boolean);
    if (lines.length < 2) return;
    const hdrs = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, "").toLowerCase());
    const data  = lines.slice(1).map(line => {
      const vals = line.split(",").map(v => v.trim().replace(/^"|"$/g, ""));
      return Object.fromEntries(hdrs.map((h, i) => [h, vals[i] ?? ""]));
    });
    setHeaders(hdrs); setRows(data);
  };

  const hasRequired = headers.includes("name") && headers.includes("image_url");
  const handleIndex = () =>
    runBatchIndex(rows, auth.store_name, auth.mall || "", auth.access_token,
      setProgress, setErrors, setStatus);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <InfoBox title="Required columns">
        <div style={{ fontSize: "0.72rem", color: T.textMuted, fontFamily: "'DM Mono', monospace", lineHeight: 1.9 }}>
          <strong style={{ color: T.text }}>name</strong> &nbsp;·&nbsp;
          <strong style={{ color: T.text }}>image_url</strong> &nbsp;·&nbsp;
          price (optional) &nbsp;·&nbsp; category (optional)
        </div>
      </InfoBox>

      {rows.length > 0 && hasRequired && status === "idle" && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
          background: T.accentBg, border: `1px solid ${T.accentRing}`,
          borderRadius: 12, padding: "16px 20px" }}>
          <div>
            <div style={{ fontSize: "0.82rem", color: T.text, fontWeight: 500 }}>
              <span style={{ color: T.accent }}>{rows.length}</span> products ready
            </div>
            <div style={{ fontSize: "0.7rem", color: T.textMuted, marginTop: 2 }}>
              {file?.name} · ~{Math.max(1, Math.ceil(rows.length * 0.5 / 60))} min estimated
            </div>
          </div>
          <button className="btn-primary" onClick={handleIndex}>
            <span style={{ color: T.accent }}>✦</span> Index {rows.length} products
          </button>
        </div>
      )}

      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); }}
        style={{
          border: `2px dashed ${file ? T.accent : T.border}`,
          borderRadius: 12, padding: "44px 24px", textAlign: "center",
          cursor: "pointer", transition: "border-color 0.2s",
          background: file ? T.accentBg : "transparent",
        }}
      >
        <div style={{ fontSize: "1.5rem", marginBottom: 10 }}>{file ? "📄" : "⬆️"}</div>
        <div style={{ fontSize: "0.82rem", color: file ? T.accent : T.textMuted }}>
          {file ? file.name : "Drop CSV / Excel here or click to browse"}
        </div>
        <input ref={inputRef} type="file" accept=".csv,.xlsx,.xls"
          style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />
      </div>

      {headers.length > 0 && (
        <div style={{ fontSize: "0.72rem", color: hasRequired ? T.green : T.red }}>
          {hasRequired ? "✓ Required columns found" : "✕ Missing: name, image_url"} — Detected: {headers.join(", ")}
        </div>
      )}

      {rows.length > 0 && hasRequired && status === "idle" && (
        <details>
          <summary style={{ fontSize: "0.72rem", color: T.textMuted, cursor: "pointer",
            marginBottom: 10, userSelect: "none" }}>
            Preview first 5 rows
          </summary>
          <PreviewTable rows={rows.slice(0, 5)} headers={headers} />
        </details>
      )}

      {status === "indexing" && <IndexingProgress progress={progress} />}
      {status === "done"     && <DoneCard progress={progress} errors={errors} storeName={auth.store_name} />}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCRAPE WEBSITE PANEL
// ═══════════════════════════════════════════════════════════════════════════════

function ScrapeWebsitePanel({ auth }) {
  const [url,       setUrl]       = useState("");
  const [scraping,  setScraping]  = useState(false);
  const [products,  setProducts]  = useState(null);
  const [selected,  setSelected]  = useState({});
  const [status,    setStatus]    = useState("idle");
  const [progress,  setProgress]  = useState({ done: 0, total: 0, success: 0, failed: 0, current: "" });
  const [errors,    setErrors]    = useState([]);
  const [scrapeErr, setScrapeErr] = useState("");

  const handleScrape = async () => {
    if (!url.startsWith("http")) { setScrapeErr("Please enter a valid URL starting with https://"); return; }
    setScrapeErr(""); setScraping(true); setProducts(null); setSelected({}); setStatus("idle");
    try {
      const resp = await fetch(`${API}/scrape`, {
        method: "POST",
        headers: authHeaders(auth.access_token),
        body: JSON.stringify({ url, max_products: 0 }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data  = await resp.json();
      const prods = data.products || [];
      setProducts(prods);
      setSelected(Object.fromEntries(prods.map((_, i) => [i, true])));
    } catch (e) {
      setScrapeErr(`Scrape failed: ${e.message}`);
    } finally {
      setScraping(false);
    }
  };

  const toggleAll     = (val) => setSelected(Object.fromEntries((products || []).map((_, i) => [i, val])));
  const selectedItems = (products || []).filter((_, i) => selected[i]);
  const handleIndex   = () =>
    runBatchIndex(selectedItems, auth.store_name, auth.mall || "", auth.access_token,
      setProgress, setErrors, setStatus);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <InfoBox title="How scraping works">
        <div style={{ fontSize: "0.72rem", color: T.textMuted, lineHeight: 1.8 }}>
          Paste your store's product listing URL. Works best with{" "}
          <span style={{ color: T.accent }}>Shopify</span> stores — fetches all products automatically.
        </div>
      </InfoBox>

      <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 12, padding: "20px 24px" }}>
        <label style={labelStyle}>Product listing URL</label>
        <div style={{ display: "flex", gap: 10, marginTop: 8 }}>
          <input
            value={url} onChange={e => setUrl(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleScrape()}
            placeholder="https://yourstore.com/collections/all"
            style={{ flex: 1, ...inputStyle }}
          />
          <button className="btn-primary" onClick={handleScrape}
            disabled={scraping || !url.trim()}
            style={{ whiteSpace: "nowrap", opacity: (scraping || !url.trim()) ? 0.5 : 1 }}>
            {scraping ? (
              <><Spinner small /> Fetching…</>
            ) : (
              <><span style={{ color: T.accent }}>✦</span> Scrape</>
            )}
          </button>
        </div>
      </div>

      {scraping && (
        <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 12, padding: "24px 28px" }}>
          <div style={{ fontSize: "0.68rem", color: T.accent, letterSpacing: "0.14em",
            textTransform: "uppercase", marginBottom: 12 }}>Fetching products</div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <Spinner />
            <div style={{ fontSize: "0.82rem", color: T.text }}>Scraping product listings from store website…</div>
          </div>
          <div style={{ fontSize: "0.72rem", color: T.textMuted, marginTop: 8 }}>
            This may take 15–60 seconds for large stores
          </div>
        </div>
      )}

      {scrapeErr && <ErrorBanner msg={scrapeErr} />}

      {products !== null && status === "idle" && (
        <>
          {selectedItems.length > 0 && (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
              background: T.accentBg, border: `1px solid ${T.accentRing}`,
              borderRadius: 12, padding: "16px 20px" }}>
              <div>
                <div style={{ fontSize: "0.82rem", color: T.text, fontWeight: 500 }}>
                  <span style={{ color: T.accent }}>{selectedItems.length}</span> selected
                </div>
                <div style={{ fontSize: "0.7rem", color: T.textMuted, marginTop: 2 }}>
                  ~{Math.max(1, Math.ceil(selectedItems.length * 0.5 / 60))} min estimated
                </div>
              </div>
              <button className="btn-primary" onClick={handleIndex}>
                <span style={{ color: T.accent }}>✦</span> Index {selectedItems.length}
              </button>
            </div>
          )}

          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ fontSize: "0.72rem", color: T.textMuted }}>
              {products.length} found · {selectedItems.length} selected
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn-ghost" onClick={() => toggleAll(true)}  style={{ fontSize: "0.72rem" }}>All</button>
              <button className="btn-ghost" onClick={() => toggleAll(false)} style={{ fontSize: "0.72rem" }}>None</button>
            </div>
          </div>

          {products.length === 0 ? (
            <EmptyState icon="🔍" title="No products found" body="Try a different URL." />
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 14 }}>
              {products.map((p, i) => (
                <ScrapedProductCard key={i} product={p} selected={!!selected[i]}
                  onToggle={() => setSelected(s => ({ ...s, [i]: !s[i] }))} />
              ))}
            </div>
          )}
        </>
      )}

      {status === "indexing" && <IndexingProgress progress={progress} />}
      {status === "done"     && <DoneCard progress={progress} errors={errors} storeName={auth.store_name} />}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHARED SUB-COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

function CatalogueCard({ product: p, catColor, deleting, onDelete }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: T.surface, border: `1px solid ${hovered ? T.accentDeep : T.border}`,
        borderRadius: 10, overflow: "hidden", position: "relative", transition: "border-color 0.2s",
      }}
    >
      <div style={{ width: "100%", aspectRatio: "1/1", background: T.bgDeep, overflow: "hidden" }}>
        {p.image_url ? (
          <img src={p.image_url} alt={p.name}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
            onError={e => { e.target.style.display = "none"; }} />
        ) : (
          <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center",
            justifyContent: "center", color: T.textMuted, fontSize: "1.8rem" }}>📦</div>
        )}
      </div>
      <div style={{ padding: "10px 10px 8px" }}>
        <div style={{ fontSize: "0.75rem", color: T.text, fontWeight: 500, lineHeight: 1.35,
          overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box",
          WebkitLineClamp: 2, WebkitBoxOrient: "vertical", marginBottom: 8 }}>
          {p.name}
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 }}>
          {p.category_tag && (
            <span style={{
              fontSize: "0.6rem",
              color: catColor[p.category_tag] || T.textMuted,
              background: `${catColor[p.category_tag] || T.textMuted}18`,
              border: `1px solid ${catColor[p.category_tag] || T.textMuted}40`,
              borderRadius: 4, padding: "2px 6px",
              textTransform: "capitalize", letterSpacing: "0.06em",
            }}>{p.category_tag}</span>
          )}
          {p.price && <span style={{ fontSize: "0.7rem", color: T.accent, whiteSpace: "nowrap" }}>{p.price}</span>}
        </div>
      </div>
      {hovered && (
        <button onClick={onDelete} disabled={deleting} style={{
          position: "absolute", top: 6, right: 6,
          width: 26, height: 26, background: "rgba(0,0,0,0.78)",
          border: `1px solid ${T.border}`, borderRadius: "50%",
          color: T.red, cursor: "pointer", fontSize: "0.7rem",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          {deleting ? "…" : "✕"}
        </button>
      )}
    </div>
  );
}

function MiniProductCard({ product: p, catColor }) {
  return (
    <div style={{ background: T.surfaceEl, border: `1px solid ${T.border}`,
      borderRadius: 8, overflow: "hidden" }}>
      <div style={{ width: "100%", aspectRatio: "1/1", background: T.bgDeep, overflow: "hidden" }}>
        {p.image_url ? (
          <img src={p.image_url} alt={p.name}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
            onError={e => { e.target.style.display = "none"; }} />
        ) : (
          <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center",
            justifyContent: "center", color: T.textMuted, fontSize: "1.4rem" }}>📦</div>
        )}
      </div>
      <div style={{ padding: "7px 8px" }}>
        <div style={{ fontSize: "0.7rem", color: T.text, lineHeight: 1.3,
          overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box",
          WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>{p.name}</div>
        {p.category_tag && (
          <span style={{
            display: "inline-block", marginTop: 5, fontSize: "0.58rem",
            color: catColor[p.category_tag] || T.textMuted,
            background: `${catColor[p.category_tag] || T.textMuted}15`,
            border: `1px solid ${catColor[p.category_tag] || T.textMuted}35`,
            borderRadius: 3, padding: "1px 5px",
            textTransform: "capitalize",
          }}>{p.category_tag}</span>
        )}
      </div>
    </div>
  );
}

function ScrapedProductCard({ product, selected, onToggle }) {
  return (
    <div onClick={onToggle} style={{
      background: T.surface, border: `1px solid ${selected ? T.accent : T.border}`,
      borderRadius: 10, overflow: "hidden", cursor: "pointer",
      transition: "border-color 0.2s, transform 0.15s",
      transform: selected ? "none" : "scale(0.97)", opacity: selected ? 1 : 0.6,
    }}>
      <div style={{ width: "100%", aspectRatio: "1/1", background: T.bgDeep, overflow: "hidden" }}>
        {product.image_url ? (
          <img src={product.image_url} alt={product.name}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
            onError={e => { e.target.style.display = "none"; }} />
        ) : (
          <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center",
            justifyContent: "center", color: T.textMuted, fontSize: "1.5rem" }}>📦</div>
        )}
      </div>
      <div style={{ padding: "8px 10px" }}>
        <div style={{ fontSize: "0.72rem", color: T.text, lineHeight: 1.35,
          overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box",
          WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>{product.name}</div>
        {product.price && <div style={{ fontSize: "0.68rem", color: T.accent, marginTop: 4 }}>{product.price}</div>}
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, sub, color }) {
  return (
    <div style={{
      background: T.surface, border: `1px solid ${T.border}`,
      borderRadius: 12, padding: "20px 22px",
    }}>
      <div style={{ fontSize: "1rem", marginBottom: 12, color: color }}>{icon}</div>
      <div style={{ fontSize: "1.5rem", fontFamily: "'Cormorant Garamond', serif",
        color: T.text, fontWeight: 500, lineHeight: 1 }}>
        {value}
      </div>
      {sub && <div style={{ fontSize: "0.68rem", color: T.textMuted, marginTop: 4 }}>{sub}</div>}
      <div style={{ fontSize: "0.68rem", color: T.textMuted, marginTop: 8,
        letterSpacing: "0.1em", textTransform: "uppercase" }}>{label}</div>
    </div>
  );
}

function Section({ title, action, children }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
        marginBottom: 16 }}>
        <div style={{ fontSize: "0.68rem", color: T.textMuted, letterSpacing: "0.14em",
          textTransform: "uppercase" }}>{title}</div>
        {action && (
          <button onClick={action.onClick} style={{
            background: "none", border: "none", cursor: "pointer",
            fontSize: "0.72rem", color: T.accent,
            fontFamily: "'DM Sans', sans-serif", padding: 0,
          }}>{action.label}</button>
        )}
      </div>
      {children}
    </div>
  );
}

function PageTitle({ label, sub }) {
  return (
    <div>
      <h2 style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "1.8rem",
        fontWeight: 500, color: T.text, margin: 0 }}>{label}</h2>
      {sub && <p style={{ marginTop: 6, fontSize: "0.78rem", color: T.textMuted }}>{sub}</p>}
    </div>
  );
}

function AuthInput({ label, type = "text", value, onChange, placeholder, required }) {
  return (
    <div>
      <label style={labelStyle}>{label}{required && <span style={{ color: T.red }}> *</span>}</label>
      <input type={type} value={value} onChange={e => onChange(e.target.value)}
        placeholder={placeholder} required={required}
        style={inputStyle} />
    </div>
  );
}

function InfoBox({ title, children }) {
  return (
    <div style={{ background: T.surface, border: `1px solid ${T.border}`,
      borderLeft: `3px solid ${T.accent}`, borderRadius: "0 10px 10px 0",
      padding: "14px 18px" }}>
      <div style={{ fontSize: "0.65rem", color: T.accent, letterSpacing: "0.14em",
        textTransform: "uppercase", marginBottom: 8 }}>{title}</div>
      {children}
    </div>
  );
}

function InfoRow({ label, value, mono }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ fontSize: "0.65rem", color: T.textMuted, letterSpacing: "0.1em",
        textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: mono ? "0.7rem" : "0.82rem",
        color: T.text, fontFamily: mono ? "'DM Mono', monospace" : "'DM Sans', sans-serif",
        wordBreak: "break-all" }}>{value}</div>
    </div>
  );
}

function PreviewTable({ rows, headers }) {
  const cols = headers.slice(0, 4);
  return (
    <div style={{ overflowX: "auto", borderRadius: 10, border: `1px solid ${T.border}` }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.75rem" }}>
        <thead>
          <tr style={{ background: T.surfaceHov }}>
            {cols.map(h => (
              <th key={h} style={{ padding: "10px 14px", textAlign: "left",
                color: T.textMuted, fontWeight: 500, letterSpacing: "0.08em",
                textTransform: "uppercase", fontSize: "0.65rem",
                borderBottom: `1px solid ${T.border}` }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${T.borderFaint}` }}>
              {cols.map(h => (
                <td key={h} style={{ padding: "9px 14px", color: T.text,
                  maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis",
                  whiteSpace: "nowrap" }}>{row[h] || "—"}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function IndexingProgress({ progress }) {
  const pct = progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;
  return (
    <div style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 12, padding: "24px 28px" }}>
      <div style={{ fontSize: "0.68rem", color: T.accent, letterSpacing: "0.14em",
        textTransform: "uppercase", marginBottom: 16 }}>Indexing in progress</div>
      <div style={{ height: 2, background: T.border, borderRadius: 2, marginBottom: 12, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct}%`,
          background: `linear-gradient(90deg, ${T.accentDeep}, ${T.accent})`,
          borderRadius: 2, transition: "width 0.4s ease" }} />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 14 }}>
        <div style={{ fontSize: "0.72rem", color: T.textMuted }}>
          {progress.current && <span>Processing: <span style={{ color: T.text }}>{progress.current}</span></span>}
        </div>
        <div style={{ fontSize: "0.72rem", color: T.textMuted }}>{progress.done} / {progress.total}</div>
      </div>
      <div style={{ display: "flex", gap: 20 }}>
        <div style={{ fontSize: "0.78rem" }}><span style={{ color: T.green }}>✓ {progress.success}</span><span style={{ color: T.textMuted }}> indexed</span></div>
        {progress.failed > 0 && <div style={{ fontSize: "0.78rem" }}><span style={{ color: T.red }}>✕ {progress.failed}</span><span style={{ color: T.textMuted }}> failed</span></div>}
      </div>
    </div>
  );
}

function DoneCard({ progress, errors, storeName }) {
  const ok = progress.success > 0;
  return (
    <div style={{
      background: ok ? "rgba(122,171,138,0.05)" : "rgba(201,112,112,0.05)",
      border: `1px solid ${ok ? T.green : T.red}`,
      borderRadius: 12, padding: "24px 28px",
    }}>
      <div style={{ fontFamily: "'Cormorant Garamond', serif", fontSize: "1.4rem",
        color: ok ? T.green : T.red, marginBottom: 6 }}>
        {ok ? "Done!" : "Indexing failed"}
      </div>
      <div style={{ fontSize: "0.82rem", color: T.textMuted, marginBottom: 16 }}>
        {ok && <><span style={{ color: T.green }}>{progress.success} products</span> from{" "}
          <strong style={{ color: T.text }}>{storeName}</strong> are now searchable in Locus.</>}
        {progress.failed > 0 && <> <span style={{ color: T.red }}>{progress.failed} failed.</span></>}
      </div>
      {errors.length > 0 && (
        <details>
          <summary style={{ fontSize: "0.72rem", color: T.textMuted, cursor: "pointer", marginBottom: 8 }}>
            Show {errors.length} errors
          </summary>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {errors.map((e, i) => (
              <div key={i} style={{ fontSize: "0.68rem", color: T.red, fontFamily: "'DM Mono', monospace" }}>{e}</div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function LoadingState({ label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center",
      padding: "60px 0", gap: 12, color: T.textMuted, fontSize: "0.82rem" }}>
      <Spinner /> {label}
    </div>
  );
}

function EmptyState({ icon, title, body, action }) {
  return (
    <div style={{ background: T.surface, border: `1px solid ${T.border}`,
      borderRadius: 12, padding: "60px 24px", textAlign: "center" }}>
      <div style={{ fontSize: "2rem", marginBottom: 14 }}>{icon}</div>
      <div style={{ fontSize: "0.9rem", color: T.text, fontWeight: 500, marginBottom: 8 }}>{title}</div>
      <div style={{ fontSize: "0.78rem", color: T.textMuted, marginBottom: action ? 20 : 0 }}>{body}</div>
      {action && (
        <button onClick={action.onClick} style={{
          background: T.accentBg, border: `1px solid ${T.accentRing}`,
          borderRadius: 8, padding: "9px 20px", cursor: "pointer",
          fontSize: "0.78rem", color: T.accent,
          fontFamily: "'DM Sans', sans-serif",
        }}>{action.label}</button>
      )}
    </div>
  );
}

function ErrorBanner({ msg }) {
  return (
    <div style={{ background: "rgba(201,112,112,0.06)", border: `1px solid rgba(201,112,112,0.2)`,
      borderRadius: 8, padding: "10px 14px", fontSize: "0.78rem", color: T.red }}>
      ⚠ {msg}
    </div>
  );
}

function Spinner({ small }) {
  return (
    <div style={{
      width: small ? 12 : 16, height: small ? 12 : 16,
      border: `1.5px solid ${T.border}`,
      borderTop: `1.5px solid ${T.accent}`,
      borderRadius: "50%", animation: "spin 0.9s linear infinite",
      display: "inline-block", flexShrink: 0,
    }} />
  );
}

// ── Hook: stable category color map ───────────────────────────────────────────
function useCategoryColors() {
  return {
    shirt: "#6e9ecf", sweater: "#9b8abf", jacket: "#7aab8a", coat: "#7aab8a",
    dress: "#c97070", jumpsuit: "#c9a96e", skirt: "#c9c06e", pants: "#8aabcf",
    shorts: "#8aabcf", shoes: "#c9a96e", bag: "#bf9a7a", glasses: "#aaaaaa",
    hat: "#aaaaaa", watch: "#c9c96e", scarf: "#bf9a7a",
  };
}

// ── Shared styles ──────────────────────────────────────────────────────────────
const labelStyle = {
  fontSize: "0.65rem", color: "#6b6458",
  letterSpacing: "0.12em", textTransform: "uppercase",
  display: "block", marginBottom: 8,
};

const inputStyle = {
  width: "100%",
  background: "#040404",
  border: "1px solid #222222",
  borderRadius: 8,
  padding: "10px 14px",
  color: "#f0ede8",
  fontSize: "0.82rem",
  fontFamily: "'DM Sans', sans-serif",
  outline: "none",
  boxSizing: "border-box",
};
