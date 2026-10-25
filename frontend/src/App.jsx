import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import StoreDashboardView from "./StoreDashboardView";

// ── Fix Leaflet marker icons broken by Vite ────────────────────────
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  iconUrl:       "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  shadowUrl:     "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
});

const API = "";

const T = {
  bg:          "#FAF9F7",
  bgDeep:      "#F0EBE5",
  surface:     "#FFFFFF",
  surfaceHov:  "#FAFAF8",
  border:      "rgba(30,20,12,0.12)",
  borderFaint: "rgba(30,20,12,0.06)",
  text:        "#1C1714",
  textMuted:   "rgba(28,23,20,0.48)",
  textFaint:   "rgba(28,23,20,0.22)",
  accent:      "#8B6F5E",
  accentBg:    "rgba(139,111,94,0.08)",
  accentRing:  "rgba(139,111,94,0.22)",
  accentDeep:  "#6A4F40",
  green:       "#5A8A6A",
  yellow:      "#C49A3C",
  red:         "#B85858",
};

const GLOBAL_CSS = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html, body, #root {
    height: 100%;
    width: 100%;
    background: ${T.bg};
    color: ${T.text};
    font-family: 'Josefin Sans', sans-serif;
    font-size: 16px;
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
  }

  ::-webkit-scrollbar { width: 3px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(60,60,67,0.2); border-radius: 2px; }

  .leaflet-container {
    background: ${T.bgDeep} !important;
    font-family: 'Josefin Sans', sans-serif !important;
  }
  .leaflet-popup-content-wrapper {
    background: ${T.surface} !important;
    color: ${T.text} !important;
    border: 1px solid ${T.border} !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12) !important;
    font-family: 'Josefin Sans', sans-serif !important;
    font-size: 0.85rem !important;
  }
  .leaflet-popup-tip { background: ${T.surface} !important; }
  .leaflet-popup-close-button { color: ${T.textMuted} !important; top: 8px !important; right: 8px !important; }
  .leaflet-control-zoom a {
    background: ${T.surface} !important;
    color: ${T.textMuted} !important;
    border-color: ${T.border} !important;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  @keyframes slideUp {
    from { transform: translateY(100%); }
    to   { transform: translateY(0); }
  }

  @keyframes clothFloat {
    0%   { transform: translateY(0vh)    rotate(-7deg) scale(1); }
    45%  { transform: translateY(-52vh)  rotate(5deg)  scale(1.04); }
    100% { transform: translateY(-112vh) rotate(-4deg) scale(0.96); }
  }
  @keyframes clothFloatR {
    0%   { transform: translateY(0vh)    rotate(7deg)  scale(1); }
    45%  { transform: translateY(-52vh)  rotate(-5deg) scale(1.04); }
    100% { transform: translateY(-112vh) rotate(3deg)  scale(0.96); }
  }
  @keyframes blob1 {
    0%, 100% { transform: translate(0px, 0px) scale(1); }
    25%  { transform: translate(22px, -28px) scale(1.05); }
    50%  { transform: translate(-16px, 14px) scale(0.96); }
    75%  { transform: translate(10px, 24px) scale(1.03); }
  }
  @keyframes blob2 {
    0%, 100% { transform: translate(0px, 0px) scale(1); }
    30%  { transform: translate(-24px, 18px) scale(1.07); }
    60%  { transform: translate(20px, -12px) scale(0.94); }
  }

  .fade-up { animation: fadeUp 0.4s cubic-bezier(0.16,1,0.3,1) forwards; }
  .fade-in { animation: fadeIn 0.3s ease forwards; }

  @keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position:  400px 0; }
  }
  .shimmer-box {
    background: linear-gradient(90deg, ${T.bgDeep} 25%, ${T.border} 50%, ${T.bgDeep} 75%);
    background-size: 800px 100%;
    animation: shimmer 1.4s infinite linear;
  }

  .discover-scroll {
    display: flex;
    gap: 10px;
    overflow-x: auto;
    scroll-snap-type: x mandatory;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
    -ms-overflow-style: none;
    padding-bottom: 4px;
  }
  .discover-scroll::-webkit-scrollbar { display: none; }

  .font-serif {
    font-family: 'Cormorant Garamond', 'Playfair Display', Georgia, serif;
  }

  .font-logo {
    font-family: 'Bodoni Moda', 'Didot', 'Bodoni MT', Georgia, serif;
    font-style: italic;
  }

  .results-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
  }
  @media (min-width: 560px)  { .results-grid { grid-template-columns: repeat(3, 1fr); gap: 16px; } }
  @media (min-width: 1024px) { .results-grid { grid-template-columns: repeat(4, 1fr); gap: 18px; } }

  .upload-zone {
    border: 1.5px dashed ${T.border};
    border-radius: 12px;
    background: transparent;
    transition: border-color 0.2s, background 0.2s;
    cursor: pointer;
  }
  .upload-zone:hover, .upload-zone.drag-over {
    border-color: ${T.accent};
    background: ${T.accentBg};
  }

  .btn-primary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 15px 28px;
    background: ${T.text};
    color: ${T.bg};
    border: none;
    border-radius: 12px;
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.92rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.12s;
    -webkit-tap-highlight-color: transparent;
  }
  .btn-primary:hover { opacity: 0.80; }
  .btn-primary:active { transform: scale(0.97); }

  .btn-secondary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 14px 28px;
    background: transparent;
    color: ${T.text};
    border: 1px solid ${T.border};
    border-radius: 10px;
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0.02em;
    cursor: pointer;
    transition: background 0.15s;
    -webkit-tap-highlight-color: transparent;
  }
  .btn-secondary:hover { background: ${T.bgDeep}; }

  .btn-ghost {
    background: none;
    border: none;
    color: ${T.textMuted};
    cursor: pointer;
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 400;
    padding: 8px 14px;
    border-radius: 8px;
    transition: background 0.15s, color 0.15s;
    -webkit-tap-highlight-color: transparent;
  }
  .btn-ghost:hover { background: ${T.bgDeep}; color: ${T.text}; }

  .star-btn {
    background: none;
    border: none;
    padding: 2px;
    cursor: pointer;
    font-size: 1rem;
    line-height: 1;
    transition: transform 0.1s;
    color: rgba(60,60,67,0.22);
    -webkit-tap-highlight-color: transparent;
  }
  .star-btn:hover:not(:disabled) { transform: scale(1.2); }
  .star-btn:disabled { cursor: default; }
  .star-btn.filled { color: ${T.yellow}; }
  .star-btn.dimmed { color: rgba(60,60,67,0.1); }

  input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    height: 4px;
    background: rgba(60,60,67,0.15);
    border-radius: 2px;
    outline: none;
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 20px; height: 20px;
    border-radius: 50%;
    background: #FFFFFF;
    cursor: pointer;
    box-shadow: 0 1px 6px rgba(0,0,0,0.2);
    border: 2px solid ${T.accent};
  }

  select {
    background: ${T.surface};
    color: ${T.text};
    border: 1px solid ${T.border};
    border-radius: 10px;
    font-family: 'Josefin Sans', sans-serif;
  }
`;

function StyleInjector() {
  useEffect(() => {
    const el = document.createElement("style");
    el.textContent = GLOBAL_CSS;
    document.head.appendChild(el);
    return () => document.head.removeChild(el);
  }, []);
  return null;
}

function MapRecenter({ center }) {
  const map = useMap();
  useEffect(() => { map.setView(center, map.getZoom()); }, [center, map]);
  return null;
}

const makeStoreIcon = (highlighted) => L.divIcon({
  html: `<div style="
    width:22px; height:22px;
    background:${highlighted ? T.accent : "#6b6458"};
    border-radius:50% 50% 50% 0;
    transform:rotate(-45deg);
    border:2px solid ${highlighted ? "#fff" : T.border};
    box-shadow:0 2px 10px rgba(0,0,0,0.5);
    transition:all 0.2s;
  "></div>`,
  className: "",
  iconSize: [22, 22],
  iconAnchor: [11, 22],
  popupAnchor: [0, -24],
});

function haversineKm([lat1, lon1], [lat2, lon2]) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ══════════════════════════════════════════════════════════════════
// HISTORY PERSISTENCE
// ══════════════════════════════════════════════════════════════════
const HISTORY_KEY = "locus_history";
const HISTORY_MAX = 20;

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function clearHistory() {
  localStorage.removeItem(HISTORY_KEY);
}

function saveHistoryEntry(category, cropBlob, results) {
  return new Promise((resolve) => {
    const finish = (cropImageBase64) => {
      const entry = {
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        category,
        cropImageBase64,
        results: results.slice(0, 10),
      };
      const existing = loadHistory();
      const updated = [entry, ...existing].slice(0, HISTORY_MAX);
      localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
      resolve(updated);
    };

    if (!cropBlob) { finish(null); return; }

    // Resize to 200×200 thumbnail before base64-encoding
    const url = URL.createObjectURL(cropBlob);
    const img = new Image();
    img.onload = () => {
      const MAX = 200;
      const scale = Math.min(MAX / img.width, MAX / img.height, 1);
      const canvas = document.createElement("canvas");
      canvas.width  = Math.round(img.width  * scale);
      canvas.height = Math.round(img.height * scale);
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      finish(canvas.toDataURL("image/jpeg", 0.75));
    };
    img.onerror = () => { URL.revokeObjectURL(url); finish(null); };
    img.src = url;
  });
}

function timeAgo(isoString) {
  const diff = Date.now() - new Date(isoString).getTime();
  const mins  = Math.floor(diff / 60000);
  if (mins < 1)   return "just now";
  if (mins < 60)  return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs  < 24)  return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7)   return `${days}d ago`;
  return new Date(isoString).toLocaleDateString();
}

// ══════════════════════════════════════════════════════════════════
// SAVED PERSISTENCE
// ══════════════════════════════════════════════════════════════════
const SAVED_KEY = "locus_saved";

function loadSaved() {
  try {
    return JSON.parse(localStorage.getItem(SAVED_KEY) || "[]");
  } catch {
    return [];
  }
}

function isItemSaved(productId, saved) {
  return saved.some(s => s.productId === productId);
}

function toggleSavedItem(result, saved) {
  const payload    = result.payload ?? {};
  const productId  = payload.product_id || payload.item_id || payload.image_url;
  const alreadySaved = isItemSaved(productId, saved);
  let updated;
  if (alreadySaved) {
    updated = saved.filter(s => s.productId !== productId);
  } else {
    updated = [{ productId, savedAt: new Date().toISOString(), result }, ...saved];
  }
  localStorage.setItem(SAVED_KEY, JSON.stringify(updated));
  return updated;
}

// ══════════════════════════════════════════════════════════════════
// NAVBAR
// ══════════════════════════════════════════════════════════════════
// ── iOS bottom tab bar ────────────────────────────────────────────
const TAB_ICONS = {
  Discover: (active) => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
      stroke={active ? T.accent : T.textMuted} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"/>
      <line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
  ),
  Saved: (active) => (
    <svg width="24" height="24" viewBox="0 0 24 24"
      fill={active ? T.accent : "none"}
      stroke={active ? T.accent : T.textMuted} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
    </svg>
  ),
  History: (active) => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
      stroke={active ? T.accent : T.textMuted} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <polyline points="12 6 12 12 16 14"/>
    </svg>
  ),
  Store: (active) => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
      stroke={active ? T.accent : T.textMuted} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"/>
      <line x1="3" y1="6" x2="21" y2="6"/>
      <path d="M16 10a4 4 0 0 1-8 0"/>
    </svg>
  ),
};

function Navbar({ activeTab, onTab }) {
  return (
    <nav style={{
      position: "fixed", bottom: 0, left: 0, right: 0, zIndex: 200,
      background: "rgba(249,249,249,0.94)",
      backdropFilter: "saturate(180%) blur(20px)",
      WebkitBackdropFilter: "saturate(180%) blur(20px)",
      borderTop: `1px solid ${T.border}`,
      display: "flex",
      paddingBottom: "env(safe-area-inset-bottom, 0px)",
    }}>
      {["Discover", "Saved", "Store"].map(tab => {
        const active = activeTab === tab;
        const label = tab === "Store" ? "Stores" : tab;
        return (
          <button
            key={tab}
            onClick={() => onTab(tab)}
            style={{
              flex: 1, padding: "10px 0 12px",
              background: "none", border: "none", cursor: "pointer",
              display: "flex", flexDirection: "column", alignItems: "center", gap: 3,
              color: active ? T.accent : T.textMuted,
              transition: "color 0.15s",
              WebkitTapHighlightColor: "transparent",
            }}
          >
            {TAB_ICONS[tab](active)}
            <span style={{ fontSize: "0.6rem", fontWeight: active ? 600 : 400, letterSpacing: "0.06em", textTransform: "uppercase" }}>
              {label}
            </span>
          </button>
        );
      })}
    </nav>
  );
}

// ══════════════════════════════════════════════════════════════════
// FLOATING CLOTHES BACKGROUND
// ══════════════════════════════════════════════════════════════════
const CLOTH_EMOJIS = [
  "👗","🧥","👜","👠","👒","👔","👗","🧣","👟","🕶️",
  "🧤","👗","👜","🧥","👠","👗","👒","🧥","👗","👟",
];

function LocusLogo({ layout = "stacked", markSize = 48, fontSize = "1rem", color, accent, style }) {
  const c = color ?? T.text;
  const a = accent ?? T.accent;
  const sw = 1.2;

  const mark = (
    <svg width={markSize} height={markSize} viewBox="0 0 48 48" fill="none">
      <circle cx="24" cy="24" r="15" stroke={a} strokeWidth={sw}/>
      <circle cx="24" cy="24" r="3" fill={a}/>
      <line x1="24" y1="4"  x2="24" y2="10" stroke={a} strokeWidth={sw} strokeLinecap="round"/>
      <line x1="24" y1="38" x2="24" y2="44" stroke={a} strokeWidth={sw} strokeLinecap="round"/>
      <line x1="4"  y1="24" x2="10" y2="24" stroke={a} strokeWidth={sw} strokeLinecap="round"/>
      <line x1="38" y1="24" x2="44" y2="24" stroke={a} strokeWidth={sw} strokeLinecap="round"/>
    </svg>
  );

  const wordmark = (
    <span style={{
      fontFamily: "'Josefin Sans', sans-serif",
      fontSize, fontWeight: 300, color: c,
      letterSpacing: "0.4em",
      textTransform: "uppercase",
      paddingRight: "0.4em",
    }}>locus</span>
  );

  if (layout === "stacked") {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16, ...style }}>
        {mark}
        {wordmark}
      </div>
    );
  }

  return (
    <div style={{ display: "inline-flex", alignItems: "center", gap: 10, ...style }}>
      {mark}
      {wordmark}
    </div>
  );
}

function FloatingClothes({ opacity = 1 }) {
  const items = useMemo(() => CLOTH_EMOJIS.map((emoji, i) => {
    const total = CLOTH_EMOJIS.length;
    const slotW = 100 / total;
    const jitter = (i % 3 - 1) * slotW * 0.35;
    return {
      id: i,
      emoji,
      left: `${Math.max(1, Math.min(96, slotW * i + slotW * 0.4 + jitter))}%`,
      fontSize: `${1.5 + (i % 5) * 0.22}rem`,
      duration: `${22 + (i % 6) * 3.5}s`,
      delay: `${-((i / total) * (22 + (i % 6) * 3.5))}s`,
      itemOpacity: (0.11 + (i % 4) * 0.022) * opacity,
      anim: i % 2 === 0 ? "clothFloat" : "clothFloatR",
    };
  }), [opacity]);

  return (
    <div style={{
      position: "absolute", inset: 0,
      overflow: "hidden", pointerEvents: "none", zIndex: 0,
    }}>
      {items.map(item => (
        <span
          key={item.id}
          style={{
            position: "absolute",
            left: item.left,
            bottom: "-6%",
            fontSize: item.fontSize,
            opacity: item.itemOpacity,
            animation: `${item.anim} ${item.duration} linear ${item.delay} infinite`,
            filter: "brightness(1.05) saturate(1.1)",
            userSelect: "none",
            lineHeight: 1,
            display: "block",
            willChange: "transform",
          }}
        >
          {item.emoji}
        </span>
      ))}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// DISCOVER FEED  — horizontal strips on the home screen
// ══════════════════════════════════════════════════════════════════
function DiscoverCard({ item, isSaved, onToggleSave, onCardClick, onFeedback }) {
  const [hovered, setHovered] = useState(false);
  const [cardRating, setCardRating] = useState(null);
  const raw      = item.image_url ?? "";
  const imageURL = raw ? (raw.startsWith("http") ? raw : `${API}${raw}`) : null;
  const catColor = CATEGORY_COLORS[item.category] || T.accent;

  return (
    <div
      style={{
        flex: "0 0 130px",
        background: T.surface,
        border: `1px solid ${hovered ? T.accent : T.border}`,
        borderRadius: 12,
        overflow: "hidden",
        scrollSnapAlign: "start",
        transition: "border-color 0.2s, transform 0.22s cubic-bezier(0.16,1,0.3,1), box-shadow 0.22s",
        transform: hovered ? "translateY(-3px)" : "none",
        boxShadow: hovered ? "0 8px 24px rgba(0,0,0,0.08)" : "none",
        cursor: "pointer",
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={() => onCardClick?.(item)}
    >
      <div style={{ aspectRatio: "3/4", background: T.bgDeep, position: "relative", overflow: "hidden" }}>
        {imageURL
          ? <img
              src={imageURL}
              alt={item.name || "product"}
              style={{ width: "100%", height: "100%", objectFit: "cover", transition: "transform 0.5s ease", transform: hovered ? "scale(1.05)" : "scale(1)" }}
            />
          : <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: T.textFaint, fontSize: "1.2rem" }}>✦</div>
        }
        <button
          onClick={(e) => { e.stopPropagation(); onToggleSave?.(item); }}
          style={{
            position: "absolute", bottom: 6, right: 6,
            background: isSaved ? T.accent : "rgba(250,249,247,0.88)",
            backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
            border: "none", borderRadius: "50%",
            width: 26, height: 26, cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "0.72rem", color: isSaved ? "#fff" : T.textMuted,
            transition: "background 0.2s, color 0.2s",
          }}
        >
          {isSaved ? "♥" : "♡"}
        </button>
      </div>
      <div style={{ padding: "8px 10px 10px" }}>
        <div style={{ height: 2, borderRadius: 1, background: `${catColor}40`, marginBottom: 5 }} />
        <div style={{
          fontSize: "0.7rem", fontWeight: 500, color: T.text, lineHeight: 1.3,
          overflow: "hidden", display: "-webkit-box",
          WebkitLineClamp: 2, WebkitBoxOrient: "vertical",
        }}>
          {item.name || "Product"}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 4 }}>
          <span style={{ fontSize: "0.58rem", color: T.textMuted, letterSpacing: "0.01em" }}>{item.store || ""}</span>
          {item.price && <span style={{ fontSize: "0.65rem", color: T.accent, fontWeight: 600 }}>{item.price}</span>}
        </div>
        {onFeedback && (
          <div style={{ display: "flex", gap: 2, marginTop: 6 }} onClick={(e) => e.stopPropagation()}>
            {[1, 2, 3, 4, 5].map(star => (
              <button
                key={star}
                onClick={(e) => {
                  e.stopPropagation();
                  setCardRating(star);
                  onFeedback(item.product_id, star, item.name, item.store, item.category, item.image_url);
                }}
                style={{
                  background: "none", border: "none", cursor: "pointer", padding: 0,
                  fontSize: "0.75rem", lineHeight: 1,
                  color: star <= (cardRating ?? 0) ? T.yellow : T.textFaint,
                  transition: "color 0.15s",
                }}
                title={`Rate ${star} star${star > 1 ? "s" : ""}`}
              >★</button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function DiscoverSection({ title, subtitle, badge, items, saved, onToggleSave, onCardClick, onFeedback, loading }) {
  const isItemSaved = (pid) => saved.some(s => s.productId === pid);

  const wrapDiscover = (item) => ({
    score: 0,
    payload: {
      product_id:   item.product_id,
      name:         item.name,
      item_name:    item.name,
      image_url:    item.image_url,
      store_name:   item.store,
      store:        item.store,
      mall_name:    item.mall,
      price:        item.price,
      category_tag: item.category,
    },
  });

  if (!loading && items.length === 0) return null;

  return (
    <div style={{ marginBottom: 32 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, paddingLeft: 24, paddingRight: 24 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 2 }}>
            <span style={{ fontSize: "0.55rem", letterSpacing: "0.18em", textTransform: "uppercase", color: T.textMuted }}>{subtitle}</span>
            {badge && (
              <span style={{
                fontSize: "0.48rem", letterSpacing: "0.1em", textTransform: "uppercase",
                background: T.accentBg, color: T.accent, borderRadius: 20,
                padding: "2px 7px", border: `1px solid ${T.accentRing}`,
              }}>
                {badge}
              </span>
            )}
          </div>
          <div className="font-serif" style={{ fontSize: "1.1rem", fontWeight: 500, color: T.text, fontStyle: "italic" }}>
            {title}
          </div>
        </div>
      </div>

      {loading ? (
        <div className="discover-scroll" style={{ paddingLeft: 24, paddingRight: 24 }}>
          {[1, 2, 3, 4].map(i => (
            <div key={i} style={{ flex: "0 0 130px", scrollSnapAlign: "start" }}>
              <div className="shimmer-box" style={{ borderRadius: 12, aspectRatio: "3/4" }} />
              <div className="shimmer-box" style={{ borderRadius: 6, height: 10, marginTop: 8, width: "80%" }} />
              <div className="shimmer-box" style={{ borderRadius: 6, height: 8, marginTop: 5, width: "50%" }} />
            </div>
          ))}
        </div>
      ) : (
        <div className="discover-scroll" style={{ paddingLeft: 24, paddingRight: 24 }}>
          {items.map((item) => (
            <DiscoverCard
              key={item.product_id}
              item={item}
              isSaved={isItemSaved(item.product_id)}
              onToggleSave={() => onToggleSave?.(wrapDiscover(item))}
              onCardClick={onCardClick}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// LANDING
// ══════════════════════════════════════════════════════════════════
const LANDING_CATEGORIES = ["Dress", "Jacket", "Bag", "Shoes", "Top", "Pants", "Sweater", "Hat"];

function LandingView({ onUpload, error, history, saved, onToggleSave, onCardClick, onFeedback }) {
  const inputRef  = useRef(null);
  const cameraRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [discover, setDiscover] = useState(null); // null = loading

  const handleFile = (file) => {
    if (file && file.type.startsWith("image/")) onUpload(file);
  };

  // Fetch discover feed once on mount
  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/discover?limit=15`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (!cancelled) setDiscover(data || { trending: [], women: [], men: [] }); })
      .catch(() => { if (!cancelled) setDiscover({ trending: [], women: [], men: [] }); });
    return () => { cancelled = true; };
  }, []);

  // Derive "For You" from search history — pull top results across recent searches
  const forYou = useMemo(() => {
    if (!history.length) return [];
    const pool = [];
    const seen = new Set();
    for (const entry of history.slice(0, 8)) {
      for (const r of (entry.results || []).slice(0, 3)) {
        const p   = r.payload ?? {};
        const pid = p.product_id || p.item_id || p.image_url;
        if (!pid || seen.has(pid) || !p.image_url) continue;
        seen.add(pid);
        pool.push({
          product_id: pid,
          name:       p.name || p.item_name || "",
          price:      p.price || "",
          category:   p.category_tag || entry.category || "",
          image_url:  p.image_url,
          store:      p.store_name || p.store || "",
          mall:       p.mall_name || p.mall || "",
        });
      }
    }
    return pool.slice(0, 15);
  }, [history]);

  // Rank items by how often they appear across all search results
  const mostSearched = useMemo(() => {
    if (!history.length) return [];
    const counts = new Map();
    for (const entry of history) {
      for (const r of (entry.results || [])) {
        const p   = r.payload ?? {};
        const pid = p.product_id || p.item_id || p.image_url;
        if (!pid || !p.image_url) continue;
        if (counts.has(pid)) {
          counts.get(pid).count++;
        } else {
          counts.set(pid, {
            count: 1,
            item: {
              product_id: pid,
              name:       p.name || p.item_name || "",
              price:      p.price || "",
              category:   p.category_tag || entry.category || "",
              image_url:  p.image_url,
              store:      p.store_name || p.store || "",
              mall:       p.mall_name || p.mall || "",
            },
          });
        }
      }
    }
    return Array.from(counts.values())
      .sort((a, b) => b.count - a.count)
      .slice(0, 15)
      .map(({ item }) => item);
  }, [history]);

  const loading = discover === null;

  return (
    <div className="fade-in" style={{
      minHeight: "100svh",
      background: T.bg,
      paddingBottom: "calc(83px + env(safe-area-inset-bottom, 0px) + 32px)",
    }}>
      {/* Hero section — upload + branding */}
      <div style={{
        position: "relative",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        padding: "28px 28px 20px",
        overflow: "hidden",
      }}>
        <FloatingClothes opacity={1} />

        {/* Editorial branding */}
        <div className="fade-up" style={{ textAlign: "center", marginBottom: 20, position: "relative", zIndex: 1 }}>
          <LocusLogo layout="stacked" markSize={48} fontSize="2rem" style={{ marginBottom: 10 }} />
          <p style={{ fontSize: "0.82rem", color: T.textMuted, lineHeight: 1.6, maxWidth: 220, margin: "0 auto" }}>
            Photograph any piece. Find it in stores near you.
          </p>
        </div>

        {/* Upload actions */}
        <div className="fade-up" style={{ width: "100%", maxWidth: 320, display: "flex", flexDirection: "row", gap: 8, position: "relative", zIndex: 1 }}>
          <button className="btn-primary" style={{ flex: 1, height: 42, fontSize: "0.85rem" }} onClick={() => cameraRef.current?.click()}>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
            Take photo
          </button>

          <div
            className={`upload-zone ${dragging ? "drag-over" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
            style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 7, padding: "10px 12px", cursor: "pointer" }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={T.textMuted} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2"/>
              <circle cx="8.5" cy="8.5" r="1.5"/>
              <polyline points="21 15 16 10 5 21"/>
            </svg>
            <span style={{ fontSize: "0.82rem", color: T.textMuted, fontWeight: 400 }}>
              {dragging ? "Drop here" : "Library"}
            </span>
          </div>

          <input ref={cameraRef} type="file" accept="image/*" capture="environment" style={{ display: "none" }}
            onChange={(e) => { if (e.target.files?.[0]) { handleFile(e.target.files[0]); e.target.value = null; } }} />
          <input ref={inputRef} type="file" accept="image/jpeg, image/png, image/webp, image/heic, image/*" style={{ display: "none" }}
            onChange={(e) => { if (e.target.files?.[0]) { handleFile(e.target.files[0]); e.target.value = null; } }} />
        </div>


        {error && (
          <div className="fade-up" style={{
            marginTop: 20, padding: "14px 18px",
            background: "rgba(184,88,88,0.08)", border: "1px solid rgba(184,88,88,0.2)",
            borderRadius: 10, fontSize: "0.9rem", color: T.red,
            maxWidth: 340, width: "100%", textAlign: "center", position: "relative", zIndex: 1,
          }}>
            {error}
          </div>
        )}

      </div>

      {/* ── Discover sections ─────────────────────────────────── */}
      <div style={{ paddingTop: 4 }}>

        {/* Divider */}
        <div style={{ display: "flex", alignItems: "center", gap: 14, paddingLeft: 24, paddingRight: 24, marginBottom: 20 }}>
          <div style={{ flex: 1, height: "1px", background: T.borderFaint }} />
          <span style={{ fontSize: "0.48rem", letterSpacing: "0.2em", textTransform: "uppercase", color: T.textFaint, whiteSpace: "nowrap" }}>
            what's happening
          </span>
          <div style={{ flex: 1, height: "1px", background: T.borderFaint }} />
        </div>

        {/* Trending Now */}
        <DiscoverSection
          title="Trending Now"
          subtitle="This week"
          badge="live"
          items={discover?.trending ?? []}
          saved={saved}
          onToggleSave={onToggleSave}
          onCardClick={onCardClick}
          onFeedback={onFeedback}
          loading={loading}
        />

        {/* For You — only show when user has history */}
        {(forYou.length > 0 || loading) && (
          <DiscoverSection
            title="For You"
            subtitle="Based on your searches"
            items={forYou}
            saved={saved}
            onToggleSave={onToggleSave}
            onCardClick={onCardClick}
            onFeedback={onFeedback}
            loading={false}
          />
        )}

        {/* Most Searched — only show when user has sufficient history */}
        {mostSearched.length > 0 && (
          <DiscoverSection
            title="Most Searched"
            subtitle="Top picks from your searches"
            badge="popular"
            items={mostSearched}
            saved={saved}
            onToggleSave={onToggleSave}
            onCardClick={onCardClick}
            onFeedback={onFeedback}
            loading={false}
          />
        )}


      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// LOADING
// ══════════════════════════════════════════════════════════════════
function LoadingView({ label }) {
  return (
    <div style={{
      minHeight: "100svh", display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      paddingBottom: "calc(83px + env(safe-area-inset-bottom, 0px))",
      background: T.bg,
      position: "relative",
      overflow: "hidden",
    }}>
      <FloatingClothes opacity={0.6} />
      <div style={{ textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center", gap: 18, position: "relative", zIndex: 1 }}>
        <LocusLogo layout="horizontal" markSize={22} fontSize="0.95rem" style={{ opacity: 0.7 }} />
        <div style={{ width: 28, height: 28, border: `2px solid ${T.borderFaint}`, borderTop: `2px solid ${T.accent}`, borderRadius: "50%", animation: "spin 0.9s linear infinite" }} />
        <span style={{ fontSize: "0.72rem", color: T.textMuted, letterSpacing: "0.1em", textTransform: "uppercase" }}>{label}</span>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// CATEGORY LABELS  (mirrors clip_labels.py CANONICAL_LABELS)
// ══════════════════════════════════════════════════════════════════
const CATEGORY_LABELS = {
  top:        "Top",
  sports_bra: "Sports Bra",
  pants:      "Pants",
  leggings:   "Leggings",
  shorts:     "Shorts",
  skirt:      "Skirt",
  dress:      "Dress",
  sweater:    "Sweater",
  jacket:     "Jacket",
  shoes:      "Shoes",
  hat:        "Hat",
  bag:        "Bag",
  jumpsuit:   "Jumpsuit",
};

// ══════════════════════════════════════════════════════════════════
// DRAW VIEW  — user drags a bbox around the item they want to find
// ══════════════════════════════════════════════════════════════════
function DrawView({ imageURL, imageFile, onConfirm, onBack }) {
  const containerRef = useRef(null);
  const imgRef       = useRef(null);

  const [imgNatural,  setImgNatural]  = useState({ w: 1, h: 1 });
  const [imgRendered, setImgRendered] = useState({ w: 0, h: 0, x: 0, y: 0 });
  const [drawing,     setDrawing]     = useState(false);
  const [startPt,     setStartPt]     = useState(null);
  const [box,         setBox]         = useState(null); // rendered coords {x1,y1,x2,y2}
  const [resizing,    setResizing]    = useState(null); // null | "tl" | "tr" | "bl" | "br"
  const [loading,     setLoading]     = useState(false);
  const [showTutorial, setShowTutorial] = useState(true);
  const TUTORIAL_VIDEO = "/draw-hint.mp4"; // drop your video here: frontend/public/draw-hint.mp4

  const measureRendered = (el) => {
    const { width: cw, height: ch } = el.getBoundingClientRect();
    const s = Math.min(cw / el.naturalWidth, ch / el.naturalHeight);
    setImgRendered({ w: el.naturalWidth * s, h: el.naturalHeight * s, x: (cw - el.naturalWidth * s) / 2, y: (ch - el.naturalHeight * s) / 2 });
  };

  const onImgLoad = (e) => {
    const el = e.target;
    setImgNatural({ w: el.naturalWidth, h: el.naturalHeight });
    measureRendered(el);
  };

  useEffect(() => {
    const onResize = () => { if (imgRef.current?.naturalWidth) measureRendered(imgRef.current); };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Returns pointer position clamped to image bounds, relative to container
  const getPoint = (e) => {
    if (!containerRef.current) return { x: 0, y: 0 };
    const rect  = containerRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: Math.max(imgRendered.x, Math.min(imgRendered.x + imgRendered.w, clientX - rect.left)),
      y: Math.max(imgRendered.y, Math.min(imgRendered.y + imgRendered.h, clientY - rect.top)),
    };
  };

  const onPointerDown = (e) => {
    if (loading || imgRendered.w === 0) return;
    // If a box is already drawn, ignore new touches on the container so the
    // user can scroll freely. They must press "Redraw" to start over.
    const boxIsSet = box && (box.x2 - box.x1) >= 10 && (box.y2 - box.y1) >= 10;
    if (boxIsSet) return;
    e.preventDefault();
    const pt = getPoint(e);
    setStartPt(pt);
    setBox(null);
    setDrawing(true);
  };

  const onPointerMove = (e) => {
    // Handle corner-resize drag
    if (resizing) {
      e.preventDefault();
      const pt = getPoint(e);
      setBox(prev => {
        if (!prev) return prev;
        const b = { ...prev };
        if (resizing.includes("l")) b.x1 = Math.min(pt.x, prev.x2 - 20);
        if (resizing.includes("r")) b.x2 = Math.max(pt.x, prev.x1 + 20);
        if (resizing.includes("t")) b.y1 = Math.min(pt.y, prev.y2 - 20);
        if (resizing.includes("b")) b.y2 = Math.max(pt.y, prev.y1 + 20);
        return b;
      });
      return;
    }
    if (!drawing || !startPt) return;
    e.preventDefault();
    const pt = getPoint(e);
    setBox({
      x1: Math.min(startPt.x, pt.x), y1: Math.min(startPt.y, pt.y),
      x2: Math.max(startPt.x, pt.x), y2: Math.max(startPt.y, pt.y),
    });
  };

  const onPointerUp = (e) => {
    if (resizing) {
      setResizing(null);
      return;
    }
    if (!drawing) return;
    e.preventDefault();
    setDrawing(false);
    // Discard boxes smaller than 10×10 rendered px (accidental tap)
    setBox(prev => (prev && (prev.x2 - prev.x1) >= 10 && (prev.y2 - prev.y1) >= 10) ? prev : null);
  };

  // Convert rendered-space box → natural image pixel coords
  const toNatural = (b) => {
    const sx = imgNatural.w / imgRendered.w;
    const sy = imgNatural.h / imgRendered.h;
    return {
      x1: Math.round((b.x1 - imgRendered.x) * sx),
      y1: Math.round((b.y1 - imgRendered.y) * sy),
      x2: Math.round((b.x2 - imgRendered.x) * sx),
      y2: Math.round((b.y2 - imgRendered.y) * sy),
    };
  };

  const handleFindItem = async () => {
    if (!box || loading) return;
    setLoading(true);
    const natural = toNatural(box);

    // Crop client-side for the confirm preview
    let cropBlob = null;
    try {
      const bitmap = await createImageBitmap(imageFile);
      const cw = Math.max(1, natural.x2 - natural.x1);
      const ch = Math.max(1, natural.y2 - natural.y1);
      const canvas = document.createElement("canvas");
      canvas.width = cw; canvas.height = ch;
      canvas.getContext("2d").drawImage(bitmap, natural.x1, natural.y1, cw, ch, 0, 0, cw, ch);
      cropBlob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.9));
    } catch { /* non-critical — confirm view will show placeholder */ }

    // Ask CLIP for the category
    const form = new FormData();
    form.append("file", imageFile);
    form.append("x1", natural.x1); form.append("y1", natural.y1);
    form.append("x2", natural.x2); form.append("y2", natural.y2);
    try {
      const res  = await fetch(`${API}/classify-crop`, { method: "POST", body: form });
      const data = res.ok ? await res.json() : {};
      onConfirm({ bbox: natural, cropBlob, category: data.category || null, allScores: data.all_scores || {} });
    } catch {
      // Network error: still proceed, user can pick category manually
      onConfirm({ bbox: natural, cropBlob, category: null, allScores: {} });
    }
  };

  const hasBox = box && (box.x2 - box.x1) >= 10 && (box.y2 - box.y1) >= 10;

  return (
    <div className="fade-in" style={{ minHeight: "100svh", background: T.bg }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "16px 20px", borderBottom: `1px solid ${T.borderFaint}`, background: T.surface }}>
        <button className="btn-ghost" onClick={onBack} style={{ padding: "6px 10px" }}>← Back</button>
        <span style={{ fontSize: "0.85rem", fontWeight: 500, color: T.textMuted }}>
          {hasBox ? "Looks good — tap Find or redraw" : "Draw a tight box around the item"}
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "24px" }}>
        <div
          ref={containerRef}
          style={{ position: "relative", width: "100%", maxWidth: 640, cursor: loading ? "wait" : (hasBox ? "default" : "crosshair"), touchAction: (hasBox && !resizing) ? "pan-y" : "none", userSelect: "none" }}
          onMouseDown={onPointerDown}
          onMouseMove={onPointerMove}
          onMouseUp={onPointerUp}
          onMouseLeave={onPointerUp}
          onTouchStart={onPointerDown}
          onTouchMove={onPointerMove}
          onTouchEnd={onPointerUp}
        >
          <img
            ref={imgRef}
            src={imageURL}
            onLoad={onImgLoad}
            alt="uploaded"
            draggable={false}
            onContextMenu={(e) => e.preventDefault()}
            style={{ width: "100%", display: "block", borderRadius: 12, pointerEvents: "none", userSelect: "none" }}
          />
          {imgRendered.w > 0 && (
            <svg style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}>
              {/* Dim everything outside the drawn box */}
              {hasBox && (
                <>
                  <defs>
                    <mask id="crop-mask">
                      <rect x="0" y="0" width="100%" height="100%" fill="white" />
                      <rect x={box.x1} y={box.y1} width={box.x2 - box.x1} height={box.y2 - box.y1} fill="black" />
                    </mask>
                  </defs>
                  <rect x="0" y="0" width="100%" height="100%" fill="rgba(0,0,0,0.38)" mask="url(#crop-mask)" rx="12" />
                </>
              )}
              {/* The box itself */}
              {box && (
                <rect
                  x={box.x1} y={box.y1} width={box.x2 - box.x1} height={box.y2 - box.y1}
                  fill="rgba(0,122,255,0.08)"
                  stroke={T.accent}
                  strokeWidth={2}
                  strokeDasharray={drawing ? "6 3" : "none"}
                  rx={4}
                />
              )}
            </svg>
          )}
          {/* Tight-box hint overlay — shown until user starts drawing */}
          {!drawing && !box && imgRendered.w > 0 && (
            <div style={{
              position: "absolute", inset: 0,
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
              background: "rgba(0,0,0,0.45)", backdropFilter: "blur(2px)",
              borderRadius: 12,
              pointerEvents: "none",
              gap: 10,
            }}>
              <div style={{
                background: "rgba(0,0,0,0.7)", borderRadius: 16, padding: "18px 24px",
                display: "flex", flexDirection: "column", alignItems: "center", gap: 8,
                maxWidth: 280, textAlign: "center",
              }}>
                <span style={{ fontSize: "2rem", lineHeight: 1 }}>✂️</span>
                <span style={{ fontSize: "1.05rem", color: "#fff", fontWeight: 700, lineHeight: 1.3 }}>
                  Draw a tight box around the item
                </span>
                <span style={{ fontSize: "0.78rem", color: "rgba(255,255,255,0.75)", lineHeight: 1.5 }}>
                  Hug the edges as closely as possible — don't leave extra background inside the box.
                </span>
              </div>
            </div>
          )}

          {/* Draggable corner handles — HTML divs so pointer events work on mobile */}
          {hasBox && !drawing && (
            [
              { id: "tl", cx: box.x1, cy: box.y1, cursor: "nwse-resize" },
              { id: "tr", cx: box.x2, cy: box.y1, cursor: "nesw-resize" },
              { id: "bl", cx: box.x1, cy: box.y2, cursor: "nesw-resize" },
              { id: "br", cx: box.x2, cy: box.y2, cursor: "nwse-resize" },
            ].map(({ id, cx, cy, cursor }) => (
              <div
                key={id}
                onMouseDown={(e) => { e.stopPropagation(); e.preventDefault(); setResizing(id); }}
                onTouchStart={(e) => { e.stopPropagation(); e.preventDefault(); setResizing(id); }}
                style={{
                  position: "absolute",
                  left: cx, top: cy,
                  width: 28, height: 28,
                  transform: "translate(-50%, -50%)",
                  borderRadius: "50%",
                  background: T.accent,
                  cursor,
                  touchAction: "none",
                  zIndex: 10,
                }}
              />
            ))
          )}
        </div>


        <div style={{ width: "100%", maxWidth: 640, marginTop: 16, display: "flex", gap: 10 }}>
          {hasBox && (
            <button className="btn-ghost" style={{ flexShrink: 0 }} onClick={() => { setBox(null); setStartPt(null); setResizing(null); }}>
              Redraw
            </button>
          )}
          <button
            className="btn-primary"
            style={{ flex: 1, opacity: hasBox && !loading ? 1 : 0.4, pointerEvents: hasBox && !loading ? "auto" : "none" }}
            onClick={handleFindItem}
          >
            {loading
              ? <><div style={{ width: 16, height: 16, border: "2px solid rgba(255,255,255,0.4)", borderTop: "2px solid #fff", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} /> Identifying…</>
              : <>Find this item</>}
          </button>
        </div>

        <p style={{ marginTop: 12, fontSize: "0.7rem", color: T.textFaint, textAlign: "center" }}>
          Drag to draw · adjust corners · works on mobile
        </p>
      </div>

      {/* Tutorial modal */}
      {showTutorial && (
        <div
          style={{
            position: "fixed", inset: 0, zIndex: 9999,
            background: "rgba(0,0,0,0.72)", backdropFilter: "blur(8px)",
            display: "flex", alignItems: "center", justifyContent: "center",
            padding: 24,
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div style={{
            background: T.surface, borderRadius: 20, padding: "24px 20px",
            width: "100%", maxWidth: 360,
            maxHeight: "88svh", overflowY: "auto",
            display: "flex", flexDirection: "column", alignItems: "center", gap: 16,
            boxShadow: "0 24px 60px rgba(0,0,0,0.5)",
          }}>
            <div style={{ textAlign: "center", flexShrink: 0 }}>
              <p style={{ margin: 0, fontSize: "1.1rem", fontWeight: 700, color: T.text }}>How to select an item</p>
              <p style={{ margin: "6px 0 0", fontSize: "0.78rem", color: T.textMuted }}>
                Draw a box that hugs the item tightly — no extra background.
              </p>
            </div>

            <video
              src={TUTORIAL_VIDEO}
              autoPlay loop muted playsInline
              onError={() => {}}
              style={{ width: "100%", borderRadius: 12, background: "#000", maxHeight: "min(50svh, 50vh)", objectFit: "cover" }}
            />

            <button
              className="btn-primary"
              style={{ width: "100%", padding: "12px 0", fontSize: "0.9rem", flexShrink: 0 }}
              onClick={() => setShowTutorial(false)}
            >
              Got it
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// LOCATION PICKER MODAL
// ══════════════════════════════════════════════════════════════════
function MapClickHandler({ onMapClick }) {
  useMapEvents({ click(e) { onMapClick([e.latlng.lat, e.latlng.lng]); } });
  return null;
}

const searchPinIcon = (active) => L.divIcon({
  html: `<div style="
    width:24px; height:24px;
    background:${active ? T.accent : "#999"};
    border-radius:50% 50% 50% 0;
    transform:rotate(-45deg);
    border:2.5px solid #fff;
    box-shadow:0 2px 12px rgba(0,0,0,0.45);
  "></div>`,
  className: "",
  iconSize: [24, 24],
  iconAnchor: [12, 24],
  popupAnchor: [0, -26],
});

const gpsUserIcon = L.divIcon({
  html: `<div style="width:16px;height:16px;background:#4A90E2;border-radius:50%;border:2.5px solid #fff;box-shadow:0 0 0 4px rgba(74,144,226,0.25);"></div>`,
  className: "",
  iconSize: [16, 16],
  iconAnchor: [8, 8],
});

function LocationPickerModal({ userLocation, storeLocations, currentSearchLocation, onConfirm, onClose }) {
  const fallback = [33.8869, 35.5131];
  const gpsLL    = userLocation || fallback;
  const [pin, setPin] = useState(currentSearchLocation || gpsLL);
  const isGPS = userLocation && pin[0] === gpsLL[0] && pin[1] === gpsLL[1];

  const handleUseGPS = () => {
    if (!userLocation) return;
    setPin([...gpsLL]);
  };

  return (
    <div style={{
      position: "fixed", inset: 0, zIndex: 2000,
      background: T.bg, display: "flex", flexDirection: "column",
      animation: "slideUp 0.22s cubic-bezier(0.16,1,0.3,1) forwards",
    }}>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 20px",
        borderBottom: `1px solid ${T.borderFaint}`,
        background: T.surface,
        flexShrink: 0,
      }}>
        <div>
          <div style={{ fontSize: "0.95rem", fontWeight: 600, color: T.text }}>Search location</div>
          <div style={{ fontSize: "0.68rem", color: T.textMuted, marginTop: 2 }}>
            Tap anywhere on the map to place your search pin
          </div>
        </div>
        <button onClick={onClose} style={{ background: "none", border: "none", fontSize: "1.2rem", color: T.textMuted, cursor: "pointer", padding: "4px 8px" }}>✕</button>
      </div>

      {/* Map */}
      <div style={{ flex: 1, position: "relative", minHeight: 0 }}>
        <MapContainer
          center={pin}
          zoom={13}
          style={{ height: "100%", width: "100%" }}
          zoomControl={true}
        >
          <TileLayer url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" attribution="&copy; CartoDB" />
          <MapClickHandler onMapClick={setPin} />
          <MapRecenter center={pin} />

          {/* Store pins (faded reference) */}
          {Object.entries(storeLocations).map(([store, coords]) => (
            <Marker key={store} position={coords} icon={L.divIcon({
              html: `<div style="width:10px;height:10px;background:${T.textFaint};border-radius:50%;border:1.5px solid #fff;"></div>`,
              className: "",
              iconSize: [10, 10],
              iconAnchor: [5, 5],
            })}>
              <Popup><span style={{ fontSize: "0.8rem", color: T.textMuted }}>{store}</span></Popup>
            </Marker>
          ))}

          {/* GPS dot */}
          {userLocation && (
            <Marker position={gpsLL} icon={gpsUserIcon}>
              <Popup><span style={{ fontSize: "0.8rem" }}>Your GPS location</span></Popup>
            </Marker>
          )}

          {/* Search pin */}
          <Marker position={pin} icon={searchPinIcon(true)}>
            <Popup>
              <span style={{ fontSize: "0.8rem", color: T.accent, fontWeight: 600 }}>Search here</span><br />
              <span style={{ fontSize: "0.72rem", color: T.textMuted }}>{pin[0].toFixed(4)}, {pin[1].toFixed(4)}</span>
            </Popup>
          </Marker>
        </MapContainer>

        {/* Use GPS button floating on map */}
        {userLocation && (
          <button
            onClick={handleUseGPS}
            style={{
              position: "absolute", bottom: 20, left: "50%", transform: "translateX(-50%)",
              zIndex: 1000,
              background: isGPS ? T.accent : T.surface,
              color: isGPS ? "#fff" : T.text,
              border: `1px solid ${isGPS ? T.accent : T.border}`,
              borderRadius: 24, padding: "10px 20px",
              fontSize: "0.82rem", fontFamily: "inherit", fontWeight: 500,
              cursor: "pointer",
              boxShadow: "0 2px 12px rgba(0,0,0,0.18)",
              display: "flex", alignItems: "center", gap: 8,
              transition: "all 0.15s",
            }}
          >
            <span style={{ fontSize: "0.7rem", lineHeight: 1 }}>◉</span>
            {isGPS ? "Using GPS location" : "Use GPS location"}
          </button>
        )}
      </div>

      {/* Bottom action bar */}
      <div style={{
        padding: "16px 20px", paddingBottom: "calc(16px + env(safe-area-inset-bottom, 0px))",
        borderTop: `1px solid ${T.borderFaint}`,
        background: T.surface,
        display: "flex", alignItems: "center", gap: 14,
        flexShrink: 0,
      }}>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: "0.62rem", color: T.textMuted, letterSpacing: "0.1em", textTransform: "uppercase" }}>Pinned at</div>
          <div style={{ fontSize: "0.82rem", color: T.text, marginTop: 2, fontVariantNumeric: "tabular-nums" }}>
            {pin[0].toFixed(5)}, {pin[1].toFixed(5)}
          </div>
        </div>
        <button
          onClick={() => onConfirm(pin)}
          style={{
            padding: "12px 28px", background: T.text, color: T.bg,
            border: "none", borderRadius: 12,
            fontSize: "0.88rem", fontFamily: "inherit", fontWeight: 500,
            cursor: "pointer", letterSpacing: "0.02em",
          }}
        >
          Search here →
        </button>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// CONFIRM VIEW  — shows crop + CLIP category prediction for approval
// ══════════════════════════════════════════════════════════════════
function ConfirmView({ cropBlob, predictedCategory, allScores, onSearch, onBack, searchLocation, userLocation, onOpenLocationPicker }) {
  const cropURL = useMemo(() => {
    if (!cropBlob) return null;
    const url = URL.createObjectURL(cropBlob);
    return url;
  }, [cropBlob]);

  // Revoke object URL when cropBlob changes or component unmounts
  useEffect(() => {
    return () => { if (cropURL) URL.revokeObjectURL(cropURL); };
  }, [cropURL]);

  const isNotFashion = predictedCategory === "not_fashion";
  const [category, setCategory] = useState(
    isNotFashion ? Object.keys(CATEGORY_LABELS)[0] : (predictedCategory || Object.keys(CATEGORY_LABELS)[0])
  );

  const conf      = (allScores && category) ? (allScores[category] ?? 0) : 0;
  const confLabel = conf > 0.6 ? "high confidence" : conf > 0.35 ? "medium confidence" : "low confidence";
  const confColor = conf > 0.6 ? T.green : conf > 0.35 ? T.yellow : T.textMuted;

  return (
    <div className="fade-in" style={{ minHeight: "100svh", background: T.bg }}>
      <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "16px 20px", borderBottom: `1px solid ${T.borderFaint}`, background: T.surface }}>
        <button className="btn-ghost" onClick={onBack} style={{ padding: "6px 10px" }}>← Back</button>
        <span style={{ fontSize: "0.9rem", fontWeight: 600, color: T.text }}>Confirm item</span>
      </div>

      {isNotFashion && (
        <div style={{ margin: "24px 24px 0", padding: "18px 20px", background: "rgba(255,59,48,0.07)", border: "1px solid rgba(255,59,48,0.25)", borderRadius: 14, display: "flex", flexDirection: "column", gap: 10 }}>
          <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "#e05a5a" }}>Not a fashion item</div>
          <div style={{ fontSize: "0.75rem", color: T.textMuted, lineHeight: 1.6 }}>
            The selected area doesn't look like a clothing item or accessory (socks, underwear, belts, etc. aren't searchable).<br />
            Go back and draw a box around a clothing item instead.
          </div>
          <button className="btn-ghost" style={{ alignSelf: "flex-start", marginTop: 4 }} onClick={onBack}>← Redraw</button>
        </div>
      )}

      <div style={{ flexDirection: "column", alignItems: "center", padding: "28px 20px", paddingBottom: "calc(83px + env(safe-area-inset-bottom, 0px) + 12px)", gap: 24, display: isNotFashion ? "none" : "flex" }}>

        <div style={{ display: "flex", gap: 20, alignItems: "flex-start", width: "100%", maxWidth: 480 }}>
          {/* Crop thumbnail */}
          <div style={{ width: 100, height: 130, flexShrink: 0, background: T.surface, border: `1.5px solid ${T.accent}`, borderRadius: 10, overflow: "hidden" }}>
            {cropURL
              ? <img src={cropURL} alt="crop" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
              : <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: T.textFaint, fontSize: "1.4rem" }}>✦</div>
            }
          </div>

          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Predicted category */}
            <div>
              <div style={{ fontSize: "0.6rem", color: T.textMuted, letterSpacing: "0.1em", marginBottom: 6 }}>DETECTED AS</div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                <span style={{ fontSize: "1.4rem", fontWeight: 700, color: T.text, letterSpacing: "-0.02em" }}>
                  {CATEGORY_LABELS[category] ?? "Unknown"}
                </span>
                {conf > 0 && (
                  <span style={{ fontSize: "0.62rem", color: confColor, border: `1px solid ${confColor}`, borderRadius: 20, padding: "2px 8px" }}>
                    {confLabel}
                  </span>
                )}
              </div>
            </div>

            {/* Correction dropdown */}
            <div>
              <div style={{ fontSize: "0.6rem", color: T.textMuted, letterSpacing: "0.1em", marginBottom: 6 }}>NOT RIGHT? CHANGE IT</div>
              <div style={{ position: "relative" }}>
                <select
                  value={category}
                  onChange={e => setCategory(e.target.value)}
                  style={{
                    width: "100%", background: T.surface, color: T.text,
                    border: `1px solid ${T.border}`, borderRadius: 8,
                    padding: "7px 32px 7px 10px", fontSize: "0.82rem",
                     cursor: "pointer",
                    appearance: "none", outline: "none",
                  }}
                >
                  {Object.entries(CATEGORY_LABELS).map(([val, lbl]) => (
                    <option key={val} value={val}>{lbl}</option>
                  ))}
                </select>
                <svg style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }} width="10" height="6" viewBox="0 0 10 6">
                  <path d="M0 0l5 6 5-6z" fill={T.textMuted} />
                </svg>
              </div>
            </div>
          </div>
        </div>

        {/* Location row */}
        <div style={{
          width: "100%", maxWidth: 480,
          display: "flex", alignItems: "center", justifyContent: "space-between",
          background: T.surface, border: `1px solid ${T.border}`,
          borderRadius: 12, padding: "12px 16px",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: "1rem", lineHeight: 1 }}>
              {searchLocation ? "📍" : "◉"}
            </span>
            <div>
              <div style={{ fontSize: "0.62rem", color: T.textMuted, letterSpacing: "0.1em", textTransform: "uppercase" }}>Searching near</div>
              <div style={{ fontSize: "0.82rem", color: T.text, marginTop: 2, fontWeight: 500 }}>
                {searchLocation
                  ? `${searchLocation[0].toFixed(4)}, ${searchLocation[1].toFixed(4)}`
                  : (userLocation ? "Your current location" : "Detecting location…")}
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={onOpenLocationPicker}
            style={{
              background: T.accentBg, border: `1px solid ${T.accentRing}`,
              borderRadius: 8, padding: "7px 14px",
              fontSize: "0.75rem", color: T.accent, cursor: "pointer",
              fontFamily: "inherit", fontWeight: 500, whiteSpace: "nowrap",
            }}
          >
            Change
          </button>
        </div>

        <button
          className="btn-primary"
          style={{ width: "100%", maxWidth: 480 }}
          onClick={() => onSearch(category)}
        >
          <span style={{ color: T.accent }}>✦</span> Search
        </button>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// PRODUCT DETAIL SHEET
// ══════════════════════════════════════════════════════════════════
function ProductDetailSheet({ result, category, judgeScore, onFeedback, onClose, saved, onToggleSave, storeLocations = {} }) {
  const [vote, setVote]         = useState(null);
  const [hoverStar, setHover]   = useState(null);
  const [imgLoaded, setImgLoaded] = useState(false);
  const [imgError, setImgError]   = useState(false);
  const [copied, setCopied]       = useState(false);

  const payload      = result.payload ?? {};
  const raw          = payload.image_url ?? "";
  const imageURL     = raw ? (raw.startsWith("http") ? raw : `${API}${raw}`) : null;
  const score        = result.score ?? 0;
  const displayScore = judgeScore ?? score;
  const isJudged     = judgeScore !== null && judgeScore !== undefined;
  const scoreColor   = displayScore >= 0.80 ? T.green : displayScore >= 0.60 ? T.yellow : T.red;
  const storeCoords  = storeLocations[payload.store] ?? null;
  const catColor     = CATEGORY_COLORS[payload.category_tag || category] || T.accent;
  const productUrl   = payload.product_url || payload.source_url || payload.url || null;

  const mapsHref = payload.maps_url ||
    (payload.store
      ? `https://maps.google.com/?q=${encodeURIComponent([payload.store, payload.mall_name].filter(Boolean).join(" "))}`
      : null);
  const hasLocation = !!(storeCoords || mapsHref);

  // Close on Escape for keyboard users
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose?.(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const handleVote = async (stars) => {
    if (stars === vote) return;
    setVote(stars);
    setHover(null);
    await onFeedback(
      payload.product_id || payload.item_id || payload.image_url,
      stars,
      payload.item_name || "",
      payload.store     || "",
      category          || "",
      payload.image_url || "",
    );
  };

  const handleShare = async () => {
    const shareData = {
      title: payload.item_name || "Locus find",
      text : [payload.item_name, payload.store && `at ${payload.store}`, payload.price].filter(Boolean).join(" • "),
      url  : productUrl || (typeof window !== "undefined" ? window.location.href : ""),
    };
    try {
      if (navigator.share && navigator.canShare?.(shareData) !== false) {
        await navigator.share(shareData);
        return;
      }
    } catch { /* user cancelled — fall through to clipboard */ }
    try {
      await navigator.clipboard.writeText(`${shareData.text} ${shareData.url}`.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch { /* clipboard unavailable — silently ignore */ }
  };

  const fillUpTo = vote !== null ? vote : (hoverStar ?? 0);

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: "fixed", inset: 0,
          background: "rgba(0,0,0,0.5)",
          backdropFilter: "blur(3px)",
          WebkitBackdropFilter: "blur(3px)",
          zIndex: 500,
          animation: "fadeIn 0.2s ease forwards",
        }}
      />

      {/* Sheet */}
      <div
        role="dialog"
        aria-modal="true"
        aria-label={payload.item_name || "Product details"}
        style={{
          position: "fixed", bottom: 0, left: 0, right: 0,
          maxHeight: "92dvh",
          background: T.surface,
          borderRadius: "20px 20px 0 0",
          zIndex: 501,
          display: "flex",
          flexDirection: "column",
          overflowY: "auto",
          animation: "slideUp 0.32s cubic-bezier(0.16,1,0.3,1) forwards",
          boxShadow: "0 -8px 40px rgba(0,0,0,0.6)",
        }}
      >
        {/* Drag handle */}
        <div style={{ display: "flex", justifyContent: "center", padding: "10px 0 4px", flexShrink: 0 }}>
          <div style={{ width: 40, height: 4, borderRadius: 2, background: T.border }} />
        </div>

        {/* Product image — full item visible on portrait canvas */}
        <div style={{
          position: "relative",
          width: "100%",
          aspectRatio: "3 / 4",
          maxHeight: "58dvh",
          background: `radial-gradient(ellipse at center, ${T.bgDeep} 0%, #07060500 100%), ${T.bgDeep}`,
          overflow: "hidden",
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}>
          {imageURL && !imgError ? (
            <>
              <img
                src={imageURL}
                alt={payload.item_name || "product"}
                onLoad={() => setImgLoaded(true)}
                onError={() => setImgError(true)}
                style={{
                  maxWidth: "100%",
                  maxHeight: "100%",
                  width: "auto",
                  height: "auto",
                  objectFit: "contain",
                  opacity: imgLoaded ? 1 : 0,
                  transition: "opacity 0.3s ease",
                  filter: "drop-shadow(0 12px 24px rgba(0,0,0,0.45))",
                }}
                draggable={false}
              />
              {!imgLoaded && (
                <div style={{
                  position: "absolute",
                  color: T.textFaint,
                  fontSize: "0.7rem",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                }}>
                  loading…
                </div>
              )}
            </>
          ) : (
            <div style={{ color: T.textFaint, fontSize: "2.5rem" }}>✦</div>
          )}

          {/* Close button — floats over image, high contrast */}
          <button
            onClick={onClose}
            aria-label="Close"
            style={{
              position: "absolute", top: 12, right: 12,
              background: "rgba(0,0,0,0.55)",
              backdropFilter: "blur(6px)",
              WebkitBackdropFilter: "blur(6px)",
              border: `1px solid ${T.border}`,
              borderRadius: "50%",
              width: 34, height: 34, cursor: "pointer",
              color: T.text, fontSize: "1.1rem", lineHeight: 1,
              display: "flex", alignItems: "center", justifyContent: "center",
              zIndex: 2,
              transition: "transform 0.15s, background 0.15s",
            }}
            onMouseEnter={e => { e.currentTarget.style.background = "rgba(0,0,0,0.9)"; e.currentTarget.style.transform = "scale(1.06)"; }}
            onMouseLeave={e => { e.currentTarget.style.background = "rgba(0,0,0,0.55)"; e.currentTarget.style.transform = "scale(1)"; }}
          >×</button>

          {/* Match badge — floats top-left on image */}
          <div style={{
            position: "absolute", top: 12, left: 12,
            display: "flex", alignItems: "center", gap: 5,
            background: "rgba(0,0,0,0.55)",
            backdropFilter: "blur(6px)",
            WebkitBackdropFilter: "blur(6px)",
            border: `1px solid ${scoreColor}55`,
            borderRadius: 20, padding: "5px 10px",
            color: scoreColor, fontSize: "0.68rem", fontWeight: 600,
          }}>
            {Math.round(displayScore * 100)}% match
            {isJudged && (
              <span style={{ fontSize: "0.55rem", color: T.accent, border: `1px solid ${T.accent}`, borderRadius: 3, padding: "0 3px", lineHeight: 1.6 }}>AI</span>
            )}
          </div>
        </div>

        {/* Content */}
        <div style={{ padding: "22px 22px 32px", display: "flex", flexDirection: "column", gap: 18 }}>

          {/* Header: name, price, category pill */}
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {(payload.category_tag || category) && (
              <span style={{
                fontSize: "0.62rem", fontWeight: 500,
                color: catColor,
                letterSpacing: "0.15em",
                textTransform: "uppercase",
              }}>
                {(payload.category_tag || category).replace(/_/g, " ")}
              </span>
            )}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 14 }}>
              <h2 style={{
                fontSize: "1.35rem",
                fontWeight: 700,
                color: T.text,
                lineHeight: 1.25,
                letterSpacing: "-0.02em",
                flex: 1,
                margin: 0,
              }}>
                {payload.item_name || "Product"}
              </h2>
              {payload.price && (
                <span style={{
                  fontSize: "1.1rem",
                  fontWeight: 600,
                  color: T.accent,
                  whiteSpace: "nowrap",
                  paddingTop: 3,
                }}>
                  {payload.price}
                </span>
              )}
            </div>
          </div>

          {/* Store card (only if store is known) */}
          {payload.store && (
            <div style={{
              display: "flex", alignItems: "center", gap: 12,
              padding: "12px 14px",
              background: T.bgDeep,
              borderRadius: 12,
              border: `1px solid ${T.borderFaint}`,
            }}>
              <div style={{
                width: 36, height: 36, borderRadius: 10,
                background: `${T.accent}1a`,
                border: `1px solid ${T.accent}40`,
                display: "flex", alignItems: "center", justifyContent: "center",
                color: T.accent, fontSize: "1rem",
                flexShrink: 0,
              }}>
                {/* location pin glyph */}
                ◎
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
                <span style={{ fontSize: "0.88rem", color: T.text, fontWeight: 500 }}>{payload.store}</span>
                {payload.mall_name && (
                  <span style={{ fontSize: "0.72rem", color: T.textMuted }}>inside {payload.mall_name}</span>
                )}
              </div>
            </div>
          )}

          {/* Action row: Save • Share • View in store (conditional) */}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button
              onClick={() => onToggleSave?.(result)}
              style={{
                flex: "1 1 90px",
                background: saved ? `${T.accent}22` : T.bgDeep,
                border: `1px solid ${saved ? T.accent : T.border}`,
                borderRadius: 10,
                padding: "10px 12px",
                cursor: "pointer",
                color: saved ? T.accent : T.text,
                fontSize: "0.82rem", fontWeight: 500,
                
                display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                transition: "all 0.2s",
              }}
              title={saved ? "Remove from saved" : "Save item"}
            >
              <span style={{ fontSize: "0.95rem" }}>{saved ? "♥" : "♡"}</span>
              {saved ? "Saved" : "Save"}
            </button>

            <button
              onClick={handleShare}
              style={{
                flex: "1 1 90px",
                background: T.bgDeep,
                border: `1px solid ${T.border}`,
                borderRadius: 10,
                padding: "10px 12px",
                cursor: "pointer",
                color: copied ? T.green : T.text,
                fontSize: "0.82rem", fontWeight: 500,
                
                display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                transition: "all 0.2s",
              }}
              title="Share"
            >
              <span style={{ fontSize: "0.9rem" }}>{copied ? "✓" : "↗"}</span>
              {copied ? "Copied" : "Share"}
            </button>

            {productUrl && (
              <a
                href={productUrl}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => {
                  fetch(`${API}/track-event`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ event_type: "website_click", store: payload.store || "" }),
                  }).catch(() => {});
                }}
                style={{
                  flex: "1 1 100%",
                  background: T.accent,
                  border: `1px solid ${T.accent}`,
                  borderRadius: 10,
                  padding: "11px 14px",
                  textDecoration: "none",
                  color: T.bgDeep,
                  fontSize: "0.85rem", fontWeight: 600,
                  
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 6,
                  transition: "filter 0.2s",
                }}
                onMouseEnter={e => e.currentTarget.style.filter = "brightness(1.08)"}
                onMouseLeave={e => e.currentTarget.style.filter = "brightness(1)"}
              >
                View on {payload.store || "store"} →
              </a>
            )}
          </div>

          {/* Location section */}
          {hasLocation && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <span style={{ fontSize: "0.65rem", color: T.textMuted, letterSpacing: "0.12em", textTransform: "uppercase" }}>Find in store</span>
                {storeCoords && <span style={{ fontSize: "0.62rem", color: T.textFaint }}>drag to explore</span>}
              </div>
              {storeCoords && (
                <div style={{ borderRadius: 12, overflow: "hidden", border: `1px solid ${T.border}`, height: 220 }}>
                  <MapContainer
                    center={storeCoords}
                    zoom={15}
                    style={{ height: "100%", width: "100%" }}
                    zoomControl={true}
                    scrollWheelZoom={true}
                    dragging={true}
                    doubleClickZoom={true}
                    touchZoom={true}
                  >
                    <TileLayer url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" attribution="&copy; CartoDB" />
                    <Marker position={storeCoords}>
                      <Popup>
                        <strong style={{ color: T.accent }}>{payload.store}</strong>
                        {payload.mall_name && <><br />{payload.mall_name}</>}
                      </Popup>
                    </Marker>
                  </MapContainer>
                </div>
              )}
              <a
                href={mapsHref}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => {
                  fetch(`${API}/track-event`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ event_type: "directions_click", store: payload.store || "" }),
                  }).catch(() => {});
                }}
                style={{
                  display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                  padding: "12px", borderRadius: 10,
                  background: T.accentBg, border: `1px solid ${T.accentRing}`,
                  color: T.accent, fontSize: "0.82rem", fontWeight: 500,
                  textDecoration: "none",
                  transition: "background 0.2s",
                }}
                onMouseEnter={e => e.currentTarget.style.background = T.accentRing}
                onMouseLeave={e => e.currentTarget.style.background = T.accentBg}
              >
                <span style={{ fontSize: "1rem" }}>◎</span> Get directions →
              </a>
            </div>
          )}

          {/* Star rating */}
          <div style={{
            display: "flex", flexDirection: "column", gap: 10,
            padding: "14px 16px",
            background: T.bgDeep,
            borderRadius: 12,
            border: `1px solid ${T.borderFaint}`,
          }}>
            <span style={{ fontSize: "0.65rem", color: T.textMuted, letterSpacing: "0.12em", textTransform: "uppercase" }}>
              Rate this match
            </span>
            <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
              {[1, 2, 3, 4, 5].map(star => (
                <button
                  key={star}
                  className={`star-btn ${star <= fillUpTo ? "filled" : ""}`}
                  onMouseEnter={() => setHover(star)}
                  onMouseLeave={() => setHover(null)}
                  onClick={() => handleVote(star)}
                  style={{ fontSize: "1.5rem" }}
                  title={`Rate ${star} star${star > 1 ? "s" : ""}`}
                >
                  ★
                </button>
              ))}
              {vote !== null && (
                <span style={{ fontSize: "0.72rem", color: T.textMuted, marginLeft: 10 }}>
                  {vote >= 4 ? "great find!" : vote === 3 ? "okay match" : "poor match"}
                </span>
              )}
            </div>
          </div>

        </div>
      </div>
    </>
  );
}

// ══════════════════════════════════════════════════════════════════
// RESULT CARD
// ══════════════════════════════════════════════════════════════════
function ResultCard({ result, category, onFeedback, onCardClick, highlighted, judgeScore, saved, onToggleSave, onFlag }) {
  const [flagged, setFlagged] = useState(false);
  const [cardRating, setCardRating] = useState(null);

  const score    = result.score ?? 0;
  const payload  = result.payload ?? {};
  const raw      = payload.image_url ?? "";
  const imageURL = raw ? (raw.startsWith("http") ? raw : `${API}${raw}`) : null;

  const displayScore = judgeScore ?? score;
  const scoreColor   = displayScore >= 0.80 ? T.green : displayScore >= 0.60 ? T.yellow : T.red;
  const catColor     = CATEGORY_COLORS[payload.category_tag || category] || T.accent;

  return (
    <div
      style={{
        background: T.surface,
        border: `1px solid ${highlighted ? T.accent : T.border}`,
        borderRadius: "14px",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        transition: "border-color 0.2s, transform 0.22s cubic-bezier(0.16,1,0.3,1), box-shadow 0.22s cubic-bezier(0.16,1,0.3,1)",
        cursor: "pointer",
      }}
      onMouseEnter={e => {
        e.currentTarget.style.transform = "translateY(-3px)";
        e.currentTarget.style.boxShadow = "0 10px 32px rgba(0,0,0,0.08)";
      }}
      onMouseLeave={e => {
        e.currentTarget.style.transform = "";
        e.currentTarget.style.boxShadow = "";
      }}
      onClick={() => onCardClick?.(result)}
    >
      {/* Image */}
      <div style={{ aspectRatio: "3/4", background: T.bgDeep, overflow: "hidden", position: "relative" }}>
        {imageURL
          ? <img
              src={imageURL}
              alt={payload.item_name || "product"}
              style={{ width: "100%", height: "100%", objectFit: "cover", transition: "transform 0.5s ease", display: "block" }}
              onMouseEnter={e => { e.target.style.transform = "scale(1.05)"; }}
              onMouseLeave={e => { e.target.style.transform = "scale(1)"; }}
            />
          : <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: T.textFaint, fontSize: "1.5rem" }}>✦</div>
        }

        {/* Match score — light frosted badge top-right */}
        <div style={{
          position: "absolute", top: 8, right: 8,
          background: "rgba(250,249,247,0.88)", backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
          borderRadius: 20, padding: "3px 9px",
        }}>
          <span style={{ fontSize: "0.62rem", fontWeight: 600, color: scoreColor }}>{Math.round(displayScore * 100)}%</span>
        </div>

        {/* Save button — bottom right, frosted */}
        <button
          onClick={(e) => { e.stopPropagation(); onToggleSave?.(result); }}
          style={{
            position: "absolute", bottom: 8, right: 8,
            background: saved ? T.accent : "rgba(250,249,247,0.88)",
            backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
            border: "none", borderRadius: "50%",
            width: 30, height: 30, cursor: "pointer",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "0.82rem",
            color: saved ? "#fff" : T.textMuted,
            transition: "background 0.2s, color 0.2s",
          }}
          title={saved ? "Unsave" : "Save"}
        >
          {saved ? "♥" : "♡"}
        </button>

        {/* Flag button — bottom left */}
        {onFlag && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              if (flagged) return;
              setFlagged(true);
              onFlag(
                payload.product_id || payload.item_id || payload.image_url,
                payload.item_name || "",
                payload.store     || "",
                payload.image_url || "",
              );
            }}
            style={{
              position: "absolute", bottom: 8, left: 8,
              background: flagged ? "rgba(180,50,50,0.85)" : "rgba(250,249,247,0.85)",
              backdropFilter: "blur(8px)", WebkitBackdropFilter: "blur(8px)",
              border: "none", borderRadius: "50%",
              width: 28, height: 28, cursor: flagged ? "default" : "pointer",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: "0.72rem",
              color: flagged ? "#fff" : T.textMuted,
              transition: "background 0.2s",
            }}
            title={flagged ? "Flagged — hidden from search" : "Flag: hide this result"}
          >
            🚩
          </button>
        )}
      </div>

      {/* Info */}
      <div style={{ padding: "11px 13px 13px", display: "flex", flexDirection: "column", gap: 5 }}>
        {/* Category accent bar */}
        <div style={{ height: 2, borderRadius: 1, background: `${catColor}38`, marginBottom: 1 }} />

        <div style={{ fontSize: "0.77rem", fontWeight: 500, color: T.text, lineHeight: 1.35, letterSpacing: "-0.01em" }}>
          {payload.item_name || "Product"}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 1 }}>
          <span style={{ fontSize: "0.63rem", color: T.textMuted, letterSpacing: "0.02em" }}>
            {payload.store || ""}
          </span>
          {payload.price && (
            <span style={{ fontSize: "0.75rem", color: T.accent, fontWeight: 600 }}>
              {payload.price}
            </span>
          )}
        </div>
        {onFeedback && (
          <div style={{ display: "flex", gap: 2, marginTop: 5 }} onClick={(e) => e.stopPropagation()}>
            {[1, 2, 3, 4, 5].map(star => (
              <button
                key={star}
                onClick={(e) => {
                  e.stopPropagation();
                  setCardRating(star);
                  onFeedback(
                    payload.product_id || payload.item_id || payload.image_url,
                    star,
                    payload.item_name || "",
                    payload.store || "",
                    payload.category_tag || category || "",
                    payload.image_url || "",
                  );
                }}
                style={{
                  background: "none", border: "none", cursor: "pointer", padding: 0,
                  fontSize: "0.8rem", lineHeight: 1,
                  color: star <= (cardRating ?? 0) ? T.yellow : T.borderFaint,
                  transition: "color 0.15s",
                }}
                title={`Rate ${star} star${star > 1 ? "s" : ""}`}
              >★</button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// ATTRIBUTE PANEL
// ══════════════════════════════════════════════════════════════════
function AttributePanel({ attributes, refineMode, onRefine }) {
  const isLoading = attributes === null;
  const hasAttrs  = attributes && Object.keys(attributes).length > 0;

  const chips = [];
  if (hasAttrs) {
    const a = attributes;
    if (a.colors?.length)       chips.push(...a.colors.slice(0, 2).map(c => ({ label: c, key: "color" })));
    if (a.style)                chips.push({ label: a.style, key: "style" });
    if (a.silhouette)           chips.push({ label: a.silhouette, key: "silhouette" });
    if (a.pattern && a.pattern !== "solid") chips.push({ label: a.pattern, key: "pattern" });
    if (a.trend_tags?.length)   chips.push(...a.trend_tags.slice(0, 1).map(t => ({ label: t, key: "trend" })));
  }

  const modes = [
    { id: "visual", label: "Visual Match" },
    { id: "style",  label: "Same Style, Any Color" },
    { id: "color",  label: "Same Color, Any Style" },
  ];

  return (
    <div style={{
      padding: "12px 28px",
      borderBottom: `1px solid ${T.borderFaint}`,
      background: T.bgDeep,
    }}>
      {isLoading ? (
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 6, height: 6, borderRadius: "50%", background: T.accent,
            animation: "spin 1.2s linear infinite",
          }} />
          <span style={{ fontSize: "0.65rem", color: T.textMuted, letterSpacing: "0.08em" }}>
            ANALYSING STYLE…
          </span>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {chips.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {chips.map((c, i) => (
                <span key={i} style={{
                  fontSize: "0.62rem", color: T.textMuted,
                  background: T.surface, border: `1px solid ${T.border}`,
                  borderRadius: 20, padding: "3px 10px",
                  letterSpacing: "0.04em", textTransform: "capitalize",
                }}>
                  {c.label}
                </span>
              ))}
            </div>
          )}
          <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
            {modes.map(m => {
              const active = refineMode === m.id || (refineMode === null && m.id === "visual");
              return (
                <button
                  key={m.id}
                  onClick={() => onRefine(m.id)}
                  style={{
                    fontSize: "0.65rem", padding: "5px 14px",
                    borderRadius: 20, cursor: "pointer",
                    border: `1px solid ${active ? T.accent : T.border}`,
                    background: active ? T.accentBg : "transparent",
                    color: active ? T.accent : T.textMuted,
                    
                    transition: "all 0.15s",
                  }}
                >
                  {m.label}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// RESULTS VIEW
// ══════════════════════════════════════════════════════════════════
function ResultsView({ results, categoryInfo, selectedItem, queryImageURL, judgeScores, judging, attributes, refineMode, onRefine, onRefineCrop, radius, setRadius, userLocation, searchLocation, storeLocations = {}, onFeedback, onReset, saved = [], onToggleSave, onFlag, searchWarning }) {
  const [highlightedStore, setHighlightedStore] = useState(null);
  const [activeDetail,     setActiveDetail]     = useState(null);
  const impressionsFired = useRef(false);
  const fallbackLL = [33.8869, 35.5131];
  const gpsLL      = userLocation || fallbackLL;
  // searchLocation = user-chosen pin; falls back to GPS, then hardcoded default
  const userLL     = searchLocation || gpsLL;
  const isCustomLocation = !!(searchLocation);
  const category = categoryInfo?.category || selectedItem?.search_label || "";

  // Fire one batch impression event when results first render
  useEffect(() => {
    if (impressionsFired.current || results.length === 0) return;
    impressionsFired.current = true;
    const impressions = results.map(r => ({
      store:     r.payload?.store     || "",
      item_name: r.payload?.item_name || "",
    }));
    fetch(`${API}/track-event`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ event_type: "impressions", impressions }),
    }).catch(() => {});
  }, [results]);

  const storeResults = {};
  results.forEach(r => {
    const store = r.payload?.store;
    if (store) {
      if (!storeResults[store]) storeResults[store] = [];
      storeResults[store].push(r);
    }
  });

  // Stores with known coordinates that fall within the chosen radius — shown on map
  const storesOnMap = Object.keys(storeResults).filter(store => {
    const coords = storeLocations[store];
    return coords && haversineKm(userLL, coords) <= radius;
  });

  // Stores included in the results list: those on the map PLUS any store with no
  // pinned coordinates (we can't filter what we don't know, so include by default)
  const storesInRadius = Object.keys(storeResults).filter(store => {
    const coords = storeLocations[store];
    if (!coords) return true;
    return haversineKm(userLL, coords) <= radius;
  });

  // Show all results regardless of radius — radius only controls map pins.
  // Nearby-store results are sorted to the top so the user sees closest matches first.
  const visibleResults = [...results].sort((a, b) => {
    const aClose = storesInRadius.includes(a.payload?.store);
    const bClose = storesInRadius.includes(b.payload?.store);
    if (aClose && !bClose) return -1;
    if (!aClose && bClose) return 1;
    return 0;
  });
  const unfilteredResults = highlightedStore
    ? visibleResults.filter(r => r.payload?.store === highlightedStore)
    : visibleResults;

  // Re-sort by AI judge score once available; unjudged items stay at end in CLIP order
  const displayResults = [...unfilteredResults].sort((a, b) => {
    const flatA = a.payload ?? a;
    const flatB = b.payload ?? b;
    const pidA  = flatA.product_id || flatA.item_id || flatA.image_url;
    const pidB  = flatB.product_id || flatB.item_id || flatB.image_url;
    const sA    = judgeScores[pidA];
    const sB    = judgeScores[pidB];
    if (sA !== undefined && sB !== undefined) return sB - sA;
    if (sA !== undefined) return -1;
    if (sB !== undefined) return 1;
    return (b.score ?? 0) - (a.score ?? 0);
  });

  return (
    <div className="fade-in" style={{ minHeight: "100svh", paddingBottom: "calc(83px + env(safe-area-inset-bottom, 0px) + 24px)" }}>

      {/* ── Sticky header bar ── */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "14px 28px",
        borderBottom: `1px solid ${T.borderFaint}`,
        position: "sticky", top: 0, zIndex: 100,
        background: "rgba(242,242,247,0.92)", backdropFilter: "saturate(180%) blur(20px)", WebkitBackdropFilter: "saturate(180%) blur(20px)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <button className="btn-ghost" onClick={onReset} style={{ padding: "6px 10px" }}>← Back</button>
          <div>
            <div className="font-serif" style={{ fontSize: "1.2rem", fontWeight: 300, fontStyle: "italic", color: T.text, letterSpacing: "0.02em", textTransform: "capitalize", lineHeight: 1.2 }}>
              {selectedItem?.label || category || "Results"}
            </div>
            <div style={{ fontSize: "0.63rem", color: T.textMuted, letterSpacing: "0.02em", marginTop: 2, display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
              {displayResults.length} match{displayResults.length !== 1 ? "es" : ""}
              {storesOnMap.length > 0 && ` · ${storesOnMap.length} store${storesOnMap.length > 1 ? "s" : ""} nearby`}
              {isCustomLocation && (
                <span style={{ background: T.accentBg, color: T.accent, border: `1px solid ${T.accentRing}`, borderRadius: 10, padding: "1px 7px", fontSize: "0.6rem", letterSpacing: "0.05em" }}>
                  📍 custom pin
                </span>
              )}
              {judging && (
                <span style={{ display: "inline-flex", alignItems: "center", gap: 4, color: T.accent, fontWeight: 500 }}>
                  · <span style={{ width: 7, height: 7, borderRadius: "50%", border: `1.5px solid ${T.accent}`, borderTopColor: "transparent", display: "inline-block", animation: "spin 0.8s linear infinite" }} /> AI scoring
                </span>
              )}
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "0.6rem", color: T.textFaint, letterSpacing: "0.06em", textTransform: "uppercase" }}>radius</span>
          <input type="range" min={1} max={50} value={radius} onChange={e => setRadius(Number(e.target.value))} style={{ width: 76 }} />
          <span style={{ fontSize: "0.7rem", color: T.accent, minWidth: 34, fontWeight: 500 }}>{radius} km</span>
        </div>
      </div>

      {/* ── Query image strip ── */}
      {queryImageURL && (
        <div style={{
          display: "flex", alignItems: "center", gap: "16px",
          padding: "10px 28px",
          borderBottom: `1px solid ${T.borderFaint}`,
          background: T.bgDeep,
        }}>
          <span style={{ fontSize: "0.62rem", color: T.textMuted, letterSpacing: "0.1em", textTransform: "uppercase", flexShrink: 0 }}>
            You searched
          </span>
          <img
            src={queryImageURL}
            alt="your search crop"
            style={{
              height: 68,
              width: "auto",
              maxWidth: 100,
              objectFit: "cover",
              borderRadius: 8,
              border: `1.5px solid ${T.accent}`,
              flexShrink: 0,
            }}
          />
          <div style={{ fontSize: "0.72rem", color: T.textMuted, lineHeight: 1.6 }}>
            <div style={{ color: T.text, fontWeight: 500, textTransform: "capitalize", marginBottom: 2 }}>
              {CATEGORY_LABELS[selectedItem?.search_label] || selectedItem?.label || "item"}
            </div>
            {categoryInfo?.confidence != null && (
              <div>{Math.round(categoryInfo.confidence * 100)}% confidence</div>
            )}
          </div>
          {onRefineCrop && (
            <button
              onClick={onRefineCrop}
              style={{
                marginLeft: "auto", fontSize: "0.65rem", padding: "5px 12px",
                border: `1px solid ${T.border}`, borderRadius: 20,
                background: "transparent", color: T.textMuted,
                cursor: "pointer", 
                flexShrink: 0,
              }}
            >
              Adjust crop
            </button>
          )}
        </div>
      )}

      {/* ── Attribute panel ── */}
      {attributes !== undefined && (
        <AttributePanel attributes={attributes} refineMode={refineMode} onRefine={onRefine} />
      )}

      <div style={{ padding: "24px 28px" }}>
        {/* Map */}
        <div style={{ borderRadius: 12, overflow: "hidden", border: `1px solid ${T.border}`, marginBottom: 28, height: 220 }}>
          <MapContainer center={userLL} zoom={11} style={{ height: "100%", width: "100%" }} zoomControl={false}>
            <TileLayer url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" attribution="&copy; CartoDB" />
            <MapRecenter center={userLL} />
            {/* All registered stores as faded background pins */}
            {Object.entries(storeLocations).filter(([store]) => !storesOnMap.includes(store)).map(([store, coords]) => (
              <Marker key={`bg-${store}`} position={coords} icon={L.divIcon({
                html: `<div style="width:10px;height:10px;background:${T.textFaint};border-radius:50%;border:1.5px solid #fff;"></div>`,
                className: "", iconSize: [10, 10], iconAnchor: [5, 5],
              })}>
                <Popup><span style={{ fontSize: "0.75rem", color: T.textMuted }}>{store}</span></Popup>
              </Marker>
            ))}
            {/* Stores with matching results — bright pins */}
            {storesOnMap.map(store => {
              const coords = storeLocations[store];
              return (
                <Marker key={store} position={coords} icon={makeStoreIcon(store === highlightedStore)} eventHandlers={{ click: () => setHighlightedStore(store === highlightedStore ? null : store) }}>
                  <Popup>
                    <div style={{ lineHeight: 1.7 }}>
                      <strong style={{ color: T.accent }}>{store}</strong><br />
                      {storeResults[store]?.length} matching item{storeResults[store]?.length !== 1 ? "s" : ""}<br />
                      <span style={{ color: T.textMuted }}>{haversineKm(userLL, coords).toFixed(1)} km away</span>
                    </div>
                  </Popup>
                </Marker>
              );
            })}
            {/* GPS dot (always shown when available) */}
            {userLocation && (
              <Marker position={gpsLL} icon={gpsUserIcon}>
                <Popup><span style={{ fontSize: "0.8rem" }}>Your GPS location</span></Popup>
              </Marker>
            )}
            {/* Custom search pin (shown when user picked a different location) */}
            {isCustomLocation && (
              <Marker position={userLL} icon={searchPinIcon(true)}>
                <Popup>
                  <span style={{ fontSize: "0.8rem", color: T.accent, fontWeight: 600 }}>Search pin</span><br />
                  <span style={{ fontSize: "0.72rem", color: T.textMuted }}>{userLL[0].toFixed(4)}, {userLL[1].toFixed(4)}</span>
                </Popup>
              </Marker>
            )}
          </MapContainer>
        </div>

        {storesOnMap.length === 0 && Object.keys(storeLocations).length > 0 && (
          <div style={{ fontSize: "0.72rem", color: T.textMuted, textAlign: "center", marginTop: "10px" }}>
            No pinned stores within {radius} km · Try increasing the radius.
          </div>
        )}

        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
          <span style={{ fontSize: "0.6rem", color: T.textMuted, letterSpacing: "0.14em", textTransform: "uppercase" }}>
            Matches
            {highlightedStore && (
              <span style={{ color: T.accent, marginLeft: "8px" }}>
                · {highlightedStore}
                <button onClick={() => setHighlightedStore(null)} style={{ background: "none", border: "none", color: T.textMuted, cursor: "pointer", marginLeft: "6px" }}>✕</button>
              </span>
            )}
          </span>
        </div>

        {displayResults.length === 0 ? (
          <div style={{ textAlign: "center", padding: "60px 0", fontSize: "0.8rem", color: T.textMuted, lineHeight: 2 }}>
            {searchWarning
              ? <>No matches found.<br /><span style={{ fontSize: "0.72rem" }}>Try uploading a photo of a clothing item or accessory.</span></>
              : <>No results found.<br /><span style={{ fontSize: "0.72rem" }}>The database may be empty — index items first.</span></>
            }
          </div>
        ) : (
          <div className="results-grid">
            {displayResults.map((result, idx) => {
              const flat     = result.payload ?? result;
              const pid      = flat.product_id || flat.item_id || flat.image_url;
              const position = idx + 1;
              const handleCardClick = (r) => {
                fetch(`${API}/track-event`, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    event_type: "result_click",
                    position,
                    store:     flat.store     || "",
                    item_name: flat.item_name || "",
                  }),
                }).catch(() => {});
                setActiveDetail(r);
              };
              return (
                <ResultCard
                  key={pid}
                  result={result}
                  category={category}
                  onFeedback={onFeedback}
                  onCardClick={handleCardClick}
                  highlighted={flat.store === highlightedStore}
                  judgeScore={judgeScores[pid] ?? null}
                  saved={isItemSaved(pid, saved)}
                  onToggleSave={onToggleSave}
                  onFlag={onFlag}
                />
              );
            })}
          </div>
        )}
      </div>

      {activeDetail && (() => {
        const flat = activeDetail.payload ?? activeDetail;
        const pid  = flat.product_id || flat.item_id || flat.image_url;
        return (
          <ProductDetailSheet
            result={activeDetail}
            category={category}
            judgeScore={judgeScores[pid] ?? null}
            onFeedback={onFeedback}
            onClose={() => setActiveDetail(null)}
            saved={isItemSaved(pid, saved)}
            onToggleSave={onToggleSave}
            storeLocations={storeLocations}
          />
        );
      })()}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// CATEGORY COLOR MAP (shared by Saved + History views)
// ══════════════════════════════════════════════════════════════════
const CATEGORY_COLORS = {
  top:        "#7A9AB0",
  pants:      "#7878A0",
  leggings:   "#7878A0",
  dress:      "#B07898",
  skirt:      "#C09498",
  shorts:     "#B09470",
  sweater:    "#7A9E8C",
  jacket:     "#728880",
  shoes:      "#A08868",
  hat:        "#B0983C",
  bag:        "#A07060",
  jumpsuit:   "#9A6878",
  sports_bra: "#6898B0",
};

// ══════════════════════════════════════════════════════════════════
// SAVED VIEW
// ══════════════════════════════════════════════════════════════════
function SavedView({ saved, onOpenDetail, onToggleSave }) {
  return (
    <div style={{ minHeight: "100svh", padding: "24px 20px", paddingBottom: "calc(83px + env(safe-area-inset-bottom, 0px) + 24px)", maxWidth: 640, margin: "0 auto" }}>
      <div className="fade-up" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
        <div>
          <h2 className="font-serif" style={{ fontSize: "2rem", fontWeight: 400, fontStyle: "italic", color: T.text, letterSpacing: "0.02em" }}>
            Saved
          </h2>
        </div>
        {saved.length > 0 && (
          <span style={{ fontSize: "0.72rem", color: T.textMuted }}>{saved.length} item{saved.length !== 1 ? "s" : ""}</span>
        )}
      </div>

      {saved.length === 0 ? (
        <div className="fade-up" style={{ textAlign: "center", padding: "88px 0", color: T.textMuted }}>
          <div className="font-serif" style={{ fontSize: "2.8rem", fontWeight: 300, fontStyle: "italic", color: T.textFaint, marginBottom: 16, lineHeight: 1 }}>
            empty
          </div>
          <p style={{ fontSize: "0.82rem", color: T.textMuted }}>No saved items yet</p>
          <p style={{ fontSize: "0.72rem", marginTop: 8, color: T.textFaint }}>Tap ♡ on any result to save it here</p>
        </div>
      ) : (
        <div className="results-grid">
          {saved.map(({ productId, savedAt, result }) => {
            const payload  = result.payload ?? {};
            const raw      = payload.image_url ?? "";
            const imageURL = raw ? (raw.startsWith("http") ? raw : `${API}${raw}`) : null;
            const catColor = CATEGORY_COLORS[payload.category_tag] || T.accent;

            return (
              <div
                key={productId}
                style={{
                  background: T.surface, border: `1px solid ${T.border}`, borderRadius: 14,
                  overflow: "hidden", display: "flex", flexDirection: "column",
                  position: "relative", cursor: "pointer",
                  transition: "transform 0.22s cubic-bezier(0.16,1,0.3,1), box-shadow 0.22s cubic-bezier(0.16,1,0.3,1)",
                }}
                onClick={() => onOpenDetail(result)}
                onMouseEnter={e => { e.currentTarget.style.transform = "translateY(-3px)"; e.currentTarget.style.boxShadow = "0 10px 32px rgba(0,0,0,0.08)"; }}
                onMouseLeave={e => { e.currentTarget.style.transform = ""; e.currentTarget.style.boxShadow = ""; }}
              >
                {/* Image */}
                <div style={{ aspectRatio: "3/4", background: T.bgDeep, overflow: "hidden" }}>
                  {imageURL
                    ? <img src={imageURL} alt={payload.item_name || "product"} style={{ width: "100%", height: "100%", objectFit: "cover", transition: "transform 0.5s ease" }}
                        onMouseEnter={e => { e.target.style.transform = "scale(1.05)"; }}
                        onMouseLeave={e => { e.target.style.transform = "scale(1)"; }}
                      />
                    : <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: T.textFaint, fontSize: "1.5rem" }}>✦</div>
                  }
                  {/* Unsave button — frosted, bottom-right */}
                  <button
                    onClick={(e) => { e.stopPropagation(); onToggleSave(result); }}
                    style={{
                      position: "absolute", bottom: 8, right: 8, zIndex: 1,
                      background: T.accent, border: "none", borderRadius: "50%",
                      width: 30, height: 30, cursor: "pointer",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: "0.82rem", color: "#fff",
                    }}
                    title="Unsave"
                  >♥</button>
                </div>

                {/* Info */}
                <div style={{ padding: "11px 13px 13px", display: "flex", flexDirection: "column", gap: 4 }}>
                  <div style={{ height: 2, borderRadius: 1, background: `${catColor}38`, marginBottom: 1 }} />
                  <div style={{ fontSize: "0.77rem", fontWeight: 500, color: T.text, lineHeight: 1.35, letterSpacing: "-0.01em" }}>
                    {payload.item_name || "Product"}
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 1 }}>
                    <span style={{ fontSize: "0.63rem", color: T.textMuted, letterSpacing: "0.02em" }}>{payload.store || ""}</span>
                    {payload.price && <span style={{ fontSize: "0.75rem", color: T.accent, fontWeight: 600 }}>{payload.price}</span>}
                  </div>
                  <div style={{ fontSize: "0.58rem", color: T.textFaint, marginTop: 2, letterSpacing: "0.03em" }}>
                    Saved {timeAgo(savedAt)}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// RATING STATS PANEL
// ══════════════════════════════════════════════════════════════════
function RatingStatsPanel() {
  const [stats,   setStats]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(false);
  const [source,  setSource]  = useState("all"); // "all" | "user" | "auto_judge"

  const load = () => {
    setLoading(true);
    setError(false);
    fetch(`${API}/rating-stats`)
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(data => { setStats(data); setLoading(false); })
      .catch(() => { setError(true); setLoading(false); });
  };

  useEffect(() => { load(); }, []);

  if (loading) {
    return (
      <div style={{ padding: "32px 24px", maxWidth: 1100, margin: "0 auto" }}>
        <div style={{ color: T.textMuted, fontSize: "0.85rem" }}>Loading rating stats…</div>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div style={{ padding: "32px 24px", maxWidth: 1100, margin: "0 auto" }}>
        <div style={{
          background: T.surface, border: `1px solid ${T.border}`,
          borderRadius: 10, padding: "20px 24px",
          display: "flex", alignItems: "center", gap: 16,
        }}>
          <span style={{ color: T.red, fontSize: "0.85rem" }}>
            Could not load rating stats from gateway.
          </span>
          <button onClick={load} style={{
            padding: "5px 14px", borderRadius: 20,
            border: `1px solid ${T.border}`, background: "transparent",
            color: T.textMuted, fontSize: "0.78rem", cursor: "pointer",
          }}>Retry</button>
        </div>
      </div>
    );
  }

  const byStarMap = source === "user"
    ? stats.by_star_user
    : source === "auto_judge"
      ? stats.by_star_judge
      : stats.by_star;

  const sourceTotal = source === "user"
    ? stats.by_source.user
    : source === "auto_judge"
      ? stats.by_source.auto_judge
      : stats.total;

  const maxCount = Math.max(...Object.values(byStarMap), 1);

  const STAR_COLOR = "#c9a96e";
  const SIGNAL_COLORS = {
    positive: { bg: "rgba(122,171,138,0.12)", border: "rgba(122,171,138,0.3)", text: T.green },
    negative: { bg: "rgba(201,112,112,0.12)", border: "rgba(201,112,112,0.3)",  text: T.red },
    neutral:  { bg: "rgba(201,169,110,0.08)", border: "rgba(201,169,110,0.2)",  text: T.yellow },
  };

  const filledStars = (n) =>
    Array.from({ length: 5 }, (_, i) => (
      <span key={i} style={{ color: i < Math.round(n) ? STAR_COLOR : T.textFaint, fontSize: "1rem" }}>★</span>
    ));

  const userPct  = stats.total > 0 ? ((stats.by_source.user  / stats.total) * 100).toFixed(1) : "0";
  const judgePct = stats.total > 0 ? ((stats.by_source.auto_judge / stats.total) * 100).toFixed(1) : "0";

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px 0" }}>

      {/* ── Header ── */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 14, marginBottom: 24 }}>
        <h2 style={{
          fontWeight: 600, letterSpacing: "-0.02em",
          fontSize: "1.6rem", color: T.text,
        }}>
          Ratings Overview
        </h2>
        <span style={{ color: T.textMuted, fontSize: "0.82rem" }}>
          {stats.total.toLocaleString()} total · avg {stats.avg_rating.toFixed(1)} ★ · {filledStars(stats.avg_rating)}
        </span>
      </div>

      {stats.total === 0 ? (
        <div style={{ textAlign: "center", padding: "48px 0", color: T.textMuted, fontSize: "0.85rem" }}>
          <div style={{ fontSize: "2rem", marginBottom: 10, opacity: 0.3 }}>★</div>
          No ratings yet
        </div>
      ) : (
        <>
          {/* ── Source breakdown (the two numbers you're looking for) ── */}
          <div style={{
            background: T.surface, border: `1px solid ${T.border}`,
            borderRadius: 12, padding: "20px 24px", marginBottom: 16,
          }}>
            <div style={{ color: T.textMuted, fontSize: "0.7rem", fontWeight: 700,
              textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 16 }}>
              Rating Source
            </div>

            {/* Big numbers side by side */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>

              {/* Actual users */}
              <div style={{
                background: "rgba(122,171,138,0.07)", border: "1px solid rgba(122,171,138,0.25)",
                borderRadius: 10, padding: "16px 20px",
              }}>
                <div style={{ color: T.green, fontSize: "0.7rem", fontWeight: 700,
                  textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 6 }}>
                  Actual User Ratings
                </div>
                <div style={{ color: T.text, fontSize: "2.2rem", fontWeight: 700, lineHeight: 1 }}>
                  {stats.by_source.user.toLocaleString()}
                </div>
                <div style={{ color: T.textMuted, fontSize: "0.78rem", marginTop: 6 }}>
                  {userPct}% of total — submitted by real shoppers
                </div>
              </div>

              {/* AI judge */}
              <div style={{
                background: "rgba(201,169,110,0.07)", border: "1px solid rgba(201,169,110,0.2)",
                borderRadius: 10, padding: "16px 20px",
              }}>
                <div style={{ color: T.accent, fontSize: "0.7rem", fontWeight: 700,
                  textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 6 }}>
                  AI Judge Ratings
                </div>
                <div style={{ color: T.text, fontSize: "2.2rem", fontWeight: 700, lineHeight: 1 }}>
                  {stats.by_source.auto_judge.toLocaleString()}
                </div>
                <div style={{ color: T.textMuted, fontSize: "0.78rem", marginTop: 6 }}>
                  {judgePct}% of total — automated quality scoring
                </div>
              </div>
            </div>

            {/* Visual split bar */}
            <div style={{ height: 6, borderRadius: 3, background: T.bgDeep, overflow: "hidden" }}>
              <div style={{
                height: "100%", width: `${userPct}%`,
                background: T.green, borderRadius: 3,
              }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 5 }}>
              <span style={{ color: T.green, fontSize: "0.7rem" }}>Users {userPct}%</span>
              <span style={{ color: T.accent, fontSize: "0.7rem" }}>AI Judge {judgePct}%</span>
            </div>
          </div>

          {/* ── Ready-to-train pill ── */}
          <div style={{ marginBottom: 20 }}>
            <span style={{
              display: "inline-block",
              background: stats.ready_to_train ? "rgba(122,171,138,0.12)" : "rgba(201,112,112,0.1)",
              border: `1px solid ${stats.ready_to_train ? "rgba(122,171,138,0.3)" : "rgba(201,112,112,0.3)"}`,
              color: stats.ready_to_train ? T.green : T.red,
              fontSize: "0.75rem", fontWeight: 600,
              padding: "4px 14px", borderRadius: 20,
            }}>
              {stats.ready_to_train ? "Ready to train" : "Not enough pairs to train"}
            </span>
          </div>

          {/* ── Star distribution ── */}
          <div style={{
            background: T.surface, border: `1px solid ${T.border}`,
            borderRadius: 12, padding: "18px 24px", marginBottom: 16,
          }}>
            {/* Section header + filter tabs inline */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
              <div style={{ color: T.textMuted, fontSize: "0.7rem", fontWeight: 700,
                textTransform: "uppercase", letterSpacing: "0.07em" }}>
                Star Distribution
              </div>
              <div style={{ display: "flex", gap: 4 }}>
                {[
                  { key: "all",        label: "All" },
                  { key: "user",       label: "Users only" },
                  { key: "auto_judge", label: "AI Judge only" },
                ].map(opt => (
                  <button key={opt.key} onClick={() => setSource(opt.key)} style={{
                    padding: "3px 10px", borderRadius: 20, border: "1px solid",
                    fontSize: "0.72rem", cursor: "pointer", transition: "all 0.15s",
                    background:  source === opt.key ? T.accentBg  : "transparent",
                    borderColor: source === opt.key ? T.accent     : T.border,
                    color:       source === opt.key ? T.accent      : T.textMuted,
                    fontWeight:  source === opt.key ? 600 : 400,
                  }}>
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {[5, 4, 3, 2, 1].map(star => {
              const count = byStarMap[String(star)] || 0;
              const pct   = sourceTotal > 0 ? ((count / sourceTotal) * 100).toFixed(1) : "0.0";
              const barW  = sourceTotal > 0 ? (count / maxCount) * 100 : 0;
              return (
                <div key={star} style={{
                  display: "grid",
                  gridTemplateColumns: "72px 1fr 56px 44px",
                  alignItems: "center", gap: 10, marginBottom: 9,
                }}>
                  <div style={{ display: "flex", gap: 2, justifyContent: "flex-end" }}>
                    {Array.from({ length: 5 }, (_, i) => (
                      <span key={i} style={{ color: i < star ? STAR_COLOR : T.textFaint, fontSize: "0.78rem" }}>★</span>
                    ))}
                  </div>
                  <div style={{ height: 8, borderRadius: 4, background: T.bgDeep, overflow: "hidden" }}>
                    <div style={{
                      height: "100%", width: `${barW}%`,
                      background: `linear-gradient(90deg, ${T.accent}, ${T.accentDeep})`,
                      borderRadius: 4, transition: "width 0.4s ease",
                    }} />
                  </div>
                  <div style={{ color: T.text, fontSize: "0.82rem", fontWeight: 600, textAlign: "right" }}>
                    {count.toLocaleString()}
                  </div>
                  <div style={{ color: T.textMuted, fontSize: "0.75rem", textAlign: "right" }}>
                    {pct}%
                  </div>
                </div>
              );
            })}
          </div>

          {/* ── Training signal ── */}
          <div style={{
            background: T.surface, border: `1px solid ${T.border}`,
            borderRadius: 12, padding: "18px 24px", marginBottom: 28,
          }}>
            <div style={{ color: T.textMuted, fontSize: "0.7rem", fontWeight: 700,
              textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 14 }}>
              Training Signal Breakdown
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
              {[
                { key: "positive", label: "Positive",  desc: "4–5 ★", bg: "rgba(122,171,138,0.1)",  border: "rgba(122,171,138,0.25)", color: T.green },
                { key: "neutral",  label: "Neutral",   desc: "3 ★",   bg: "rgba(201,169,110,0.07)", border: "rgba(201,169,110,0.2)",  color: T.yellow },
                { key: "negative", label: "Negative",  desc: "1–2 ★", bg: "rgba(201,112,112,0.1)",  border: "rgba(201,112,112,0.25)", color: T.red },
              ].map(sig => {
                const cnt = stats.by_signal[sig.key];
                const pct = stats.total > 0 ? ((cnt / stats.total) * 100).toFixed(0) : "0";
                return (
                  <div key={sig.key} style={{
                    background: sig.bg, border: `1px solid ${sig.border}`, borderRadius: 10, padding: "14px 16px",
                  }}>
                    <div style={{ color: sig.color, fontSize: "0.7rem", fontWeight: 700,
                      textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>
                      {sig.label} · {sig.desc}
                    </div>
                    <div style={{ color: T.text, fontSize: "1.5rem", fontWeight: 700 }}>
                      {cnt.toLocaleString()}
                    </div>
                    <div style={{ color: T.textMuted, fontSize: "0.75rem", marginTop: 3 }}>
                      {pct}% of total
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}


// ══════════════════════════════════════════════════════════════════
// ADMIN FLAGS VIEW
// ══════════════════════════════════════════════════════════════════
function AdminFlagsView({ onFlagsChange }) {
  const [flags,   setFlags]   = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy,    setBusy]    = useState({});

  useEffect(() => {
    fetch(`${API}/low-score-flags`)
      .then(r => r.json())
      .then(data => { setFlags(data.flags || []); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const act = async (productId, endpoint) => {
    setBusy(prev => ({ ...prev, [productId]: true }));
    try {
      const res = await fetch(`${API}/low-score-flags/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product_id: productId }),
      });
      if (res.ok) {
        setFlags(prev => {
          const next = prev.filter(f => f.product_id !== productId);
          onFlagsChange?.(next.length);
          return next;
        });
      }
    } catch { /* non-critical */ } finally {
      setBusy(prev => ({ ...prev, [productId]: false }));
    }
  };

  if (loading) {
    return (
      <div style={{ padding: 40, textAlign: "center", color: T.textMuted, fontSize: "0.85rem" }}>
        Loading flags…
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px" }}>
      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
          <h2 style={{
            fontWeight: 600, letterSpacing: "-0.02em",
            fontSize: "1.6rem", color: T.text, margin: 0,
          }}>
            Low-Quality Match Flags
          </h2>
          {flags.length > 0 && (
            <span style={{
              background: "rgba(201,112,112,0.12)", border: "1px solid rgba(201,112,112,0.3)",
              color: T.red, fontSize: "0.72rem", fontWeight: 700,
              padding: "3px 10px", borderRadius: 20,
            }}>
              {flags.length} warning{flags.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>
        <p style={{ color: T.textMuted, fontSize: "0.82rem", lineHeight: 1.5 }}>
          {flags.length === 0
            ? "No issues detected — all top-3 results are meeting quality thresholds."
            : `${flags.length} item${flags.length !== 1 ? "s" : ""} were flagged because Gemini judge scored them below 40% in a top-3 position. They are hidden from search results until reviewed.`}
        </p>
      </div>

      {flags.length === 0 && (
        <div style={{
          textAlign: "center", padding: "60px 0",
          color: T.textMuted, fontSize: "0.85rem",
        }}>
          <div style={{ fontSize: "2rem", marginBottom: 12, opacity: 0.3 }}>✓</div>
          All clear
        </div>
      )}

      {/* Flag grid */}
      {flags.length > 0 && (
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
          gap: 16,
        }}>
          {flags.map(flag => {
            const pct   = Math.round((flag.judge_score ?? 0) * 100);
            const color = pct >= 35 ? T.yellow : T.red;
            const isBusy = !!busy[flag.product_id];
            return (
              <div key={flag.product_id} style={{
                background: T.surface,
                border: `1px solid ${T.border}`,
                borderRadius: 12,
                overflow: "hidden",
                opacity: isBusy ? 0.5 : 1,
                transition: "opacity 0.2s",
              }}>
                {/* Image */}
                <div style={{ position: "relative", aspectRatio: "3/4", background: T.bgDeep }}>
                  {flag.image_url ? (
                    <img
                      src={flag.image_url}
                      alt={flag.name}
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                      onError={e => { e.target.style.display = "none"; }}
                    />
                  ) : (
                    <div style={{
                      width: "100%", height: "100%",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      color: T.textFaint, fontSize: "0.75rem",
                    }}>No image</div>
                  )}
                  {/* Score badge */}
                  <div style={{
                    position: "absolute", top: 8, right: 8,
                    background: `${color}22`, border: `1px solid ${color}55`,
                    color, fontSize: "0.68rem", fontWeight: 700,
                    padding: "3px 7px", borderRadius: 6,
                    backdropFilter: "blur(4px)",
                  }}>
                    {pct}%
                  </div>
                </div>

                {/* Meta */}
                <div style={{ padding: "11px 13px" }}>
                  <div style={{
                    fontSize: "0.78rem", color: T.text, fontWeight: 500,
                    marginBottom: 3,
                    whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                  }}>
                    {flag.name || "Unknown item"}
                  </div>
                  <div style={{ fontSize: "0.7rem", color: T.textMuted, marginBottom: 11 }}>
                    {flag.store_name || "Unknown store"}
                    {flag.flagged_at && (
                      <span style={{ marginLeft: 6, opacity: 0.6 }}>
                        · {new Date(flag.flagged_at).toLocaleDateString()}
                      </span>
                    )}
                  </div>

                  {/* Actions */}
                  <div style={{ display: "flex", gap: 6 }}>
                    <button
                      disabled={isBusy}
                      onClick={() => act(flag.product_id, "dismiss")}
                      style={{
                        flex: 1, padding: "5px 0",
                        background: T.accentBg, border: `1px solid ${T.accentRing}`,
                        color: T.accent, borderRadius: 6, fontSize: "0.7rem",
                        fontFamily: "'DM Sans', sans-serif",
                        cursor: isBusy ? "default" : "pointer",
                      }}
                    >
                      Dismiss
                    </button>
                    <button
                      disabled={isBusy}
                      onClick={() => act(flag.product_id, "confirm-remove")}
                      style={{
                        flex: 1, padding: "5px 0",
                        background: "rgba(201,112,112,0.08)",
                        border: "1px solid rgba(201,112,112,0.25)",
                        color: T.red, borderRadius: 6, fontSize: "0.7rem",
                        fontFamily: "'DM Sans', sans-serif",
                        cursor: isBusy ? "default" : "pointer",
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
// ROOT
// ══════════════════════════════════════════════════════════════════
export default function App() {
  const [view,             setView]             = useState("landing");
  const [activeTab,        setActiveTab]        = useState("Discover");
  const [imageFile,        setImageFile]        = useState(null);
  const [imageURL,         setImageURL]         = useState(null);
  const [drawnBbox,        setDrawnBbox]        = useState(null);  // {x1,y1,x2,y2} natural px
  const [cropBlob,         setCropBlob]         = useState(null);  // Blob from DrawView crop
  const [predictedCategory,setPredictedCategory]= useState(null);
  const [allScores,        setAllScores]        = useState({});
  const [selectedItem,     setSelectedItem]     = useState(null);
  const [results,          setResults]          = useState([]);
  const [categoryInfo,     setCategoryInfo]     = useState(null);
  const [queryImageURL,    setQueryImageURL]    = useState(null);
  const [searchId,         setSearchId]         = useState(null);
  const [searchWarning,    setSearchWarning]    = useState(null);
  const [judgeScores,      setJudgeScores]      = useState({});   // {product_id: float}
  const [judging,          setJudging]          = useState(false); // true while AI scoring is in progress
  const [attributes,       setAttributes]       = useState(null); // null=loading, {}=failed/empty, obj=ready
  const [refineMode,       setRefineMode]       = useState(null); // "visual"|"style"|"color"|null
  const [radius,              setRadius]              = useState(5);
  const [userLocation,        setUserLocation]        = useState(null);
  const [searchLocation,      setSearchLocation]      = useState(null); // null = use GPS
  const [showLocationPicker,  setShowLocationPicker]  = useState(false);
  const [storeLocations,      setStoreLocations]      = useState({});
  const [discoverDetail,   setDiscoverDetail]   = useState(null);
  const [error,            setError]            = useState(null);
  const [history,          setHistory]          = useState(() => loadHistory());
  const [saved,            setSaved]            = useState(() => loadSaved());
  useEffect(() => {
    navigator.geolocation.getCurrentPosition(
      pos => setUserLocation([pos.coords.latitude, pos.coords.longitude]),
      ()  => setUserLocation([33.8869, 35.5131]),
      { enableHighAccuracy: true, timeout: 8000 }
    );
  }, []);
  useEffect(() => {
    fetch(`${API}/stores/locations`)
      .then(r => r.ok ? r.json() : {})
      .then(data => setStoreLocations(data))
      .catch(() => {});
  }, []);

  // Poll for OpenRouter judge scores after results are shown.
  useEffect(() => {
    if (view !== "results" || !searchId) return;
    setJudging(true);
    let attempts = 0;
    const total = results.length || 15;
    const interval = setInterval(async () => {
      attempts++;
      if (attempts > 75) { clearInterval(interval); setJudging(false); return; }
      try {
        const res = await fetch(`${API}/judge-scores/${searchId}`);
        if (!res.ok) return;
        const scores = await res.json();
        if (Object.keys(scores).length > 0) setJudgeScores(scores);
        if (Object.keys(scores).length >= total) { clearInterval(interval); setJudging(false); }
      } catch { /* non-critical */ }
    }, 4000);
    return () => { clearInterval(interval); setJudging(false); };
  }, [view, searchId]);

  // Track time spent on results page and fire event on exit.
  useEffect(() => {
    if (view !== "results") return;
    const entryTime = Date.now();
    return () => {
      const seconds = (Date.now() - entryTime) / 1000;
      if (seconds < 1) return;
      fetch(`${API}/track-event`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event_type: "results_exit", duration_seconds: seconds }),
      }).catch(() => {});
    };
  }, [view]);

  // Poll for attribute tagger results after search; populates the attribute panel.
  useEffect(() => {
    if (view !== "results" || !searchId) return;
    setAttributes(null);
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      if (attempts > 30) { clearInterval(interval); setAttributes({}); return; }
      try {
        const res = await fetch(`${API}/search/${searchId}/attributes`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.status === "ready")     { setAttributes(data.attributes || {}); clearInterval(interval); }
        else if (data.status === "not_found") { setAttributes({}); clearInterval(interval); }
      } catch { /* non-critical */ }
    }, 800);
    return () => clearInterval(interval);
  }, [view, searchId]);

  const prepareImage = (file) => new Promise((resolve, reject) => {
    const MAX = 800;
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      let { naturalWidth: w, naturalHeight: h } = img;
      if (w > MAX || h > MAX) {
        if (w > h) { h = Math.round(h * MAX / w); w = MAX; }
        else        { w = Math.round(w * MAX / h); h = MAX; }
      }
      const canvas = document.createElement("canvas");
      canvas.width = w; canvas.height = h;
      canvas.getContext("2d").drawImage(img, 0, 0, w, h);
      canvas.toBlob(blob => {
        if (!blob) { reject(new Error("Canvas toBlob failed")); return; }
        resolve(new File([blob], "photo.jpg", { type: "image/jpeg" }));
      }, "image/jpeg", 0.88);
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("Image load failed")); };
    img.src = url;
  });

  const handleUpload = useCallback(async (file) => {
    setError(null);
    let prepared;
    try { prepared = await prepareImage(file); } catch { prepared = file; }
    setImageFile(prepared);
    setImageURL(URL.createObjectURL(prepared));
    setView("drawing");
  }, []);

  // Called by DrawView once the user has drawn a box and CLIP has classified it
  const handleDrawConfirm = useCallback(({ bbox, cropBlob: blob, category, allScores: scores }) => {
    setDrawnBbox(bbox);
    setCropBlob(blob);
    setPredictedCategory(category);
    setAllScores(scores || {});
    setView("confirming");
  }, []);

  // Called by ConfirmView once the user has approved / corrected the category
  const handleConfirm = useCallback(async (confirmedCategory) => {
    setView("searching");
    const { x1, y1, x2, y2 } = drawnBbox;

    // Turn the crop blob into an object URL for the results strip
    if (cropBlob) {
      setQueryImageURL(prev => { if (prev) URL.revokeObjectURL(prev); return URL.createObjectURL(cropBlob); });
    }
    setSelectedItem({ search_label: confirmedCategory, label: CATEGORY_LABELS[confirmedCategory] || confirmedCategory, bbox: [x1, y1, x2, y2] });

    // Compute nearby stores for radius-aware search
    const loc = searchLocation ?? userLocation;
    const storeHasCoords = Object.keys(storeLocations).length > 0;
    const nearbyStores = (loc && storeHasCoords)
      ? Object.entries(storeLocations)
          .filter(([, coords]) => haversineKm(loc, coords) <= radius)
          .map(([name]) => name)
      : [];
    // If location is known but no stores are within radius, proceed without radius filter
    // so the user still gets results (they can filter by radius from the results view).

    const form = new FormData();
    form.append("file", imageFile);
    form.append("x1", x1); form.append("y1", y1);
    form.append("x2", x2); form.append("y2", y2);
    form.append("search_label", confirmedCategory);
    if (nearbyStores.length > 0) form.append("nearby_stores", nearbyStores.join(","));
    try {
      const res = await fetch(`${API}/search`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const reshaped = (data.matches || data.results || []).map(m => ({
        score: m.score,
        payload: {
          store:        m.store        || m.store_name,
          image_url:    m.image_url    || m.image_filename,
          item_name:    m.name,
          product_id:   m.product_id,
          item_id:      m.item_id,
          price:        m.price        || "",
          mall_name:    m.mall_name    || "",
          category_tag: m.category_tag || "",
          maps_url:     m.maps_url     || "",
        }
      }));
      const seen = new Map();
      for (const r of reshaped) {
        const id = r.payload.product_id || r.payload.item_id || r.payload.image_url;
        if (!seen.has(id) || r.score > seen.get(id).score) seen.set(id, r);
      }
      const deduped = Array.from(seen.values()).sort((a, b) => b.score - a.score);
      setResults(deduped);
      saveHistoryEntry(confirmedCategory, cropBlob, deduped).then(updated => setHistory(updated));
      const detCat  = data.detected_category;
      const confMap = data.category_confidence;
      const confVal = (confMap && typeof confMap === "object") ? (confMap[detCat] ?? 0) : 0;
      setCategoryInfo({ category: detCat, confidence: confVal });
      setSearchId(data.search_id || null);
      setSearchWarning(data.warning || null);
      setJudgeScores({});
      setAttributes(null);
      setRefineMode(null);
      setView("results");
    } catch (e) {
      setError(`Search failed: ${e.message}`);
      setView("confirming");
    }
  }, [imageFile, drawnBbox, cropBlob, searchLocation, userLocation, storeLocations, radius]);

  const handleFeedback = useCallback(async (productId, rating, name, store, category, imageUrl) => {
    try {
      await fetch(`${API}/feedback`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          result_product_id: productId || "",
          result_image_url:  imageUrl  || "",
          result_name:       name      || "",
          store_name:        store     || "",
          category:          category  || "",
          rating,
        }),
      });
    } catch { /* non-critical */ }
  }, []);

  const handleFlagItem = useCallback(async (productId, name, store, imageUrl) => {
    try {
      await fetch(`${API}/admin/flag-item`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ product_id: productId, name, store_name: store, image_url: imageUrl }),
      });
    } catch { /* non-critical */ }
  }, []);

  const handleRefine = useCallback(async (mode) => {
    console.log("[REFINE] clicked mode=", mode, "searchId=", searchId, "attributes=", attributes);
    if (!searchId) { console.warn("[REFINE] aborted — searchId is null"); return; }
    setRefineMode(mode);
    try {
      const res = await fetch(`${API}/refine`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          search_id: searchId,
          mode,
          attributes: attributes || {},
          category:   categoryInfo?.category || selectedItem?.search_label || "",
        }),
      });
      if (!res.ok) {
        const txt = await res.text();
        console.error("[REFINE] HTTP error", res.status, txt);
        return;
      }
      const data = await res.json();
      console.log("[REFINE] response:", data.mode, "results:", data.results?.length, "query:", data.query_text);
      const reshaped = (data.results || []).map(m => ({
        score:   m.score,
        payload: {
          store:      m.store      || m.store_name,
          image_url:  m.image_url  || m.image_filename,
          item_name:  m.name,
          product_id: m.product_id,
          price:      m.price      || "",
          mall_name:  m.mall_name  || "",
          maps_url:   m.maps_url   || "",
        },
      }));
      setResults(reshaped);
    } catch (err) { console.error("[REFINE] fetch threw:", err); }
  }, [searchId, attributes, categoryInfo, selectedItem]);

  const handleDiscoverCardClick = useCallback((item) => {
    setDiscoverDetail({
      score: 0,
      payload: {
        product_id:   item.product_id,
        name:         item.name,
        item_name:    item.name,
        image_url:    item.image_url,
        store_name:   item.store,
        store:        item.store,
        mall_name:    item.mall,
        price:        item.price,
        category_tag: item.category,
      },
    });
  }, []);

  const handleToggleSave = useCallback((result) => {
    setSaved(prev => toggleSavedItem(result, prev));
  }, []);

  const handleRefineCrop = useCallback(() => {
    setResults([]);
    setSearchId(null); setSearchWarning(null);
    setAttributes(null);
    setRefineMode(null);
    setJudgeScores({});
    setView("drawing");
  }, []);

  const reset = useCallback(() => {
    setView("landing");
    setImageFile(null);
    if (imageURL)      URL.revokeObjectURL(imageURL);
    if (queryImageURL) URL.revokeObjectURL(queryImageURL);
    setImageURL(null);
    setQueryImageURL(null);
    setDrawnBbox(null);
    setCropBlob(null);
    setPredictedCategory(null);
    setAllScores({});
    setResults([]);
    setSelectedItem(null);
    setCategoryInfo(null);
    setSearchId(null); setSearchWarning(null);
    setJudgeScores({});
    setError(null);
  }, [imageURL, queryImageURL]);

  return (
    <>
      <StyleInjector />
      {showLocationPicker && (
        <LocationPickerModal
          userLocation={userLocation}
          storeLocations={storeLocations}
          currentSearchLocation={searchLocation ?? userLocation}
          onConfirm={(loc) => { setSearchLocation(loc); setShowLocationPicker(false); }}
          onClose={() => setShowLocationPicker(false)}
        />
      )}
      {/* Content sits above the fixed bottom tab bar */}
      <div style={{ paddingTop: "env(safe-area-inset-top, 0px)" }}>

      {activeTab === "Store" ? (
        <StoreDashboardView />
      ) : activeTab === "Saved" ? (
        <SavedView
          saved={saved}
          onOpenDetail={(result) => {
            setResults([result]);
            setPredictedCategory((result.payload ?? {}).category_tag || "");
            setCategoryInfo({ category: (result.payload ?? {}).category_tag || "", confidence: null });
            setSelectedItem({ search_label: (result.payload ?? {}).category_tag || "", label: CATEGORY_LABELS[(result.payload ?? {}).category_tag] || "item" });
            setQueryImageURL(null);
            setJudgeScores({});
            setSearchId(null); setSearchWarning(null);
            setView("results");
            setActiveTab("Discover");
          }}
          onToggleSave={handleToggleSave}
        />
      ) : (
        <>
          {view === "landing"    && <LandingView onUpload={handleUpload} error={error} history={history} saved={saved} onToggleSave={handleToggleSave} onCardClick={handleDiscoverCardClick} onFeedback={handleFeedback} />}
          {view === "drawing"    && <DrawView imageURL={imageURL} imageFile={imageFile} onConfirm={handleDrawConfirm} onBack={reset} />}
          {view === "confirming" && <ConfirmView cropBlob={cropBlob} predictedCategory={predictedCategory} allScores={allScores} onSearch={handleConfirm} onBack={() => setView("drawing")} searchLocation={searchLocation} userLocation={userLocation} onOpenLocationPicker={() => setShowLocationPicker(true)} />}
          {view === "searching"  && <LoadingView label="Finding matches…" />}
          {view === "results"    && (
            <ResultsView
              results={results}
              categoryInfo={categoryInfo}
              selectedItem={selectedItem}
              queryImageURL={queryImageURL}
              judgeScores={judgeScores}
              judging={judging}
              attributes={attributes}
              refineMode={refineMode}
              onRefine={handleRefine}
              radius={radius}
              setRadius={setRadius}
              userLocation={userLocation}
              searchLocation={searchLocation}
              storeLocations={storeLocations}
              onFeedback={handleFeedback}
              onFlag={handleFlagItem}
              onReset={reset}
              onRefineCrop={handleRefineCrop}
              saved={saved}
              onToggleSave={handleToggleSave}
              searchWarning={searchWarning}
            />
          )}
        </>
      )}
      </div>
      {discoverDetail && (() => {
        const pid = (discoverDetail.payload || {}).product_id || (discoverDetail.payload || {}).image_url;
        return (
          <ProductDetailSheet
            result={discoverDetail}
            category={(discoverDetail.payload || {}).category_tag}
            judgeScore={null}
            onFeedback={handleFeedback}
            onClose={() => setDiscoverDetail(null)}
            saved={isItemSaved(pid, saved)}
            onToggleSave={handleToggleSave}
            storeLocations={storeLocations}
          />
        );
      })()}
      <Navbar activeTab={activeTab} onTab={setActiveTab} />
    </>
  );
}