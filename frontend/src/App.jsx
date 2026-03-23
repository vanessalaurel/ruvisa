import { useState, useEffect, useRef, useCallback, memo, useMemo } from "react";
import * as api from "./api";
import { Home, Search, ShoppingCart, ShoppingBag, ScanFace, Sparkles, User } from "lucide-react";
import FaceScanCTA from "./components/FaceScanCTA.jsx";

const COLORS = {
  /* Neutral warm ivory / cream (bases) */
  cream: "#F9F7F3",
  creamDark: "#E8E4DC",
  /* Rose gold = copper / bronze with warm gold (not pink-mauve) */
  sage: "#B8876F",
  sageDark: "#6B4538",
  sageLight: "#D4C4B0",
  forest: "#3D2E28",
  text: "#2A2826",
  textLight: "#6E6964",
  white: "#FFFFFF",
  accent: "#C9A56C",
  accentLight: "#E8D4B8",
  pink: "#C9A088",
  red: "#D45B5B",
  green: "#B8876F",
  gold: "#C9A56C",
  blue: "#6B9FC4",
  navBar: "rgba(107, 69, 56, 0.48)",
  navScanSolid: "#4A352E",
  navScanSolidMid: "#6B4E42",
  overlaySoft: "rgba(184, 135, 111, 0.14)",
  overlayMuted: "rgba(184, 135, 111, 0.09)",
  overlayMedium: "rgba(184, 135, 111, 0.11)",
  borderRose: "rgba(140, 95, 72, 0.38)",
  shadowWarm: "rgba(45, 32, 26, 0.22)",
  overlayDeep: "rgba(75, 48, 38, 0.28)",
};

const CONCERN_LABELS = {
  acne: "Acne", comedonal_acne: "Comedonal Acne", pigmentation: "Pigmentation",
  acne_scars_texture: "Scars/Texture", pores: "Pores", redness: "Redness", wrinkles: "Wrinkles",
};

const skinTypes = [
  { id: "dry", label: "Dry", emoji: "🏜️", desc: "Tight, flaky, rough texture" },
  { id: "oily", label: "Oily", emoji: "✨", desc: "Shiny, enlarged pores, acne-prone" },
  { id: "combination", label: "Combination", emoji: "🔄", desc: "Oily T-zone, dry cheeks" },
  { id: "sensitive", label: "Sensitive", emoji: "🌸", desc: "Easily irritated, redness-prone" },
];

const skinConcerns = [
  { id: "acne", label: "Acne", emoji: "🔴" },
  { id: "wrinkles", label: "Wrinkles", emoji: "〰️" },
  { id: "pigmentation", label: "Dark Spots", emoji: "🟤" },
  { id: "pores", label: "Large Pores", emoji: "⭕" },
  { id: "redness", label: "Redness", emoji: "🔺" },
  { id: "acne_scars_texture", label: "Scars/Texture", emoji: "🩹" },
  { id: "comedonal_acne", label: "Blackheads", emoji: "⚫" },
];

/* ---- Shared Components ---- */

function CircularProgress({ value, size = 80, strokeWidth = 6, color = COLORS.sage, centerLabel, showPercent = true }) {
  const radius = (size - strokeWidth) / 2;
  const circ = 2 * Math.PI * radius;
  const offset = circ - (value / 100) * circ;
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke={COLORS.creamDark} strokeWidth={strokeWidth} />
        <circle cx={size / 2} cy={size / 2} r={radius} fill="none" stroke={color} strokeWidth={strokeWidth} strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round" style={{ transition: "stroke-dashoffset 1s ease" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700, fontSize: size * 0.22, color: COLORS.text }}>
        {centerLabel ?? (showPercent ? `${Math.round(value)}%` : null)}
      </div>
    </div>
  );
}

function MatchBadge({ value }) {
  const bg = value >= 90 ? COLORS.overlaySoft : value >= 80 ? "#FFF8E1" : "#FFF3E0";
  const color = value >= 90 ? COLORS.sageDark : value >= 80 ? "#F57F17" : "#E65100";
  return <span style={{ background: bg, color, padding: "3px 10px", borderRadius: 20, fontSize: 12, fontWeight: 700 }}>{value}% Match</span>;
}

function Btn({ children, onClick, disabled, style }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      width: "100%", padding: "16px", borderRadius: 16, border: "none",
      background: disabled ? COLORS.creamDark : COLORS.sage,
      color: disabled ? COLORS.textLight : COLORS.white,
      fontSize: 16, fontWeight: 700, cursor: disabled ? "default" : "pointer",
      transition: "all 0.3s", ...style,
    }}>{children}</button>
  );
}

function BackHeader({ onBack, title, right }) {
  return (
    <div style={{ padding: "50px 24px 16px", display: "flex", alignItems: "center", gap: 12 }}>
      <button onClick={onBack} style={{ background: COLORS.white, border: "none", borderRadius: 12, width: 40, height: 40, cursor: "pointer", fontSize: 18, display: "flex", alignItems: "center", justifyContent: "center" }}>←</button>
      <h2 style={{ fontSize: 22, fontWeight: 800, color: COLORS.forest, margin: 0, flex: 1 }}>{title}</h2>
      {right}
    </div>
  );
}

function ProductImage({ url, size = 80 }) {
  const [err, setErr] = useState(false);
  if (!url || err) {
    return (
      <div style={{ width: size, height: size, borderRadius: 12, background: COLORS.creamDark, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
        <span style={{ fontSize: size * 0.4 }}>🧴</span>
      </div>
    );
  }
  return <img src={url} alt="" onError={() => setErr(true)} style={{ width: size, height: size, borderRadius: 12, objectFit: "cover", flexShrink: 0, background: COLORS.creamDark }} />;
}

function SectionHeader({ title, onSeeMore }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
      <h3 style={{ fontSize: 18, fontWeight: 800, color: COLORS.forest, margin: 0 }}>{title}</h3>
      {onSeeMore && (
        <button onClick={onSeeMore} style={{ background: "none", border: "none", color: COLORS.sage, fontSize: 13, fontWeight: 700, cursor: "pointer" }}>
          See more →
        </button>
      )}
    </div>
  );
}

function HScrollRow({ children }) {
  return (
    <div style={{ display: "flex", gap: 12, overflowX: "auto", paddingBottom: 4, scrollbarWidth: "none", msOverflowStyle: "none" }}>
      {children}
    </div>
  );
}

const ProductCardBrief = memo(function ProductCardBrief({ product, isBest, onClick, onLike, liked, onAddBag }) {
  return (
    <div
      style={{
        minWidth: 150, maxWidth: 150,
        background: COLORS.white,
        borderRadius: 14,
        padding: 12,
        cursor: "pointer",
        position: "relative",
        boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
        flexShrink: 0,
      }}
    >
      {isBest && (
        <div style={{ position: "absolute", top: -4, right: 8, background: COLORS.sage, color: COLORS.white, padding: "2px 8px", borderRadius: 6, fontSize: 9, fontWeight: 700, zIndex: 1 }}>Best</div>
      )}
      <div style={{ position: "absolute", top: 8, left: 8, zIndex: 1 }}>
        {onLike && (
          <button onClick={(e) => { e.stopPropagation(); onLike(product); }} style={{ background: "rgba(255,255,255,0.9)", border: "none", borderRadius: 20, width: 28, height: 28, cursor: "pointer", fontSize: 14, display: "flex", alignItems: "center", justifyContent: "center" }}>
            {liked ? "❤️" : "🤍"}
          </button>
        )}
      </div>
      <div onClick={onClick}>
        <ProductImage url={product.image_url} size={126} />
        <p style={{ fontSize: 10, color: COLORS.textLight, margin: "8px 0 2px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{product.brand}</p>
        <p style={{ fontSize: 12, fontWeight: 700, color: COLORS.text, margin: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", lineHeight: 1.2 }}>{product.title}</p>
        <div style={{ marginTop: 6, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 }}>
          {product.similarity ? <MatchBadge value={Math.round((product.similarity || 0) * 100)} /> : <span style={{ fontSize: 11, color: COLORS.textLight }}>⭐ {product.rating || "N/A"}</span>}
          <span style={{ fontWeight: 700, color: COLORS.forest, fontSize: 12 }}>{product.price}</span>
        </div>
      </div>
      {onAddBag && (
        <button onClick={(e) => { e.stopPropagation(); onAddBag(product); }} style={{ marginTop: 8, width: "100%", padding: "6px 0", borderRadius: 8, border: `1px solid ${COLORS.sage}`, background: "transparent", color: COLORS.sage, fontSize: 11, fontWeight: 700, cursor: "pointer" }}>
          + Add to Bag
        </button>
      )}
    </div>
  );
});

/* ---- Screens ---- */

function SplashScreen({ onNext }) {
  const [show, setShow] = useState(false);
  useEffect(() => { setShow(true); const t = setTimeout(onNext, 2500); return () => clearTimeout(t); }, []);
  return (
    <div style={{ height: "100%", background: `linear-gradient(180deg, ${COLORS.cream} 0%, ${COLORS.creamDark} 100%)`, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", opacity: show ? 1 : 0, transition: "opacity 0.8s" }}>
      <div style={{ width: 90, height: 90, borderRadius: 22, background: COLORS.sage, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 20, boxShadow: `0 8px 32px ${COLORS.shadowWarm}` }}>
        <span style={{ fontSize: 40, color: COLORS.white, fontWeight: 800, fontFamily: "serif" }}>R</span>
      </div>
      <h1 style={{ fontSize: 36, fontWeight: 800, color: COLORS.forest, fontFamily: "serif", margin: 0 }}>Ruvisa</h1>
      <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 8 }}>AI-Powered Skincare</p>
      <div style={{ marginTop: 40, display: "flex", gap: 4 }}>
        {[0, 1, 2].map(i => <div key={i} style={{ width: 6, height: 6, borderRadius: 3, background: COLORS.sage, opacity: 0.5, animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite` }} />)}
      </div>
    </div>
  );
}

function OnboardingScreen({ onNext }) {
  const [step, setStep] = useState(0);
  const slides = [
    { title: "Science-Backed\nSkincare Analysis", subtitle: "Powered by AI that analyzes real ingredients, clinical studies, and thousands of user reviews — completely unbiased.", icon: "🔬" },
    { title: "Smart Skin\nDetection", subtitle: "Our AI scans your face to detect skin concerns like acne, wrinkles, pores, and pigmentation — giving you a detailed skin health report.", icon: "📸" },
    { title: "Personalized\nRecommendations", subtitle: "Products matched to YOUR unique skin profile based on ingredient evidence and real user reviews from Sephora HK.", icon: "✨" },
    { title: "Your AI Beauty\nConsultant", subtitle: "Ask Ruvisa anything about skincare — why a product matches you, what ingredients help your concerns, or how to build a routine.", icon: "🤖" },
  ];
  return (
    <div style={{ height: "100%", background: COLORS.cream, display: "flex", flexDirection: "column" }}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "0 30px", textAlign: "center" }}>
        <div style={{ fontSize: 80, marginBottom: 30 }}>{slides[step].icon}</div>
        <h2 style={{ fontSize: 28, fontWeight: 800, color: COLORS.forest, whiteSpace: "pre-line", lineHeight: 1.2, margin: 0 }}>{slides[step].title}</h2>
        <p style={{ color: COLORS.textLight, fontSize: 15, lineHeight: 1.6, marginTop: 16, maxWidth: 300 }}>{slides[step].subtitle}</p>
      </div>
      <div style={{ display: "flex", justifyContent: "center", gap: 6, marginBottom: 20 }}>
        {slides.map((_, i) => <div key={i} style={{ width: i === step ? 24 : 8, height: 8, borderRadius: 4, background: i === step ? COLORS.sage : COLORS.creamDark, transition: "all 0.3s" }} />)}
      </div>
      <div style={{ padding: "0 24px 40px" }}>
        <Btn onClick={() => step < slides.length - 1 ? setStep(step + 1) : onNext()}>{step < slides.length - 1 ? "Next" : "Get Started"}</Btn>
        {step < slides.length - 1 && <button onClick={onNext} style={{ width: "100%", padding: "12px", border: "none", background: "transparent", color: COLORS.textLight, fontSize: 14, cursor: "pointer", marginTop: 4 }}>Skip</button>}
      </div>
    </div>
  );
}

function SkinTypeScreen({ onSelect }) {
  const [selected, setSelected] = useState(null);
  const [concerns, setConcerns] = useState([]);
  const [step, setStep] = useState(0);
  const toggleConcern = (id) => setConcerns(prev => prev.includes(id) ? prev.filter(c => c !== id) : [...prev, id]);

  if (step === 0) {
    return (
      <div style={{ height: "100%", background: COLORS.cream, display: "flex", flexDirection: "column" }}>
        <div style={{ padding: "60px 24px 20px" }}>
          <p style={{ color: COLORS.sage, fontSize: 13, fontWeight: 600, marginBottom: 4 }}>STEP 1 OF 2</p>
          <h2 style={{ fontSize: 26, fontWeight: 800, color: COLORS.forest, margin: 0 }}>What's your skin type?</h2>
          <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 6 }}>This helps us personalize your experience</p>
        </div>
        <div style={{ flex: 1, padding: "0 24px", display: "flex", flexDirection: "column", gap: 12 }}>
          {skinTypes.map(type => (
            <button key={type.id} onClick={() => setSelected(type.id)} style={{
              display: "flex", alignItems: "center", gap: 16, padding: "18px 20px", borderRadius: 16,
              border: selected === type.id ? `2px solid ${COLORS.sage}` : `2px solid ${COLORS.creamDark}`,
              background: selected === type.id ? COLORS.overlayMuted : COLORS.white, cursor: "pointer", textAlign: "left", transition: "all 0.2s"
            }}>
              <span style={{ fontSize: 32 }}>{type.emoji}</span>
              <div>
                <div style={{ fontWeight: 700, fontSize: 16, color: COLORS.text }}>{type.label}</div>
                <div style={{ fontSize: 13, color: COLORS.textLight, marginTop: 2 }}>{type.desc}</div>
              </div>
              {selected === type.id && <div style={{ marginLeft: "auto", width: 24, height: 24, borderRadius: 12, background: COLORS.sage, display: "flex", alignItems: "center", justifyContent: "center", color: "white", fontSize: 14 }}>✓</div>}
            </button>
          ))}
        </div>
        <div style={{ padding: "20px 24px 40px" }}>
          <Btn disabled={!selected} onClick={() => setStep(1)}>Continue</Btn>
        </div>
      </div>
    );
  }

  return (
    <div style={{ height: "100%", background: COLORS.cream, display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "60px 24px 20px" }}>
        <p style={{ color: COLORS.sage, fontSize: 13, fontWeight: 600, marginBottom: 4 }}>STEP 2 OF 2</p>
        <h2 style={{ fontSize: 26, fontWeight: 800, color: COLORS.forest, margin: 0 }}>What are your skin concerns?</h2>
        <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 6 }}>Select all that apply</p>
      </div>
      <div style={{ flex: 1, padding: "0 24px", display: "flex", flexWrap: "wrap", gap: 10, alignContent: "flex-start" }}>
        {skinConcerns.map(concern => (
          <button key={concern.id} onClick={() => toggleConcern(concern.id)} style={{
            padding: "12px 18px", borderRadius: 24,
            border: concerns.includes(concern.id) ? `2px solid ${COLORS.sage}` : `2px solid ${COLORS.creamDark}`,
            background: concerns.includes(concern.id) ? COLORS.overlayMuted : COLORS.white, cursor: "pointer",
            fontSize: 14, fontWeight: 600, color: concerns.includes(concern.id) ? COLORS.forest : COLORS.text, transition: "all 0.2s"
          }}>{concern.emoji} {concern.label}</button>
        ))}
      </div>
      <div style={{ padding: "20px 24px 40px" }}>
        <Btn disabled={concerns.length === 0} onClick={() => onSelect(selected, concerns)}>Analyze My Skin</Btn>
      </div>
    </div>
  );
}

function FaceScanScreen({ onComplete, skinType, userId }) {
  const [phase, setPhase] = useState("upload");
  const [imageFile, setImageFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [progress, setProgress] = useState(0);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const fileRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const handleFile = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setImageFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } } });
      streamRef.current = stream;
      setCameraActive(true);
    } catch (err) {
      setError("Camera access denied. Please select a photo instead.");
    }
  };

  useEffect(() => {
    if (cameraActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => {});
    }
  }, [cameraActive]);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  };

  const captureFromCamera = () => {
    if (!videoRef.current || !streamRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        setImageFile(file);
        setPreview(URL.createObjectURL(file));
        stopCamera();
      }
    }, "image/jpeg", 0.9);
  };

  useEffect(() => () => { if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop()); }, []);

  const startScan = async () => {
    if (!imageFile) return;
    setPhase("scanning");
    setError(null);
    const progressInterval = setInterval(() => {
      setProgress(p => Math.min(p + 1, 90));
    }, 200);
    try {
      const result = await api.analyzeImage(userId, skinType, imageFile);
      clearInterval(progressInterval);
      setProgress(100);
      setAnalysis(result);
      setTimeout(() => onComplete(result), 800);
    } catch (err) {
      clearInterval(progressInterval);
      setError(err.message);
      setPhase("upload");
    }
  };

  if (phase === "upload") {
    return (
      <div style={{ height: "100%", background: COLORS.cream, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 24 }}>
        <input ref={fileRef} type="file" accept="image/*" capture="user" onChange={handleFile} style={{ display: "none" }} />
        {cameraActive ? (
          <>
            <div style={{ width: 200, height: 260, borderRadius: "50% 50% 45% 45%", border: `3px solid ${COLORS.sage}`, overflow: "hidden", marginBottom: 12, background: COLORS.creamDark }}>
              <video ref={videoRef} autoPlay playsInline muted style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)", display: "block" }} />
            </div>
            <p style={{ fontSize: 12, color: COLORS.textLight, margin: "0 0 16px", textAlign: "center", maxWidth: 260 }}>Please make sure you're in good lighting for the best skin analysis results.</p>
            <div style={{ display: "flex", gap: 10, marginBottom: 24 }}>
              <button onClick={captureFromCamera} style={{ padding: "12px 24px", borderRadius: 20, border: "none", background: COLORS.sage, color: COLORS.white, fontSize: 14, fontWeight: 700, cursor: "pointer" }}>Capture</button>
              <button onClick={stopCamera} style={{ padding: "12px 24px", borderRadius: 20, border: `1px solid ${COLORS.sage}`, background: COLORS.white, color: COLORS.sage, fontSize: 14, fontWeight: 600, cursor: "pointer" }}>Cancel</button>
            </div>
          </>
        ) : preview ? (
          <div style={{ width: 200, height: 260, borderRadius: "50% 50% 45% 45%", border: `3px solid ${COLORS.sage}`, overflow: "hidden", marginBottom: 24 }}>
            <img src={preview} alt="preview" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </div>
        ) : (
          <div style={{ width: 200, height: 260, borderRadius: "50% 50% 45% 45%", border: `3px dashed ${COLORS.sage}`, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", marginBottom: 24 }}>
            <div style={{ width: 72, height: 72, borderRadius: 36, background: COLORS.navScanSolid, display: "flex", alignItems: "center", justifyContent: "center" }}>
              <ScanFace size={38} color={COLORS.white} strokeWidth={1.5} />
            </div>
            <p style={{ color: COLORS.sage, fontSize: 14, fontWeight: 600, marginTop: 12, textAlign: "center" }}>Take a photo or upload</p>
          </div>
        )}
        <h3 style={{ color: COLORS.forest, fontSize: 20, fontWeight: 700, margin: 0 }}>Scan Your Face</h3>
        <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 8, textAlign: "center", maxWidth: 280 }}>Our AI will analyze your skin and find the best products for you</p>
        <p style={{ color: COLORS.sage, fontSize: 12, marginTop: 4, textAlign: "center", maxWidth: 280, opacity: 0.9 }}>For best results, use good lighting when taking your photo.</p>
        {error && <p style={{ color: COLORS.red, fontSize: 13, marginTop: 8 }}>{error}</p>}
        <div style={{ width: "100%", maxWidth: 300, marginTop: 24, display: "flex", flexDirection: "column", gap: 10 }}>
          {preview && !cameraActive && <Btn onClick={startScan}>Start Analysis</Btn>}
          {!cameraActive && (
            <>
              <Btn onClick={startCamera} style={{ background: COLORS.sage, color: COLORS.white, border: "none" }}>
                📷 Take Photo
              </Btn>
              <Btn onClick={() => fileRef.current?.click()} style={{ background: preview ? COLORS.white : "transparent", color: preview ? COLORS.text : COLORS.sage, border: `1px solid ${COLORS.sage}` }}>
                {preview ? "Choose Different Photo" : "Select from Gallery"}
              </Btn>
            </>
          )}
        </div>
      </div>
    );
  }

  return (
    <div style={{ height: "100%", background: COLORS.cream, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 24 }}>
      <div style={{ width: 200, height: 260, borderRadius: "50% 50% 45% 45%", border: `3px solid ${COLORS.sage}`, overflow: "hidden", position: "relative", marginBottom: 30 }}>
        {preview && <img src={preview} alt="scanning" style={{ width: "100%", height: "100%", objectFit: "cover", opacity: 0.7 }} />}
        <div style={{ position: "absolute", bottom: 0, width: "100%", height: `${progress}%`, background: `linear-gradient(180deg, ${COLORS.overlayMuted} 0%, ${COLORS.overlayDeep} 100%)`, transition: "height 0.3s" }} />
        <div style={{ position: "absolute", top: "50%", left: 0, right: 0, height: 2, background: COLORS.sage, opacity: 0.6, transform: `translateY(${(progress - 50) * 2.6}px)`, transition: "transform 0.3s" }} />
      </div>
      <h3 style={{ color: COLORS.forest, fontSize: 20, fontWeight: 700, margin: 0 }}>Analyzing Your Skin...</h3>
      <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 8 }}>This may take a moment</p>
      <div style={{ width: 200, height: 4, background: COLORS.creamDark, borderRadius: 2, marginTop: 20, overflow: "hidden" }}>
        <div style={{ height: "100%", background: COLORS.sage, borderRadius: 2, width: `${progress}%`, transition: "width 0.3s" }} />
      </div>
      <p style={{ color: COLORS.sage, fontSize: 13, fontWeight: 600, marginTop: 8 }}>{progress}%</p>
    </div>
  );
}

/* ---- HOME SCREEN ---- */

function HomeScreen({
  onNavigate,
  userId,
  userName,
  skinType,
  analysis,
  onLogout,
  likedUrls,
  onLike,
  onAddBag,
  recs,
  trending,
  journey,
}) {
  const overall = analysis?.overall_score || 75;
  const concerns = analysis?.concerns || {};
  const activeConcerns = Object.entries(concerns).filter(([, v]) => v > 0.05);
  const recCategories = Object.keys(recs).filter(k => k !== "routine");
  const imp = journey?.improvement;
  const pastScans = journey?.analyses || [];
  const pastPurchases = journey?.purchases || [];
  const likedItems = journey ? [] : [];

  return (
    <div style={{ height: "100%", background: COLORS.cream, overflow: "auto", paddingBottom: 120 }}>
      {/* Header */}
      <div style={{ padding: "50px 24px 16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <p style={{ color: COLORS.textLight, fontSize: 14, margin: 0 }}>Welcome back</p>
            <h2 style={{ fontSize: 24, fontWeight: 800, color: COLORS.forest, margin: "4px 0 0" }}>Hi, {userName}</h2>
          </div>
          <div onClick={() => onNavigate("profile")} style={{ width: 44, height: 44, borderRadius: 22, background: COLORS.sageLight, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" }} title="Profile">
            <User size={22} color={COLORS.white} strokeWidth={2} />
          </div>
        </div>
      </div>

      {/* Skin Score card */}
      {analysis && (
        <div style={{ margin: "0 24px 16px", padding: 20, borderRadius: 20, background: `linear-gradient(135deg, ${COLORS.sage}, ${COLORS.sageDark})`, color: COLORS.white, boxShadow: `0 10px 28px ${COLORS.shadowWarm}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <p style={{ fontSize: 13, opacity: 0.8, margin: 0 }}>Your Skin Score</p>
              <p style={{ fontSize: 40, fontWeight: 800, margin: "4px 0" }}>{overall}<span style={{ fontSize: 18, opacity: 0.7 }}>/100</span></p>
              <p style={{ fontSize: 12, opacity: 0.8, margin: 0 }}>{skinType} skin • {activeConcerns.length} active concerns</p>
            </div>
            <div onClick={() => onNavigate("scan")} style={{ background: COLORS.navScanSolid, borderRadius: 14, padding: "10px 14px", cursor: "pointer", display: "flex", flexDirection: "column", alignItems: "center" }}>
              <ScanFace size={28} color={COLORS.white} strokeWidth={1.5} />
              <p style={{ fontSize: 10, margin: "4px 0 0", textAlign: "center" }}>Re-scan</p>
            </div>
          </div>
        </div>
      )}

      {!analysis && <FaceScanCTA onStartScan={() => onNavigate("scan")} />}

      {/* Skin improvement banner */}
      {imp && (
        <div style={{ margin: "0 24px 16px", padding: 16, borderRadius: 16, background: imp.improvement_pct >= 0 ? COLORS.overlaySoft : "#FFF3E0" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <span style={{ fontSize: 28 }}>{imp.improvement_pct >= 0 ? "📈" : "📉"}</span>
            <div>
              <p style={{ fontSize: 14, fontWeight: 700, color: COLORS.forest, margin: 0 }}>
                {imp.improvement_pct > 0 ? `Skin improved ${imp.improvement_pct}%` : imp.improvement_pct < 0 ? `Skin needs attention` : "No change"}
              </p>
              <p style={{ fontSize: 12, color: COLORS.textLight, margin: "2px 0 0" }}>Score: {imp.previous_score} → {imp.latest_score}</p>
            </div>
          </div>
        </div>
      )}

      {/* Concerns */}
      {activeConcerns.length > 0 && (
        <div style={{ padding: "0 24px", marginBottom: 20 }}>
          <SectionHeader title="Your Concerns" />
          <HScrollRow>
            {activeConcerns.sort((a, b) => b[1] - a[1]).map(([key, val]) => (
              <div key={key} style={{ minWidth: 100, background: COLORS.white, borderRadius: 14, padding: 12, textAlign: "center", flexShrink: 0 }}>
                <CircularProgress value={Math.round(val * 100)} size={54} strokeWidth={5} color={val > 0.5 ? COLORS.red : COLORS.gold} />
                <p style={{ fontSize: 11, fontWeight: 700, color: COLORS.text, margin: "6px 0 0" }}>{CONCERN_LABELS[key] || key}</p>
              </div>
            ))}
          </HScrollRow>
        </div>
      )}

      {/* Detection images */}
      {(analysis?.images?.acne || analysis?.images?.wrinkle) && (
        <div style={{ padding: "0 24px", marginBottom: 16 }}>
          <SectionHeader title="Scan Results" />
          <div style={{ display: "flex", gap: 10, overflowX: "auto" }}>
            {analysis.images.acne && (
              <div style={{ minWidth: "48%", background: COLORS.white, borderRadius: 14, overflow: "hidden" }}>
                <img src={analysis.images.acne} alt="Acne" style={{ width: "100%", height: "auto", display: "block" }} />
                <p style={{ fontSize: 11, fontWeight: 600, textAlign: "center", padding: 6, margin: 0, color: COLORS.textLight }}>Acne Detection</p>
              </div>
            )}
            {analysis.images.wrinkle && (
              <div style={{ minWidth: "48%", background: COLORS.white, borderRadius: 14, overflow: "hidden" }}>
                <img src={analysis.images.wrinkle} alt="Wrinkle" style={{ width: "100%", height: "auto", display: "block" }} />
                <p style={{ fontSize: 11, fontWeight: 600, textAlign: "center", padding: 6, margin: 0, color: COLORS.textLight }}>Wrinkle Analysis</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Per-lesion breakdown */}
      {analysis?.acne_summary?.detections?.length > 0 && (() => {
        const dets = analysis.acne_summary.detections;
        const classCounts = {};
        dets.forEach(d => { classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1; });
        return (
          <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: 0 }}>Acne ({analysis.acne_summary.total_detections} lesions)</h4>
              <div style={{ display: "flex", gap: 6 }}>
                {Object.entries(classCounts).map(([cls, cnt]) => (
                  <span key={cls} style={{ fontSize: 10, background: COLORS.cream, padding: "3px 8px", borderRadius: 8, fontWeight: 600, color: COLORS.text }}>{cls}: {cnt}</span>
                ))}
              </div>
            </div>
            <div style={{ maxHeight: 160, overflowY: "auto" }}>
              {dets.map((d, i) => (
                <div key={i} style={{ padding: "6px 0", borderBottom: i < dets.length - 1 ? `1px solid ${COLORS.creamDark}` : "none", display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: COLORS.text, flex: 1 }}>{d.class_name}</span>
                  <span style={{ fontSize: 10, background: COLORS.creamDark, padding: "2px 6px", borderRadius: 6, color: COLORS.textLight }}>{d.severity_name}</span>
                  <span style={{ fontSize: 10, color: COLORS.textLight, width: 36, textAlign: "right" }}>{(d.confidence * 100).toFixed(0)}%</span>
                  <span style={{ fontSize: 11, color: COLORS.sage, fontWeight: 600, width: 70, textAlign: "right" }}>{d.face_region?.replace(/_/g, " ")}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })()}

      {/* Acne by region */}
      {analysis?.acne_summary?.regions && Object.keys(analysis.acne_summary.regions).length > 0 && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: "0 0 14px" }}>Acne by Region</h4>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            {Object.entries(analysis.acne_summary.regions).map(([name, r]) => {
              const roiPct = Math.min(100, Math.round((r.roi ?? 0) * 100));
              const color = roiPct > 50 ? COLORS.red : roiPct > 20 ? COLORS.gold : COLORS.sage;
              return (
                <div key={name} style={{ background: COLORS.cream, borderRadius: 14, padding: 14, display: "flex", alignItems: "center", gap: 12 }}>
                  <CircularProgress value={roiPct} size={56} strokeWidth={5} color={color} centerLabel={`${roiPct}%`} />
                  <div>
                    <p style={{ fontSize: 13, fontWeight: 700, color: COLORS.forest, margin: 0, textTransform: "capitalize" }}>{name.replace(/_/g, " ")}</p>
                    <p style={{ fontSize: 11, color: COLORS.textLight, margin: "2px 0 0" }}>{r.count} lesion{r.count !== 1 ? "s" : ""}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Wrinkle by region */}
      {(() => {
        const wr = analysis?.wrinkle_summary?.wrinkle_regions || analysis?.wrinkle_regions;
        const ws = analysis?.wrinkle_summary || {};
        if (!wr || Object.keys(wr).length === 0) return null;
        return (
          <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
            <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: "0 0 6px" }}>Wrinkle by Region</h4>
            <p style={{ fontSize: 12, color: COLORS.textLight, margin: "0 0 14px" }}>
              Overall: {(ws.wrinkle_pct ?? 0).toFixed(2)}% · {ws.severity || "none"} ({ws.severity_score ?? 0}/3)
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
              {Object.entries(wr).map(([name, r]) => {
                const pct = r.wrinkle_pct ?? 0;
                const sevScore = r.severity_score ?? 0;
                const sevLabel = r.severity || "none";
                const chartValue = Math.min(100, pct * 20);
                const color = sevScore >= 2 ? COLORS.red : sevScore >= 1 ? COLORS.gold : COLORS.sageLight;
                return (
                  <div key={name} style={{ background: COLORS.cream, borderRadius: 14, padding: 14, display: "flex", alignItems: "center", gap: 12 }}>
                    <CircularProgress value={chartValue} size={56} strokeWidth={5} color={color} centerLabel={`${sevScore}/3`} showPercent={false} />
                    <div>
                      <p style={{ fontSize: 13, fontWeight: 700, color: COLORS.forest, margin: 0, textTransform: "capitalize" }}>{name.replace(/_/g, " ")}</p>
                      <p style={{ fontSize: 11, color: COLORS.textLight, margin: "2px 0 0" }}>{pct.toFixed(2)}% · {sevLabel}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })()}

      {/* Recommended Products */}
      {recCategories.length > 0 && (
        <div style={{ padding: "0 24px", marginBottom: 20 }}>
          <SectionHeader title="Recommended For You" onSeeMore={() => onNavigate("recommendations")} />
          {recCategories.map(cat => {
            const items = recs[cat] || [];
            if (!items.length) return null;
            return (
              <div key={cat} style={{ marginBottom: 16 }}>
                <p style={{ fontSize: 14, fontWeight: 700, color: COLORS.text, margin: "0 0 8px" }}>{cat}</p>
                <HScrollRow>
                  {items.slice(0, 4).map((p, i) => (
                    <ProductCardBrief
                      key={p.product_url}
                      product={p}
                      isBest={i === 0}
                      onClick={() => onNavigate("product", p)}
                      onLike={onLike}
                      liked={likedUrls.has(p.product_url)}
                      onAddBag={onAddBag}
                    />
                  ))}
                </HScrollRow>
              </div>
            );
          })}
        </div>
      )}

      {/* Skincare Routine - optimized for compatibility and concern coverage */}
      {analysis && (
        <div style={{ padding: "0 24px", marginBottom: 20 }}>
          <SectionHeader title="Skincare Routine" />
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {(() => {
              const routineSteps = ["Cleanser", "Toner", "Serum", "Moisturizer", "Sunscreen"];
              const routineMap = (recs.routine || []).reduce((acc, r) => {
                if (r?.step && r?.product) acc[r.step] = r.product;
                return acc;
              }, {});
              return routineSteps.map((step, i) => {
                const matched = routineMap[step] || (recs[step] || [])[0];
                return (
                  <div key={step} onClick={() => matched && onNavigate("product", matched)} style={{ background: COLORS.white, borderRadius: 14, padding: 14, display: "flex", alignItems: "center", gap: 12, cursor: matched ? "pointer" : "default" }}>
                    <div style={{ width: 32, height: 32, borderRadius: 16, background: COLORS.sageLight, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 700, color: COLORS.forest, flexShrink: 0 }}>{i + 1}</div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <p style={{ fontSize: 14, fontWeight: 700, color: COLORS.text, margin: 0 }}>{step}</p>
                      {matched ? (
                        <p style={{ fontSize: 12, color: COLORS.textLight, margin: "2px 0 0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{matched.brand} · {matched.title}</p>
                      ) : (
                        <p style={{ fontSize: 12, color: COLORS.textLight, margin: "2px 0 0" }}>Browse {step.toLowerCase()}s in Shop</p>
                      )}
                    </div>
                    <span style={{ color: COLORS.textLight }}>›</span>
                  </div>
                );
              });
            })()}
          </div>
        </div>
      )}

      {/* Trending Products */}
      {trending.length > 0 && (
        <div style={{ padding: "0 24px", marginBottom: 20 }}>
          <SectionHeader title="Trending Products" onSeeMore={() => onNavigate("shop")} />
          <HScrollRow>
            {trending.map(p => (
              <ProductCardBrief
                key={p.product_url}
                product={p}
                onClick={() => onNavigate("product", p)}
                onLike={onLike}
                liked={likedUrls.has(p.product_url)}
                onAddBag={onAddBag}
              />
            ))}
          </HScrollRow>
        </div>
      )}

      {/* Liked Products */}
      {(() => {
        const liked = Array.from(likedUrls);
        if (!liked.length) return null;
        return null; // liked shown from state in separate section via journey or bag
      })()}

      {/* Skin Journey summary */}
      {pastScans.length > 0 && (
        <div style={{ padding: "0 24px", marginBottom: 20 }}>
          <SectionHeader title="Skin Journey" />
          <HScrollRow>
            {pastScans.slice(0, 5).map((a, i) => {
              const cv = Array.isArray(a.concern_vector) ? a.concern_vector : [];
              const concernCount = cv.filter(v => v > 0.1).length;
              return (
                <div key={a.id} onClick={() => onNavigate("scanDetail", a)} style={{ minWidth: 140, background: COLORS.white, borderRadius: 14, padding: 14, flexShrink: 0, cursor: "pointer" }}>
                  <p style={{ fontSize: 12, fontWeight: 700, color: COLORS.forest, margin: 0 }}>Scan {pastScans.length - i}</p>
                  <p style={{ fontSize: 11, color: COLORS.textLight, margin: "4px 0" }}>{new Date(a.created_at).toLocaleDateString()}</p>
                  <p style={{ fontSize: 11, color: COLORS.text, margin: 0 }}>{concernCount} concerns</p>
                  <p style={{ fontSize: 10, color: COLORS.sage, fontWeight: 600, margin: "4px 0 0" }}>Tap to view →</p>
                </div>
              );
            })}
          </HScrollRow>
        </div>
      )}

      {/* Past Purchases */}
      {pastPurchases.length > 0 && (
        <div style={{ padding: "0 24px", marginBottom: 20 }}>
          <SectionHeader title="Past Purchases" />
          {pastPurchases.slice(0, 3).map(p => (
            <div key={p.id} style={{ background: COLORS.white, borderRadius: 14, padding: 14, marginBottom: 8, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ fontSize: 14, fontWeight: 600, color: COLORS.text, margin: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p.product_title || "Product"}</p>
                <p style={{ fontSize: 11, color: COLORS.textLight, margin: "2px 0 0" }}>{new Date(p.purchased_at).toLocaleDateString()}</p>
              </div>
              {p.price > 0 && <span style={{ fontSize: 13, color: COLORS.forest, fontWeight: 700 }}>${p.price}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ---- SCAN DETAIL SCREEN ---- */

function ScanDetailScreen({ scanData, onBack }) {
  if (!scanData) return null;
  const cv = Array.isArray(scanData.concern_vector) ? scanData.concern_vector : [];
  const concernNames = ["acne", "comedonal_acne", "pigmentation", "acne_scars_texture", "pores", "redness", "wrinkles"];
  const acne = scanData.acne_summary || {};
  const wrinkle = scanData.wrinkle_summary || {};
  const wr = wrinkle.wrinkle_regions || {};
  const report = scanData.full_report || {};
  const overall = cv.length === 7 ? Math.max(20, Math.min(98, Math.round(100 - (cv.reduce((a, b) => a + b, 0) / 7) * 80))) : 75;

  const toUploadUrl = (p) => {
    if (!p) return null;
    if (p.includes("uploads/")) return `/api/uploads/${p.split("uploads/").pop()}`;
    return `/api/uploads/${p.split("/").pop()}`;
  };

  const originalImg = toUploadUrl(scanData.image_path);
  const acneVis = toUploadUrl(report.acne?.visualization_path);
  const wrinkleVis = toUploadUrl(report.wrinkle?.visualization_path);

  return (
    <div style={{ height: "100%", background: COLORS.cream, overflow: "auto", paddingBottom: 120 }}>
      <BackHeader onBack={onBack} title={`Scan · ${new Date(scanData.created_at).toLocaleDateString()}`} />

      {/* Overall score */}
      <div style={{ margin: "0 24px 16px", padding: 20, borderRadius: 20, background: `linear-gradient(135deg, ${COLORS.sage}, ${COLORS.sageDark})`, color: COLORS.white, textAlign: "center" }}>
        <CircularProgress value={overall} size={80} strokeWidth={6} color="#fff" />
        <p style={{ fontSize: 16, fontWeight: 700, margin: "10px 0 0" }}>Skin Health Score: {overall}/100</p>
      </div>

      {/* Original scanned face */}
      {originalImg && (
        <div style={{ padding: "0 24px", marginBottom: 16 }}>
          <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: "0 0 10px" }}>Your Photo</h4>
          <div style={{ background: COLORS.white, borderRadius: 14, overflow: "hidden", textAlign: "center" }}>
            <img src={originalImg} alt="Your face" style={{ width: "100%", maxHeight: 300, objectFit: "contain", display: "block" }} onError={e => { e.target.parentElement.style.display = "none"; }} />
          </div>
        </div>
      )}

      {/* Detection result images */}
      {(acneVis || wrinkleVis) && (
        <div style={{ padding: "0 24px", marginBottom: 16 }}>
          <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: "0 0 10px" }}>Detection Results</h4>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {acneVis && (
              <div style={{ background: COLORS.white, borderRadius: 14, overflow: "hidden" }}>
                <img src={acneVis} alt="Acne Detection" style={{ width: "100%", height: "auto", display: "block" }} onError={e => { e.target.parentElement.style.display = "none"; }} />
                <p style={{ fontSize: 12, fontWeight: 600, textAlign: "center", padding: 8, margin: 0, color: COLORS.textLight }}>Acne & Lesion Detection</p>
              </div>
            )}
            {wrinkleVis && (
              <div style={{ background: COLORS.white, borderRadius: 14, overflow: "hidden" }}>
                <img src={wrinkleVis} alt="Wrinkle Detection" style={{ width: "100%", height: "auto", display: "block" }} onError={e => { e.target.parentElement.style.display = "none"; }} />
                <p style={{ fontSize: 12, fontWeight: 600, textAlign: "center", padding: 8, margin: 0, color: COLORS.textLight }}>Wrinkle Detection</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Concern vector */}
      {cv.length === 7 && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: "0 0 12px" }}>Detected Concerns</h4>
          {concernNames.map((c, i) => {
            const val = cv[i] || 0;
            if (val < 0.02) return null;
            return (
              <div key={c} style={{ marginBottom: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: COLORS.text }}>{CONCERN_LABELS[c] || c}</span>
                  <span style={{ fontSize: 13, fontWeight: 700, color: val > 0.5 ? COLORS.red : val > 0.2 ? COLORS.gold : COLORS.sage }}>{(val * 100).toFixed(0)}%</span>
                </div>
                <div style={{ width: "100%", height: 6, background: COLORS.creamDark, borderRadius: 3, overflow: "hidden" }}>
                  <div style={{ height: "100%", borderRadius: 3, width: `${Math.min(val * 100, 100)}%`, background: val > 0.5 ? COLORS.red : val > 0.2 ? COLORS.gold : COLORS.sage, transition: "width 0.5s" }} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Per-lesion breakdown */}
      {acne.detections?.length > 0 && (() => {
        const dets = acne.detections;
        const classCounts = {};
        dets.forEach(d => { classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1; });
        return (
          <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: 0 }}>Acne Lesions ({acne.total_detections || dets.length})</h4>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {Object.entries(classCounts).map(([cls, cnt]) => (
                  <span key={cls} style={{ fontSize: 10, background: COLORS.cream, padding: "3px 8px", borderRadius: 8, fontWeight: 600, color: COLORS.text }}>{cls}: {cnt}</span>
                ))}
              </div>
            </div>
            <div style={{ maxHeight: 220, overflowY: "auto" }}>
              {dets.map((d, i) => (
                <div key={i} style={{ padding: "6px 0", borderBottom: i < dets.length - 1 ? `1px solid ${COLORS.creamDark}` : "none", display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 12, fontWeight: 600, color: COLORS.text, flex: 1 }}>{d.class_name}</span>
                  <span style={{ fontSize: 10, background: COLORS.creamDark, padding: "2px 6px", borderRadius: 6, color: COLORS.textLight }}>{d.severity_name}</span>
                  <span style={{ fontSize: 10, color: COLORS.textLight, width: 36, textAlign: "right" }}>{(d.confidence * 100).toFixed(0)}%</span>
                  <span style={{ fontSize: 11, color: COLORS.sage, fontWeight: 600, width: 70, textAlign: "right" }}>{d.face_region?.replace(/_/g, " ")}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })()}

      {/* Acne by region */}
      {acne.regions && Object.keys(acne.regions).length > 0 && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: "0 0 14px" }}>Acne by Region</h4>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            {Object.entries(acne.regions).map(([name, r]) => {
              const roiPct = Math.min(100, Math.round((r.roi ?? 0) * 100));
              const color = roiPct > 50 ? COLORS.red : roiPct > 20 ? COLORS.gold : COLORS.sage;
              return (
                <div key={name} style={{ background: COLORS.cream, borderRadius: 14, padding: 14, display: "flex", alignItems: "center", gap: 12 }}>
                  <CircularProgress value={roiPct} size={56} strokeWidth={5} color={color} centerLabel={`${roiPct}%`} />
                  <div>
                    <p style={{ fontSize: 13, fontWeight: 700, color: COLORS.forest, margin: 0, textTransform: "capitalize" }}>{name.replace(/_/g, " ")}</p>
                    <p style={{ fontSize: 11, color: COLORS.textLight, margin: "2px 0 0" }}>{r.count} lesion{r.count !== 1 ? "s" : ""}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Severity distribution */}
      {acne.severity_distribution && Object.keys(acne.severity_distribution).length > 0 && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: "0 0 10px" }}>Severity Distribution</h4>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {Object.entries(acne.severity_distribution).map(([sev, cnt]) => (
              <div key={sev} style={{ background: COLORS.cream, borderRadius: 10, padding: "8px 14px", textAlign: "center" }}>
                <p style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: 0 }}>{cnt}</p>
                <p style={{ fontSize: 11, color: COLORS.textLight, margin: "2px 0 0", textTransform: "capitalize" }}>{sev}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Wrinkle overall */}
      {(wrinkle.severity || wrinkle.wrinkle_pct > 0) && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: "0 0 10px" }}>Wrinkle Overview</h4>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <CircularProgress value={Math.min(100, (wrinkle.wrinkle_pct ?? 0) * 20)} size={64} strokeWidth={5} color={wrinkle.severity_score >= 2 ? COLORS.red : wrinkle.severity_score >= 1 ? COLORS.gold : COLORS.sageLight} centerLabel={`${wrinkle.severity_score ?? 0}/3`} showPercent={false} />
            <div>
              <p style={{ fontSize: 14, fontWeight: 700, color: COLORS.text, margin: 0 }}>Coverage: {(wrinkle.wrinkle_pct ?? 0).toFixed(2)}%</p>
              <p style={{ fontSize: 13, color: COLORS.textLight, margin: "2px 0 0", textTransform: "capitalize" }}>Severity: {wrinkle.severity || "none"}</p>
            </div>
          </div>
        </div>
      )}

      {/* Wrinkle by region */}
      {Object.keys(wr).length > 0 && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: "0 0 14px" }}>Wrinkle by Region</h4>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
            {Object.entries(wr).map(([name, r]) => {
              const pct = r.wrinkle_pct ?? 0;
              const sevScore = r.severity_score ?? 0;
              const sevLabel = r.severity || "none";
              const chartValue = Math.min(100, pct * 20);
              const color = sevScore >= 2 ? COLORS.red : sevScore >= 1 ? COLORS.gold : COLORS.sageLight;
              return (
                <div key={name} style={{ background: COLORS.cream, borderRadius: 14, padding: 14, display: "flex", alignItems: "center", gap: 12 }}>
                  <CircularProgress value={chartValue} size={56} strokeWidth={5} color={color} centerLabel={`${sevScore}/3`} showPercent={false} />
                  <div>
                    <p style={{ fontSize: 13, fontWeight: 700, color: COLORS.forest, margin: 0, textTransform: "capitalize" }}>{name.replace(/_/g, " ")}</p>
                    <p style={{ fontSize: 11, color: COLORS.textLight, margin: "2px 0 0" }}>{pct.toFixed(2)}% · {sevLabel}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

/* ---- SHOP SCREEN ---- */

function ShopScreen({ onNavigate, onLike, likedUrls, onAddBag }) {
  const [query, setQuery] = useState("");
  const [products, setProducts] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedCat, setSelectedCat] = useState(null);
  const [loading, setLoading] = useState(true);

  const [totalProducts, setTotalProducts] = useState(0);
  const [loadingMore, setLoadingMore] = useState(false);
  const [visibleCount, setVisibleCount] = useState(48);
  const filtered = useMemo(() => products.filter(p => {
    if (selectedCat && ((p.category || "").toLowerCase() !== selectedCat.toLowerCase())) return false;
    if (query.length >= 2) {
      const q = query.toLowerCase();
      return p.title?.toLowerCase().includes(q) || p.brand?.toLowerCase().includes(q);
    }
    return true;
  }), [products, selectedCat, query]);
  useEffect(() => { setVisibleCount(48); }, [selectedCat, query]);

  const loadMore = useCallback(() => {
    if (loadingMore || products.length >= totalProducts) return;
    setLoadingMore(true);
    api.listProducts(selectedCat, 100, products.length)
      .then(data => {
        setProducts(prev => [...prev, ...(data.products || [])]);
      })
      .catch(() => {})
      .finally(() => setLoadingMore(false));
  }, [products.length, totalProducts, selectedCat, loadingMore]);

  useEffect(() => {
    setLoading(true);
    api.listProducts(selectedCat, 100, 0)
      .then(data => {
        setProducts(data.products || []);
        setCategories(data.categories || []);
        setTotalProducts(data.total || 0);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [selectedCat]);

  return (
    <div style={{ height: "100%", background: COLORS.cream, overflow: "auto", paddingBottom: 120 }}>
      <div style={{ padding: "50px 24px 12px" }}>
        <h2 style={{ fontSize: 24, fontWeight: 800, color: COLORS.forest, margin: "0 0 14px" }}>Shop</h2>
        <div style={{ background: COLORS.white, borderRadius: 14, padding: "12px 16px", display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ color: COLORS.textLight }}>🔍</span>
          <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search products..."
            style={{ border: "none", outline: "none", flex: 1, fontSize: 14, background: "transparent", color: COLORS.text }} />
        </div>
      </div>

      {/* Category pills */}
      <div style={{ padding: "0 24px 12px" }}>
        <HScrollRow>
          <button onClick={() => setSelectedCat(null)} style={{
            padding: "8px 16px", borderRadius: 20, border: "none", flexShrink: 0,
            background: !selectedCat ? COLORS.sage : COLORS.white,
            color: !selectedCat ? COLORS.white : COLORS.text,
            fontSize: 13, fontWeight: 600, cursor: "pointer",
          }}>All</button>
          {categories.map(cat => (
            <button key={cat} onClick={() => setSelectedCat(cat === selectedCat ? null : cat)} style={{
              padding: "8px 16px", borderRadius: 20, border: "none", flexShrink: 0,
              background: selectedCat === cat ? COLORS.sage : COLORS.white,
              color: selectedCat === cat ? COLORS.white : COLORS.text,
              fontSize: 13, fontWeight: 600, cursor: "pointer", whiteSpace: "nowrap",
            }}>{cat}</button>
          ))}
        </HScrollRow>
      </div>

      {loading ? (
        <p style={{ textAlign: "center", color: COLORS.textLight, marginTop: 30 }}>Loading products...</p>
      ) : (
        <div style={{ padding: "0 24px" }}>
          <p style={{ fontSize: 13, color: COLORS.textLight, margin: "0 0 12px" }}>
            {totalProducts > 0 ? `${filtered.length} of ${totalProducts} products` : `${filtered.length} products`}
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, justifyItems: "center" }}>
            {filtered.slice(0, visibleCount).map(p => (
              <ProductCardBrief
                key={p.product_url}
                product={p}
                onClick={() => onNavigate("product", p)}
                onLike={onLike}
                liked={likedUrls.has(p.product_url)}
                onAddBag={onAddBag}
              />
            ))}
          </div>
          {((products.length < totalProducts && totalProducts > 0) || filtered.length > visibleCount) && (
            <div style={{ padding: "24px 0", textAlign: "center" }}>
              <button
                onClick={() => {
                  if (products.length < totalProducts) loadMore();
                  else setVisibleCount(c => Math.min(c + 48, filtered.length));
                }}
                disabled={loadingMore}
                style={{
                  padding: "12px 32px", borderRadius: 14, border: "none",
                  background: COLORS.sage, color: COLORS.white,
                  fontSize: 14, fontWeight: 700, cursor: loadingMore ? "wait" : "pointer",
                }}
              >
                {loadingMore ? "Loading..." : products.length < totalProducts
                  ? `Load more (${totalProducts - products.length} left)`
                  : `Show more (${filtered.length - visibleCount} left)`}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ---- BAG SCREEN ---- */

function BagScreen({ userId, onNavigate }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadBag = useCallback(() => {
    if (userId === "guest") { setLoading(false); return; }
    api.getBag(userId).then(d => setItems(d.items || [])).catch(() => {}).finally(() => setLoading(false));
  }, [userId]);

  useEffect(() => { loadBag(); }, [loadBag]);

  const handleRemove = async (productUrl) => {
    try {
      await api.removeFromBag(userId, productUrl);
      setItems(prev => prev.filter(i => i.product_url !== productUrl));
    } catch {}
  };

  const handlePurchase = async (item) => {
    try {
      await api.recordPurchase(userId, { product_url: item.product_url, title: item.product_title, price: item.price });
      await handleRemove(item.product_url);
    } catch {}
  };

  const total = items.reduce((s, i) => s + (i.price || 0), 0);

  return (
    <div style={{ height: "100%", background: COLORS.cream, overflow: "auto", paddingBottom: 160 }}>
      <div style={{ padding: "50px 24px 16px" }}>
        <h2 style={{ fontSize: 24, fontWeight: 800, color: COLORS.forest, margin: 0 }}>My Bag</h2>
        <p style={{ fontSize: 13, color: COLORS.textLight, margin: "4px 0 0" }}>{items.length} item{items.length !== 1 ? "s" : ""}</p>
      </div>

      {loading ? (
        <p style={{ textAlign: "center", color: COLORS.textLight, marginTop: 30 }}>Loading...</p>
      ) : items.length === 0 ? (
        <div style={{ textAlign: "center", padding: "60px 24px" }}>
          <span style={{ fontSize: 60 }}>🛍️</span>
          <h3 style={{ color: COLORS.forest, margin: "16px 0 8px" }}>Your bag is empty</h3>
          <p style={{ color: COLORS.textLight, fontSize: 14 }}>Browse products and add them to your bag</p>
          <div style={{ maxWidth: 200, margin: "20px auto" }}>
            <Btn onClick={() => onNavigate("shop")}>Browse Shop</Btn>
          </div>
        </div>
      ) : (
        <div style={{ padding: "0 24px" }}>
          {items.map(item => (
            <div key={item.product_url} style={{ background: COLORS.white, borderRadius: 16, padding: 14, marginBottom: 12, display: "flex", gap: 12, alignItems: "center" }}>
              <ProductImage url={item.image_url} size={64} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ fontSize: 11, color: COLORS.textLight, margin: 0 }}>{item.brand}</p>
                <p style={{ fontSize: 14, fontWeight: 700, color: COLORS.text, margin: "2px 0 4px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.product_title}</p>
                <span style={{ fontWeight: 800, color: COLORS.forest, fontSize: 14 }}>{item.price ? `$${item.price.toFixed(2)}` : ""}</span>
              </div>
              <button onClick={() => handleRemove(item.product_url)} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 18, color: COLORS.textLight, padding: 4 }}>✕</button>
            </div>
          ))}
        </div>
      )}

      {items.length > 0 && (
        <div style={{ position: "fixed", bottom: 70, left: 0, right: 0, padding: "12px 24px", background: COLORS.cream }}>
          <div style={{ maxWidth: 430, margin: "0 auto" }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10, padding: "0 4px" }}>
              <span style={{ fontSize: 16, fontWeight: 700, color: COLORS.text }}>Total</span>
              <span style={{ fontSize: 18, fontWeight: 800, color: COLORS.forest }}>${total.toFixed(2)}</span>
            </div>
            <Btn onClick={() => {
              items.forEach(handlePurchase);
            }} style={{ background: COLORS.gold, color: COLORS.text }}>
              Mark All as Purchased
            </Btn>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---- RECOMMENDATIONS SCREEN (Full) ---- */

function RecommendationsScreen({ onBack, onNavigate, onAskRuvisa, userId, skinType, concernVector, analysis, onLike, likedUrls, onAddBag }) {
  const [recs, setRecs] = useState({});
  const [routine, setRoutine] = useState([]);
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.getRecommendations(userId, skinType, concernVector, null, 5)
      .then(data => {
        setRecs(data.recommendations || {});
        setRoutine(data.routine || []);
        setCategories(Object.keys(data.recommendations || {}));
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [userId, skinType, concernVector]);

  return (
    <div style={{ height: "100%", background: COLORS.cream, overflow: "auto", paddingBottom: 120 }}>
      <BackHeader onBack={onBack} title="Recommended For You" />

      {loading ? (
        <p style={{ textAlign: "center", color: COLORS.textLight, marginTop: 40 }}>Finding the best products for you...</p>
      ) : (
        <>
          {categories.map(cat => {
            const items = recs[cat] || [];
            if (items.length === 0) return null;
            return (
              <div key={cat} style={{ marginBottom: 24 }}>
                <div style={{ padding: "12px 24px 8px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <h4 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: 0 }}>{cat}</h4>
                  <button
                    onClick={() => onNavigate("category", { category: cat, products: items })}
                    style={{ background: "none", border: "none", color: COLORS.sage, fontSize: 13, fontWeight: 700, cursor: "pointer" }}
                  >
                    See more →
                  </button>
                </div>
                <div style={{ padding: "0 24px" }}>
                  <HScrollRow>
                    {items.map((p, i) => (
                      <ProductCardBrief
                        key={p.product_url}
                        product={p}
                        isBest={i === 0}
                        onClick={() => onNavigate("product", p)}
                        onLike={onLike}
                        liked={likedUrls.has(p.product_url)}
                        onAddBag={onAddBag}
                      />
                    ))}
                  </HScrollRow>
                </div>
              </div>
            );
          })}
          {/* Skincare Routine - at end, conflict-free */}
          {routine.length > 0 && (
            <div style={{ padding: "0 24px", marginBottom: 24 }}>
              <SectionHeader title="Your Skincare Routine" />
              <p style={{ fontSize: 12, color: COLORS.textLight, margin: "0 24px 12px" }}>Optimized for compatibility — no ingredient conflicts</p>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {["Cleanser", "Toner", "Serum", "Moisturizer", "Sunscreen"].map((step, i) => {
                  const r = routine.find(x => x.step === step);
                  const matched = r?.product;
                  return (
                    <div key={step} onClick={() => matched && onNavigate("product", matched)} style={{ background: COLORS.white, borderRadius: 14, padding: 14, display: "flex", alignItems: "center", gap: 12, cursor: matched ? "pointer" : "default" }}>
                      <div style={{ width: 32, height: 32, borderRadius: 16, background: COLORS.sageLight, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 700, color: COLORS.forest, flexShrink: 0 }}>{i + 1}</div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <p style={{ fontSize: 14, fontWeight: 700, color: COLORS.text, margin: 0 }}>{step}</p>
                        {matched ? (
                          <p style={{ fontSize: 12, color: COLORS.textLight, margin: "2px 0 0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{matched.brand} · {matched.title}</p>
                        ) : (
                          <p style={{ fontSize: 12, color: COLORS.textLight, margin: "2px 0 0" }}>Browse {step.toLowerCase()}s</p>
                        )}
                      </div>
                      <span style={{ color: COLORS.textLight }}>›</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function CategoryViewScreen({ category, products, onBack, onNavigate, onAskRuvisa }) {
  return (
    <div style={{ height: "100%", background: COLORS.cream, overflow: "auto", paddingBottom: 120 }}>
      <BackHeader onBack={onBack} title={category} />
      <div style={{ padding: "0 24px" }}>
        <p style={{ fontSize: 13, color: COLORS.textLight, margin: "0 0 16px" }}>{products.length} products for you</p>
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {products.map((p, i) => (
            <div
              key={p.product_url}
              onClick={() => onNavigate("product", p)}
              style={{ background: COLORS.white, borderRadius: 16, padding: 14, cursor: "pointer", display: "flex", gap: 14, alignItems: "center", position: "relative" }}
            >
              {i === 0 && <div style={{ position: "absolute", top: -4, right: 12, background: COLORS.sage, color: COLORS.white, padding: "2px 10px", borderRadius: 8, fontSize: 10, fontWeight: 700 }}>Best Match</div>}
              <ProductImage url={p.image_url} size={72} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ fontSize: 11, color: COLORS.textLight, margin: 0 }}>{p.brand}</p>
                <p style={{ fontSize: 15, fontWeight: 700, color: COLORS.text, margin: "2px 0 6px", lineHeight: 1.3 }}>{p.title}</p>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
                  <MatchBadge value={Math.round((p.similarity || 0) * 100)} />
                  <span style={{ fontWeight: 800, color: COLORS.forest, fontSize: 14 }}>{p.price}</span>
                </div>
              </div>
              <span style={{ color: COLORS.textLight, fontSize: 18 }}>›</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ProductDetailScreen({ product, onBack, onAskRuvisa, onPurchase, userId, onLike, liked, onAddBag }) {
  const [purchased, setPurchased] = useState(false);
  const [addedBag, setAddedBag] = useState(false);
  const [fullDescOpen, setFullDescOpen] = useState(false);
  const [reviews, setReviews] = useState([]);
  const [reviewsLoading, setReviewsLoading] = useState(true);

  useEffect(() => {
    if (product?.product_url) {
      setReviewsLoading(true);
      api.getProductReviews(product.product_url)
        .then(d => setReviews(d.reviews || []))
        .catch(() => setReviews([]))
        .finally(() => setReviewsLoading(false));
    } else {
      setReviewsLoading(false);
    }
  }, [product?.product_url]);

  if (!product) return null;
  const matchPct = Math.round((product.similarity || 0) * 100);
  const evidenceScores = product.evidence_scores || {};
  const activeConcerns = Object.entries(evidenceScores).filter(([, v]) => v > 0);

  return (
    <div style={{ height: "100%", background: COLORS.cream, overflow: "auto", paddingBottom: 120 }}>
      <BackHeader onBack={onBack} title="Product Details" right={
        onLike && (
          <button onClick={() => onLike(product)} style={{ background: "none", border: "none", fontSize: 24, cursor: "pointer" }}>
            {liked ? "❤️" : "🤍"}
          </button>
        )
      } />

      <div style={{ margin: "0 24px", background: COLORS.white, borderRadius: 20, padding: 24, textAlign: "center" }}>
        <ProductImage url={product.image_url} size={160} />
        <p style={{ fontSize: 12, color: COLORS.textLight, margin: "12px 0 4px" }}>{product.brand}</p>
        <h3 style={{ fontSize: 20, fontWeight: 800, color: COLORS.text, margin: "0 0 4px" }}>{product.title}</h3>
        {product.variant && <p style={{ fontSize: 12, color: COLORS.textLight, margin: "0 0 8px" }}>{product.variant}</p>}
        <p style={{ fontSize: 13, color: COLORS.textLight, margin: "0 0 12px" }}>{product.category}</p>
        <div style={{ display: "flex", justifyContent: "center", gap: 8, marginBottom: 8 }}>
          {matchPct > 0 && <MatchBadge value={matchPct} />}
          {product.skin_match && <span style={{ background: COLORS.overlaySoft, color: COLORS.sageDark, padding: "3px 10px", borderRadius: 20, fontSize: 12, fontWeight: 700 }}>Your Skin Type</span>}
        </div>
        <div style={{ display: "flex", justifyContent: "center", gap: 16, marginTop: 4 }}>
          <span style={{ fontWeight: 800, color: COLORS.forest, fontSize: 18 }}>{product.price}</span>
          <span style={{ fontSize: 14, color: COLORS.textLight }}>⭐ {product.rating || "N/A"} ({product.review_count || 0})</span>
        </div>
      </div>

      <div style={{ margin: "16px 24px", display: "flex", gap: 6, flexWrap: "wrap" }}>
        {product.skin_type && <span style={{ fontSize: 12, background: COLORS.overlaySoft, color: COLORS.sageDark, padding: "6px 12px", borderRadius: 12 }}>Skin Type: {product.skin_type}</span>}
        {product.formulation && <span style={{ fontSize: 12, background: "#E3F2FD", color: "#1565C0", padding: "6px 12px", borderRadius: 12 }}>{product.formulation}</span>}
        {product.skin_concerns && <span style={{ fontSize: 12, background: "#FFF3E0", color: "#E65100", padding: "6px 12px", borderRadius: 12 }}>For: {product.skin_concerns}</span>}
      </div>

      {product.what_it_is && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 15, fontWeight: 700, color: COLORS.forest, margin: "0 0 8px" }}>What It Is</h4>
          <p style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.6, margin: 0 }}>{product.what_it_is}</p>
        </div>
      )}

      {product.what_it_does && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 15, fontWeight: 700, color: COLORS.forest, margin: "0 0 8px" }}>What It Does</h4>
          <p style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.6, margin: 0 }}>{product.what_it_does}</p>
        </div>
      )}

      {activeConcerns.length > 0 && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 15, fontWeight: 700, color: COLORS.forest, margin: "0 0 12px" }}>How It Helps Your Concerns</h4>
          {activeConcerns.sort((a, b) => b[1] - a[1]).map(([key, val]) => (
            <div key={key} style={{ marginBottom: 10 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                <span style={{ fontSize: 13, fontWeight: 600 }}>{CONCERN_LABELS[key] || key}</span>
                <span style={{ fontSize: 13, fontWeight: 700, color: COLORS.sage }}>{(val * 100).toFixed(0)}%</span>
              </div>
              <div style={{ width: "100%", height: 5, background: COLORS.creamDark, borderRadius: 3, overflow: "hidden" }}>
                <div style={{ height: "100%", borderRadius: 3, width: `${val * 100}%`, background: COLORS.sage, transition: "width 0.5s" }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {product.top_ingredients?.length > 0 && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 15, fontWeight: 700, color: COLORS.forest, margin: "0 0 10px" }}>Key Active Ingredients</h4>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {product.top_ingredients.map((ing, i) => (
              <span key={i} style={{ background: COLORS.overlaySoft, color: COLORS.sageDark, padding: "6px 12px", borderRadius: 20, fontSize: 12, fontWeight: 600 }}>✓ {ing}</span>
            ))}
          </div>
        </div>
      )}

      {product.ingredients?.length > 0 && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 15, fontWeight: 700, color: COLORS.forest, margin: "0 0 8px" }}>Ingredients</h4>
          <p style={{ fontSize: 12, color: COLORS.textLight, lineHeight: 1.6, margin: 0 }}>{product.ingredients.join(", ")}</p>
        </div>
      )}

      {(product.age_range || (product.description_raw && product.description_raw.includes("Skincare By Age"))) && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 15, fontWeight: 700, color: COLORS.forest, margin: "0 0 8px" }}>Best For Age</h4>
          <p style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.6, margin: 0 }}>
            {product.age_range || product.description_raw?.match(/Skincare By Age:\s*([\w\s,+\-]+)/)?.[1]?.trim() || "All ages"}
          </p>
        </div>
      )}

      {product.formulation && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <h4 style={{ fontSize: 15, fontWeight: 700, color: COLORS.forest, margin: "0 0 8px" }}>Formulation</h4>
          <p style={{ fontSize: 13, color: COLORS.text, lineHeight: 1.6, margin: 0 }}>{product.formulation}</p>
        </div>
      )}

      {product.description_raw && (
        <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
          <button onClick={() => setFullDescOpen(!fullDescOpen)} style={{ width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between", background: "none", border: "none", cursor: "pointer", padding: 0 }}>
            <span style={{ fontSize: 15, fontWeight: 700, color: COLORS.forest }}>Full description</span>
            <span style={{ fontSize: 18, fontWeight: 700, color: COLORS.sage }}>{fullDescOpen ? "−" : "+"}</span>
          </button>
          {fullDescOpen && (
            <p style={{ fontSize: 12, color: COLORS.textLight, lineHeight: 1.6, margin: "12px 0 0", whiteSpace: "pre-wrap" }}>{product.description_raw}</p>
          )}
        </div>
      )}

      <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 16 }}>
        <h4 style={{ fontSize: 15, fontWeight: 700, color: COLORS.forest, margin: "0 0 12px" }}>User Reviews {reviews.length > 0 && `(${reviews.length})`}</h4>
        {reviewsLoading ? (
          <p style={{ fontSize: 13, color: COLORS.textLight, margin: 0 }}>Loading reviews...</p>
        ) : reviews.length > 0 ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 14, maxHeight: 320, overflowY: "auto" }}>
            {reviews.map((r, i) => (
              <div key={i} style={{ paddingBottom: 12, borderBottom: i < reviews.length - 1 ? `1px solid ${COLORS.creamDark}` : "none" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  <span style={{ fontSize: 13, fontWeight: 700, color: COLORS.text }}>{r.reviewer_name || "Anonymous"}</span>
                  <span style={{ fontSize: 12, color: COLORS.gold }}>{"★".repeat(r.rating || 0)}{"☆".repeat(5 - (r.rating || 0))}</span>
                </div>
                {r.headline && <p style={{ fontSize: 12, fontWeight: 600, color: COLORS.forest, margin: "0 0 4px" }}>{r.headline}</p>}
                <p style={{ fontSize: 12, color: COLORS.textLight, lineHeight: 1.5, margin: 0 }}>{r.review_text}</p>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ fontSize: 13, color: COLORS.textLight, margin: 0 }}>No reviews yet for this product.</p>
        )}
      </div>

      {product.product_claims?.length > 0 && (
        <div style={{ margin: "0 24px 16px", display: "flex", gap: 6, flexWrap: "wrap" }}>
          {product.product_claims.map((c, i) => (
            <span key={i} style={{ fontSize: 12, background: COLORS.overlaySoft, color: COLORS.sageDark, padding: "6px 12px", borderRadius: 12, fontWeight: 600 }}>✓ {c}</span>
          ))}
        </div>
      )}

      {/* Bottom actions — inside scroll, not fixed */}
      <div style={{ padding: "16px 24px 24px", display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", gap: 8 }}>
          {onAddBag && !addedBag && (
            <button onClick={async () => { try { await onAddBag(product); setAddedBag(true); } catch {} }} style={{
              flex: 1, padding: "14px", borderRadius: 14, border: `1.5px solid ${COLORS.sage}`,
              background: COLORS.white, color: COLORS.forest, fontSize: 14, fontWeight: 700, cursor: "pointer",
            }}>
              🛍️ Add to Bag
            </button>
          )}
          {addedBag && (
            <div style={{ flex: 1, padding: "14px", borderRadius: 14, background: COLORS.overlaySoft, color: COLORS.sageDark, fontSize: 14, fontWeight: 700, textAlign: "center" }}>
              ✓ In your bag
            </div>
          )}
          {onPurchase && userId && !purchased && (
            <button onClick={async () => { try { await onPurchase(userId, product); setPurchased(true); } catch {} }} style={{
              flex: 1, padding: "14px", borderRadius: 14, border: "none",
              background: COLORS.gold, color: COLORS.text, fontSize: 14, fontWeight: 700, cursor: "pointer",
            }}>
              ✓ Purchased
            </button>
          )}
          {purchased && (
            <div style={{ flex: 1, padding: "14px", borderRadius: 14, background: COLORS.overlaySoft, color: COLORS.sageDark, fontSize: 14, fontWeight: 700, textAlign: "center" }}>
              ✓ Purchased
            </div>
          )}
        </div>
        <button onClick={() => onAskRuvisa(product)} style={{
          width: "100%", padding: "12px", borderRadius: 14, border: `1.5px solid ${COLORS.sage}`,
          background: COLORS.overlayMedium, color: COLORS.forest, fontSize: 14, fontWeight: 700, cursor: "pointer",
        }}>
          🤖 Ask Ruvisa about this product
        </button>
        <Btn onClick={() => window.open(product.product_url, "_blank")}>
          View on Sephora • {product.price}
        </Btn>
      </div>
    </div>
  );
}

/* ---- CHAT SCREEN ---- */

function ChatScreen({ onBack, userId, skinType, initialMessage }) {
  const [messages, setMessages] = useState([
    { role: "assistant", text: "Hi! I'm Ruvisa — your personal skincare assistant. I can help you understand why a product matches your skin, recommend routines, explain ingredients, and answer any skincare question. How can I help?" },
  ]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const chatRef = useRef(null);
  const sentInitial = useRef(false);

  const TOOL_LABELS = {
    recommend_products: { icon: "💎", label: "Searching products for your skin" },
    get_product_info: { icon: "🔍", label: "Looking up product details" },
    search_products: { icon: "🔎", label: "Searching product database" },
    get_user_profile: { icon: "👤", label: "Reading your skin profile" },
    compare_analyses: { icon: "📊", label: "Comparing your scan history" },
    track_purchase: { icon: "🛍️", label: "Recording your purchase" },
  };

  const doSend = useCallback(async (text) => {
    if (!text.trim() || typing) return;
    setMessages(prev => [...prev, { role: "user", text }]);
    setInput("");
    setTyping(true);
    try {
      const data = await api.sendChat(userId, text);
      if (data.tools_used?.length > 0) {
        setMessages(prev => [...prev, { role: "tools", tools: data.tools_used }]);
      }
      setMessages(prev => [...prev, { role: "assistant", text: data.response }]);
    } catch {
      setMessages(prev => [...prev, { role: "assistant", text: "Sorry, something went wrong. Please try again." }]);
    } finally {
      setTyping(false);
    }
  }, [userId, typing]);

  useEffect(() => {
    if (initialMessage && !sentInitial.current) {
      sentInitial.current = true;
      doSend(initialMessage);
    }
  }, [initialMessage, doSend]);

  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [messages, typing]);

  return (
    <div style={{ height: "100%", background: COLORS.cream, display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "50px 24px 16px", display: "flex", alignItems: "center", gap: 12, background: COLORS.white }}>
        <button onClick={onBack} style={{ background: COLORS.creamDark, border: "none", borderRadius: 12, width: 40, height: 40, cursor: "pointer", fontSize: 18, display: "flex", alignItems: "center", justifyContent: "center" }}>←</button>
        <div style={{ width: 36, height: 36, borderRadius: 18, background: COLORS.sage, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span style={{ fontSize: 18 }}>🤖</span>
        </div>
        <div>
          <p style={{ fontSize: 15, fontWeight: 700, color: COLORS.text, margin: 0 }}>Ruvisa AI</p>
          <p style={{ fontSize: 11, color: COLORS.accent, margin: 0 }}>Online</p>
        </div>
      </div>

      <div ref={chatRef} style={{ flex: 1, overflow: "auto", padding: "16px 24px 24px" }}>
        {messages.map((msg, i) => {
          if (msg.role === "tools") {
            return (
              <div key={i} style={{ marginBottom: 10, display: "flex", justifyContent: "flex-start" }}>
                <div style={{ maxWidth: "85%", padding: "8px 12px", borderRadius: 12, background: COLORS.overlayMedium, border: `1px solid ${COLORS.borderRose}` }}>
                  <p style={{ fontSize: 10, fontWeight: 700, color: COLORS.sage, margin: "0 0 4px", textTransform: "uppercase", letterSpacing: 0.5 }}>Agent Tools Used</p>
                  {msg.tools.map((t, j) => {
                    const info = TOOL_LABELS[t.name] || { icon: "⚙️", label: t.name };
                    return (
                      <div key={j} style={{ display: "flex", alignItems: "center", gap: 6, padding: "3px 0" }}>
                        <span style={{ fontSize: 14 }}>{info.icon}</span>
                        <span style={{ fontSize: 12, color: COLORS.forest, fontWeight: 600 }}>{info.label}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          }
          return (
            <div key={i} style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start", marginBottom: 12 }}>
              <div style={{
                maxWidth: "80%", padding: "12px 16px", borderRadius: 18,
                background: msg.role === "user" ? COLORS.sage : COLORS.white,
                color: msg.role === "user" ? COLORS.white : COLORS.text,
                borderBottomRightRadius: msg.role === "user" ? 4 : 18,
                borderBottomLeftRadius: msg.role === "assistant" ? 4 : 18,
                fontSize: 14, lineHeight: 1.5, whiteSpace: "pre-line",
              }}>{msg.text}</div>
            </div>
          );
        })}
        {typing && (
          <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 12 }}>
            <div style={{ background: COLORS.white, padding: "12px 20px", borderRadius: 18, borderBottomLeftRadius: 4 }}>
              <span style={{ fontSize: 18, letterSpacing: 4 }}>...</span>
            </div>
          </div>
        )}
      </div>

      <div style={{ padding: "8px 24px 4px", display: "flex", gap: 8, overflowX: "auto" }}>
        {["What products help with acne?", "Compare my skin progress", "Build me a skincare routine"].map((q, i) => (
          <button key={i} onClick={() => setInput(q)} style={{
            padding: "8px 14px", borderRadius: 20, border: `1px solid ${COLORS.sageLight}`,
            background: COLORS.white, fontSize: 12, fontWeight: 600, color: COLORS.forest, cursor: "pointer", whiteSpace: "nowrap",
          }}>{q}</button>
        ))}
      </div>

      {/* paddingBottom clears floating bottom nav (~74px) + scan button lip (~24px) */}
      <div style={{ padding: "12px 24px", paddingBottom: "max(100px, calc(90px + env(safe-area-inset-bottom, 0px)))", display: "flex", gap: 10, alignItems: "center" }}>
        <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === "Enter" && doSend(input)}
          placeholder="Ask me anything about skincare..."
          style={{ flex: 1, minHeight: 50, padding: "16px 20px", borderRadius: 26, border: `1px solid ${COLORS.creamDark}`, fontSize: 15, outline: "none", background: COLORS.white, boxSizing: "border-box" }} />
        <button onClick={() => doSend(input)} disabled={typing} style={{ width: 52, height: 52, flexShrink: 0, borderRadius: 26, border: "none", background: typing ? COLORS.creamDark : COLORS.sage, color: COLORS.white, cursor: typing ? "default" : "pointer", fontSize: 18, display: "flex", alignItems: "center", justifyContent: "center" }}>↑</button>
      </div>
    </div>
  );
}

/* ---- PROFILE & SETTINGS SCREEN ---- */

function ProfileScreen({ onBack, onLogout, userId, userName, user, onUserUpdate }) {
  const [tab, setTab] = useState("profile"); // "profile" | "settings"
  const [journey, setJourney] = useState(null);
  const [loading, setLoading] = useState(true);

  // Settings state
  const [name, setName] = useState(user?.name || "");
  const [email, setEmail] = useState(user?.email || "");
  const [curPw, setCurPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState(null);

  useEffect(() => {
    if (userId !== "guest") {
      api.getJourney(userId).then(setJourney).catch(() => {}).finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, [userId]);

  const handleSave = async () => {
    setSaving(true);
    setMsg(null);
    try {
      const updated = await api.updateSettings(userId, {
        name: name !== user?.name ? name : undefined,
        email: email !== user?.email ? email : undefined,
        currentPassword: curPw || undefined,
        newPassword: newPw || undefined,
      });
      if (onUserUpdate) onUserUpdate(updated);
      setCurPw("");
      setNewPw("");
      setMsg({ type: "success", text: "Settings saved!" });
    } catch (err) {
      setMsg({ type: "error", text: err.message?.includes("400") ? "Current password is incorrect" : "Failed to save" });
    } finally {
      setSaving(false);
    }
  };

  const imp = journey?.improvement;
  const analyses = journey?.analyses || [];
  const purchases = journey?.purchases || [];

  return (
    <div style={{ height: "100%", background: COLORS.cream, overflow: "auto", paddingBottom: 120 }}>
      <div style={{ padding: "50px 24px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <button onClick={onBack} style={{ background: COLORS.white, border: "none", borderRadius: 12, width: 40, height: 40, cursor: "pointer", fontSize: 18, display: "flex", alignItems: "center", justifyContent: "center" }}>←</button>
          <h2 style={{ fontSize: 22, fontWeight: 800, color: COLORS.forest, margin: 0 }}>Profile</h2>
        </div>
      </div>

      {/* User info card */}
      <div style={{ margin: "0 24px 16px", background: COLORS.white, borderRadius: 16, padding: 20, textAlign: "center" }}>
        <div style={{ width: 64, height: 64, borderRadius: 32, background: COLORS.sageLight, display: "inline-flex", alignItems: "center", justifyContent: "center", marginBottom: 10 }}>
          <User size={30} color={COLORS.white} strokeWidth={2} />
        </div>
        <p style={{ fontSize: 18, fontWeight: 800, color: COLORS.forest, margin: "0 0 2px" }}>{user?.name || userName}</p>
        <p style={{ fontSize: 13, color: COLORS.textLight, margin: 0 }}>{user?.email || ""}</p>
        <p style={{ fontSize: 12, color: COLORS.sage, fontWeight: 600, margin: "4px 0 0", textTransform: "capitalize" }}>{user?.skin_type || "—"} skin</p>
      </div>

      {/* Tab switcher */}
      <div style={{ margin: "0 24px 16px", display: "flex", background: COLORS.creamDark, borderRadius: 12, padding: 3 }}>
        {[{ id: "profile", label: "Journey" }, { id: "settings", label: "Settings" }].map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            flex: 1, padding: "10px", borderRadius: 10, border: "none", fontSize: 14, fontWeight: 700, cursor: "pointer",
            background: tab === t.id ? COLORS.white : "transparent",
            color: tab === t.id ? COLORS.forest : COLORS.textLight,
          }}>{t.label}</button>
        ))}
      </div>

      {tab === "profile" ? (
        <>
          {/* Improvement banner */}
          {imp && (
            <div style={{ margin: "0 24px 16px", padding: 16, borderRadius: 16, background: imp.improvement_pct >= 0 ? COLORS.overlaySoft : "#FFF3E0" }}>
              <p style={{ fontSize: 14, fontWeight: 700, color: COLORS.forest, margin: 0 }}>
                {imp.improvement_pct > 0 ? `Skin improved ${imp.improvement_pct}%` : imp.improvement_pct < 0 ? "Skin needs attention" : "No change"}
              </p>
              <p style={{ fontSize: 12, color: COLORS.textLight, margin: "4px 0 0" }}>Score: {imp.previous_score} → {imp.latest_score}</p>
            </div>
          )}

          {/* Stats */}
          <div style={{ margin: "0 24px 16px", display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
            {[
              { val: analyses.length, label: "Scans" },
              { val: purchases.length, label: "Purchases" },
              { val: user?.concerns?.length || 0, label: "Concerns" },
            ].map(s => (
              <div key={s.label} style={{ background: COLORS.white, borderRadius: 14, padding: 14, textAlign: "center" }}>
                <p style={{ fontSize: 22, fontWeight: 800, color: COLORS.forest, margin: 0 }}>{s.val}</p>
                <p style={{ fontSize: 11, color: COLORS.textLight, margin: "2px 0 0" }}>{s.label}</p>
              </div>
            ))}
          </div>

          {/* Past scans */}
          <div style={{ padding: "0 24px 16px" }}>
            <h3 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: "0 0 10px" }}>Past Scans</h3>
            {analyses.length === 0 ? (
              <p style={{ color: COLORS.textLight, fontSize: 13 }}>No scans yet</p>
            ) : analyses.map((a, i) => (
              <div key={a.id} style={{ background: COLORS.white, borderRadius: 12, padding: 12, marginBottom: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 13, fontWeight: 700, color: COLORS.text }}>Scan {analyses.length - i}</span>
                  <span style={{ fontSize: 12, color: COLORS.textLight }}>{new Date(a.created_at).toLocaleDateString()}</span>
                </div>
                {a.concern_vector && (
                  <p style={{ fontSize: 12, color: COLORS.textLight, margin: "4px 0 0" }}>
                    {(Array.isArray(a.concern_vector) ? a.concern_vector : []).filter(v => v > 0.1).length} concerns detected
                  </p>
                )}
              </div>
            ))}
          </div>

          {/* Past purchases */}
          <div style={{ padding: "0 24px 16px" }}>
            <h3 style={{ fontSize: 16, fontWeight: 800, color: COLORS.forest, margin: "0 0 10px" }}>Purchases</h3>
            {purchases.length === 0 ? (
              <p style={{ color: COLORS.textLight, fontSize: 13 }}>No purchases yet</p>
            ) : purchases.map(p => (
              <div key={p.id} style={{ background: COLORS.white, borderRadius: 12, padding: 12, marginBottom: 8, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ fontSize: 13, fontWeight: 600, color: COLORS.text, margin: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p.product_title || "Product"}</p>
                  <p style={{ fontSize: 11, color: COLORS.textLight, margin: "2px 0 0" }}>{new Date(p.purchased_at).toLocaleDateString()}</p>
                </div>
                {p.price > 0 && <span style={{ fontSize: 13, color: COLORS.forest, fontWeight: 700 }}>${p.price}</span>}
              </div>
            ))}
          </div>
        </>
      ) : (
        /* Settings tab */
        <div style={{ padding: "0 24px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div>
              <label style={{ fontSize: 12, fontWeight: 700, color: COLORS.textLight, display: "block", marginBottom: 6 }}>Name</label>
              <input value={name} onChange={e => setName(e.target.value)}
                style={{ width: "100%", padding: "14px 16px", borderRadius: 12, border: `1.5px solid ${COLORS.creamDark}`, fontSize: 15, outline: "none", background: COLORS.white, color: COLORS.text }} />
            </div>
            <div>
              <label style={{ fontSize: 12, fontWeight: 700, color: COLORS.textLight, display: "block", marginBottom: 6 }}>Email</label>
              <input value={email} onChange={e => setEmail(e.target.value)} type="email"
                style={{ width: "100%", padding: "14px 16px", borderRadius: 12, border: `1.5px solid ${COLORS.creamDark}`, fontSize: 15, outline: "none", background: COLORS.white, color: COLORS.text }} />
            </div>

            <div style={{ marginTop: 8, padding: "16px", background: COLORS.white, borderRadius: 14 }}>
              <p style={{ fontSize: 14, fontWeight: 700, color: COLORS.forest, margin: "0 0 12px" }}>Change Password</p>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <input value={curPw} onChange={e => setCurPw(e.target.value)} placeholder="Current password" type="password"
                  style={{ width: "100%", padding: "12px 16px", borderRadius: 12, border: `1.5px solid ${COLORS.creamDark}`, fontSize: 14, outline: "none", background: COLORS.cream, color: COLORS.text }} />
                <input value={newPw} onChange={e => setNewPw(e.target.value)} placeholder="New password" type="password"
                  style={{ width: "100%", padding: "12px 16px", borderRadius: 12, border: `1.5px solid ${COLORS.creamDark}`, fontSize: 14, outline: "none", background: COLORS.cream, color: COLORS.text }} />
              </div>
            </div>

            {msg && (
              <p style={{ fontSize: 13, textAlign: "center", color: msg.type === "success" ? COLORS.sageDark : COLORS.red, fontWeight: 600 }}>{msg.text}</p>
            )}

            <Btn onClick={handleSave} disabled={saving}>
              {saving ? "Saving..." : "Save Changes"}
            </Btn>
          </div>

          <div style={{ marginTop: 24 }}>
            <button onClick={onLogout} style={{
              width: "100%", padding: "14px", borderRadius: 14,
              border: `1.5px solid ${COLORS.red}`, background: "transparent",
              color: COLORS.red, fontSize: 14, fontWeight: 700, cursor: "pointer",
            }}>
              Sign Out
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---- AUTH SCREENS ---- */

function AuthScreen({ onAuth }) {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!email.trim() || !password.trim()) { setError("Please fill in all fields"); return; }
    if (mode === "signup" && !name.trim()) { setError("Please enter your name"); return; }
    setLoading(true);
    setError(null);
    try {
      let user;
      if (mode === "signup") {
        user = await api.registerUser(name.trim(), email.trim().toLowerCase(), password);
      } else {
        user = await api.loginUser(email.trim().toLowerCase(), password);
      }
      localStorage.setItem("ruvisa_user", JSON.stringify(user));
      onAuth(user);
    } catch (err) {
      setError(err.message.includes("409") ? "An account with this email already exists" :
               err.message.includes("401") ? "Invalid email or password" :
               "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ height: "100%", background: COLORS.cream, display: "flex", flexDirection: "column" }}>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", padding: "0 30px" }}>
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <div style={{ width: 70, height: 70, borderRadius: 18, background: COLORS.sage, display: "inline-flex", alignItems: "center", justifyContent: "center", marginBottom: 16, boxShadow: `0 6px 24px ${COLORS.shadowWarm}` }}>
            <span style={{ fontSize: 32, color: COLORS.white, fontWeight: 800, fontFamily: "serif" }}>R</span>
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 800, color: COLORS.forest, margin: 0 }}>
            {mode === "login" ? "Welcome Back" : "Create Account"}
          </h1>
          <p style={{ color: COLORS.textLight, fontSize: 14, marginTop: 6 }}>
            {mode === "login" ? "Sign in to continue your skincare journey" : "Start your personalized skincare journey"}
          </p>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          {mode === "signup" && (
            <input value={name} onChange={e => setName(e.target.value)} placeholder="Your name"
              style={{ padding: "16px 18px", borderRadius: 14, border: `1.5px solid ${COLORS.creamDark}`, fontSize: 15, outline: "none", background: COLORS.white, color: COLORS.text }} />
          )}
          <input value={email} onChange={e => setEmail(e.target.value)} placeholder="Email address" type="email"
            style={{ padding: "16px 18px", borderRadius: 14, border: `1.5px solid ${COLORS.creamDark}`, fontSize: 15, outline: "none", background: COLORS.white, color: COLORS.text }} />
          <input value={password} onChange={e => setPassword(e.target.value)} placeholder="Password" type="password"
            onKeyDown={e => e.key === "Enter" && handleSubmit()}
            style={{ padding: "16px 18px", borderRadius: 14, border: `1.5px solid ${COLORS.creamDark}`, fontSize: 15, outline: "none", background: COLORS.white, color: COLORS.text }} />
        </div>

        {error && <p style={{ color: COLORS.red, fontSize: 13, marginTop: 10, textAlign: "center" }}>{error}</p>}

        <div style={{ marginTop: 20 }}>
          <Btn onClick={handleSubmit} disabled={loading}>
            {loading ? "Please wait..." : mode === "login" ? "Sign In" : "Create Account"}
          </Btn>
        </div>

        <button onClick={() => { setMode(mode === "login" ? "signup" : "login"); setError(null); }}
          style={{ background: "none", border: "none", color: COLORS.sage, fontSize: 14, fontWeight: 600, cursor: "pointer", marginTop: 16, textAlign: "center", width: "100%" }}>
          {mode === "login" ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
        </button>
      </div>
    </div>
  );
}

/* ---- Main App ---- */

const NAV_SCREENS = ["home", "shop", "scan", "bag", "chat"];

export default function RuvisaApp() {
  const [screen, setScreen] = useState("splash");
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [categoryView, setCategoryView] = useState(null);
  const [history, setHistory] = useState([]);
  const [likedUrls, setLikedUrls] = useState(new Set());
  const [bagCount, setBagCount] = useState(0);
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem("ruvisa_user")); } catch { return null; }
  });
  const [skinType, setSkinType] = useState(() => user?.skin_type || null);
  const [userConcerns, setUserConcerns] = useState(() => user?.concerns || []);
  const [analysis, setAnalysis] = useState(null);
  const [chatInitialMsg, setChatInitialMsg] = useState(null);
  const [scanDetailData, setScanDetailData] = useState(null);
  /** Lifted from HomeScreen: refetch only when userId / skinType / analysis change, not on tab navigation. */
  const [homeRecs, setHomeRecs] = useState({});
  const [homeTrending, setHomeTrending] = useState([]);
  const [homeJourney, setHomeJourney] = useState(null);

  const userId = user?.user_id || "guest";
  const userName = user?.name || "there";

  useEffect(() => {
    if (userId !== "guest") {
      api.getLiked(userId).then(d => {
        setLikedUrls(new Set((d.items || []).map(i => i.product_url)));
      }).catch(() => {});
      api.getBag(userId).then(d => {
        setBagCount((d.items || []).length);
      }).catch(() => {});
    }
  }, [userId]);

  useEffect(() => {
    let cancelled = false;
    const promises = [];

    if (skinType) {
      promises.push(
        api
          .getRecommendations(userId, skinType, analysis?.concern_vector, null, 5)
          .then((data) => {
            if (!cancelled) {
              setHomeRecs({ ...(data.recommendations || {}), routine: data.routine || [] });
            }
          })
          .catch(() => {
            if (!cancelled) setHomeRecs({});
          })
      );
    } else if (!cancelled) {
      setHomeRecs({});
    }

    promises.push(
      api.getTrending(8).then((d) => {
        if (!cancelled) setHomeTrending(d.products || []);
      }).catch(() => {})
    );

    if (userId !== "guest") {
      promises.push(
        api.getJourney(userId).then((j) => {
          if (!cancelled) setHomeJourney(j);
        }).catch(() => {})
      );
    } else if (!cancelled) {
      setHomeJourney(null);
    }

    return () => {
      cancelled = true;
    };
  }, [userId, skinType, analysis]);

  const navigate = useCallback((to, data) => {
    setHistory(prev => [...prev, screen]);
    if (to === "product") {
      setSelectedProduct(data);
      setScreen("product");
    } else if (to === "category") {
      setCategoryView(data);
      setScreen("category");
    } else if (to === "scanDetail") {
      setScanDetailData(data);
      setScreen("scanDetail");
    } else if (to === "recommendations") {
      setScreen("recommendations");
    } else {
      if (to === "chat" && !data) setChatInitialMsg(null);
      setScreen(to);
    }
  }, [screen]);

  const goBack = useCallback(() => {
    const prev = history[history.length - 1] || "home";
    setHistory(h => h.slice(0, -1));
    setScreen(prev);
  }, [history]);

  const handleAskRuvisa = useCallback((product) => {
    const msg = `I'd like to know more about "${product.title}" by ${product.brand}. Why is this product a good match for my skin? What are its key ingredients and how do they help my skin concerns?`;
    setChatInitialMsg(msg);
    setHistory(prev => [...prev, screen]);
    setScreen("chat");
  }, [screen]);

  const handleLike = useCallback(async (product) => {
    if (userId === "guest") return;
    try {
      const res = await api.toggleLike(userId, product);
      setLikedUrls(prev => {
        const next = new Set(prev);
        if (res.liked) next.add(product.product_url);
        else next.delete(product.product_url);
        return next;
      });
    } catch {}
  }, [userId]);

  const handleAddBag = useCallback(async (product) => {
    if (userId === "guest") return;
    try {
      await api.addToBag(userId, product);
      setBagCount(prev => prev + 1);
    } catch {}
  }, [userId]);

  const handleAuth = (userData) => {
    setUser(userData);
    if (userData.skin_type) setSkinType(userData.skin_type);
    if (userData.concerns) setUserConcerns(userData.concerns);
    if (userData.skin_type) {
      setScreen("home");
    } else {
      setScreen("skintype");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("ruvisa_user");
    setUser(null);
    setSkinType(null);
    setUserConcerns([]);
    setAnalysis(null);
    setHistory([]);
    setLikedUrls(new Set());
    setBagCount(0);
    setHomeRecs({});
    setHomeTrending([]);
    setHomeJourney(null);
    setScreen("splash");
  };

  return (
    <div style={{ width: "100%", maxWidth: 430, margin: "0 auto", height: "100vh", background: COLORS.cream, position: "relative", overflow: "hidden", fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", contain: "layout" }}>
      <style>{`
        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        ::-webkit-scrollbar { display: none; }
        @keyframes pulse { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }
      `}</style>

      <div style={{ height: "100%", overflow: "hidden" }}>
        {screen === "splash" && <SplashScreen onNext={() => setScreen(user ? "home" : "onboarding")} />}
        {screen === "onboarding" && <OnboardingScreen onNext={() => setScreen("auth")} />}
        {screen === "auth" && <AuthScreen onAuth={handleAuth} />}
        {screen === "skintype" && (
          <SkinTypeScreen onSelect={(type, concerns) => {
            setSkinType(type);
            setUserConcerns(concerns);
            if (user) {
              api.getRecommendations(userId, type, null, null, 1).catch(() => {});
              const updated = { ...user, skin_type: type, concerns };
              localStorage.setItem("ruvisa_user", JSON.stringify(updated));
              setUser(updated);
            }
            setScreen("scan");
          }} />
        )}
        {screen === "scan" && (
          <FaceScanScreen
            skinType={skinType || "oily"}
            userId={userId}
            onComplete={(result) => {
              setAnalysis(result);
              setHistory(prev => [...prev, "scan"]);
              setScreen("home");
            }}
          />
        )}
        {screen === "home" && (
          <HomeScreen
            onNavigate={navigate}
            userId={userId}
            userName={userName}
            skinType={skinType}
            analysis={analysis}
            onLogout={handleLogout}
            likedUrls={likedUrls}
            onLike={handleLike}
            onAddBag={handleAddBag}
            recs={homeRecs}
            trending={homeTrending}
            journey={homeJourney}
          />
        )}
        {screen === "shop" && (
          <ShopScreen
            onNavigate={navigate}
            onLike={handleLike}
            likedUrls={likedUrls}
            onAddBag={handleAddBag}
          />
        )}
        {screen === "bag" && (
          <BagScreen
            userId={userId}
            onNavigate={navigate}
          />
        )}
        {screen === "recommendations" && (
          <RecommendationsScreen
            onBack={goBack}
            onNavigate={navigate}
            onAskRuvisa={handleAskRuvisa}
            userId={userId}
            skinType={skinType || "oily"}
            concernVector={analysis?.concern_vector}
            analysis={analysis}
            onLike={handleLike}
            likedUrls={likedUrls}
            onAddBag={handleAddBag}
          />
        )}
        {screen === "product" && (
          <ProductDetailScreen
            product={selectedProduct}
            onBack={goBack}
            onAskRuvisa={handleAskRuvisa}
            onPurchase={api.recordPurchase}
            userId={userId}
            onLike={handleLike}
            liked={likedUrls.has(selectedProduct?.product_url)}
            onAddBag={handleAddBag}
          />
        )}
        {screen === "category" && categoryView && (
          <CategoryViewScreen
            category={categoryView.category}
            products={categoryView.products}
            onBack={goBack}
            onNavigate={navigate}
            onAskRuvisa={handleAskRuvisa}
          />
        )}
        {screen === "scanDetail" && (
          <ScanDetailScreen scanData={scanDetailData} onBack={goBack} />
        )}
        {screen === "profile" && (
          <ProfileScreen
            onBack={goBack}
            onLogout={handleLogout}
            userId={userId}
            userName={userName}
            user={user}
            onUserUpdate={(updated) => {
              const u = { ...user, ...updated };
              setUser(u);
              localStorage.setItem("ruvisa_user", JSON.stringify(u));
            }}
          />
        )}
        {screen === "chat" && <ChatScreen onBack={goBack} userId={userId} skinType={skinType} initialMessage={chatInitialMsg} />}
      </div>

      {NAV_SCREENS.includes(screen) && (
        <div style={{
          position: "absolute",
          bottom: 14,
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 1000,
          width: "min(300px, calc(100% - 56px))",
          height: 60,
          background: "rgba(249, 247, 243, 0.94)",
          border: "1px solid rgba(140, 105, 85, 0.22)",
          borderRadius: 22,
          boxShadow: `0 10px 26px ${COLORS.shadowWarm}, 0 2px 8px rgba(52, 30, 38, 0.12)`,
          display: "flex",
          justifyContent: "space-around",
          alignItems: "center",
          padding: "8px 6px 10px",
        }}>
          {[
            { id: "home", label: "Home", Icon: Home },
            { id: "shop", label: "Shop", Icon: ShoppingCart },
            { id: "scan", label: "Scan", Icon: ScanFace },
            { id: "bag", label: "Bag", Icon: ShoppingBag, badge: bagCount },
            { id: "chat", label: "Ruvisa", Icon: Sparkles },
          ].map(item => {
            const Icon = item.Icon;
            const navActiveColor = "hsl(22, 42%, 46%)";
            const navInactiveColor = "hsl(25, 14%, 36%)";
            const isActive = screen === item.id;
            return (
            <button key={item.id} onClick={() => navigate(item.id)} style={{
              background: "none", border: "none", cursor: "pointer", display: "flex", flexDirection: "column", alignItems: "center", gap: 2, padding: "4px 8px",
              opacity: 1, position: "relative",
            }}>
              {item.id === "scan" ? (
                <div style={{
                  width: 54,
                  height: 54,
                  borderRadius: 18,
                  background: "hsl(42, 28%, 98%)",
                  border: `1px solid ${COLORS.borderRose}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginTop: -24,
                  boxShadow: `0 8px 18px ${COLORS.shadowWarm}, 0 2px 8px rgba(52, 30, 38, 0.14)`,
                }}>
                  <Icon size={26} color={isActive ? navActiveColor : navInactiveColor} strokeWidth={1.75} />
                </div>
              ) : (
                <Icon size={22} color={isActive ? navActiveColor : navInactiveColor} strokeWidth={1.75} />
              )}
              {item.badge > 0 && (
                <span style={{ position: "absolute", top: -2, right: 0, background: COLORS.red, color: COLORS.white, fontSize: 9, fontWeight: 700, borderRadius: 10, minWidth: 16, height: 16, display: "flex", alignItems: "center", justifyContent: "center", padding: "0 4px" }}>{item.badge}</span>
              )}
              <span style={{ fontSize: 10, fontWeight: 700, color: isActive ? navActiveColor : navInactiveColor }}>{item.label}</span>
            </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
