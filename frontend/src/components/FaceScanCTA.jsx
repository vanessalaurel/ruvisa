import { motion } from "framer-motion";
import { Sparkles, ScanFace } from "lucide-react";

/**
 * Prominent home CTA when the user has no analysis yet.
 * Scroll-in animation + tap feedback; navigates via onStartScan (e.g. scan screen).
 */
export default function FaceScanCTA({ onStartScan }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      style={{ padding: "8px 0" }}
    >
      <motion.div
        role="button"
        tabIndex={0}
        onClick={onStartScan}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onStartScan();
          }
        }}
        whileTap={{ scale: 0.98 }}
        style={{
          position: "relative",
          margin: "0 20px 16px",
          borderRadius: 24,
          overflow: "hidden",
          background: "linear-gradient(135deg, hsl(28, 38%, 72%) 0%, hsl(32, 42%, 78%) 45%, hsl(38, 35%, 68%) 100%)",
          padding: 20,
          boxShadow: "0 20px 40px rgba(60, 42, 32, 0.18)",
          cursor: "pointer",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            right: 0,
            width: 128,
            height: 128,
            borderRadius: "50%",
            background: "rgba(255,255,255,0.1)",
            transform: "translate(25%, -50%)",
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            width: 96,
            height: 96,
            borderRadius: "50%",
            background: "rgba(255,255,255,0.1)",
            transform: "translate(-25%, 50%)",
          }}
        />

        <div style={{ position: "relative", display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ flexShrink: 0 }}>
            <PhoneIllustration />
          </div>

          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
              <ScanFace size={14} color="rgba(255,255,255,0.9)" strokeWidth={2} />
              <span
                style={{
                  color: "rgba(255,255,255,0.85)",
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                }}
              >
                AI Skin Scan
              </span>
            </div>
            <h3
              style={{
                fontFamily: "Georgia, 'Times New Roman', serif",
                fontSize: 22,
                fontWeight: 800,
                color: "#fff",
                lineHeight: 1.2,
                margin: 0,
              }}
            >
              Scan Your Face Now
            </h3>
            <p
              style={{
                color: "rgba(255,255,255,0.85)",
                fontSize: 12,
                marginTop: 6,
                lineHeight: 1.45,
              }}
            >
              Detect your skin zones, get real-time scores & personalized care
            </p>
            <div
              style={{
                marginTop: 12,
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                background: "rgba(255,255,255,0.2)",
                backdropFilter: "blur(8px)",
                WebkitBackdropFilter: "blur(8px)",
                color: "#fff",
                fontSize: 12,
                fontWeight: 700,
                padding: "6px 14px",
                borderRadius: 9999,
              }}
            >
              <Sparkles size={12} />
              Start Free Scan
            </div>
          </div>
        </div>
      </motion.div>
    </motion.section>
  );
}

function PhoneIllustration() {
  const dots = [
    { top: "8%", left: "50%", delay: 0 },
    { top: "40%", left: "18%", delay: 0.3 },
    { top: "40%", left: "82%", delay: 0.6 },
    { top: "65%", left: "50%", delay: 0.9 },
  ];

  return (
    <div style={{ position: "relative", width: 80, height: 112 }}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          borderRadius: 16,
          background: "rgba(255,255,255,0.2)",
          border: "2px solid rgba(255,255,255,0.4)",
          backdropFilter: "blur(6px)",
        }}
      />

      <div
        style={{
          position: "absolute",
          inset: 6,
          borderRadius: 12,
          background: "rgba(255,255,255,0.1)",
          overflow: "hidden",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div style={{ position: "relative", width: 44, height: 56 }}>
          <svg viewBox="0 0 44 56" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
            <ellipse
              cx="22"
              cy="28"
              rx="18"
              ry="24"
              fill="none"
              stroke="white"
              strokeWidth="1.5"
              strokeDasharray="4 3"
              opacity={0.8}
            />
          </svg>

          <div
            style={{
              position: "relative",
              zIndex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
            }}
          >
            <div
              style={{
                width: 32,
                height: 36,
                borderRadius: 999,
                background: "hsl(30, 60%, 88%)",
                border: "1px solid rgba(255,255,255,0.5)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
              }}
            >
              <div style={{ display: "flex", gap: 6, marginBottom: 4 }}>
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "hsl(20,25%,25%)" }} />
                <div style={{ width: 6, height: 6, borderRadius: "50%", background: "hsl(20,25%,25%)" }} />
              </div>
              <svg viewBox="0 0 12 6" style={{ width: 12, height: 6 }}>
                <path
                  d="M1 1 Q6 5 11 1"
                  stroke="hsl(20,25%,35%)"
                  strokeWidth="1.2"
                  fill="none"
                  strokeLinecap="round"
                />
              </svg>
              <div
                style={{
                  position: "absolute",
                  top: -6,
                  left: 0,
                  right: 0,
                  height: 12,
                  borderRadius: 999,
                  background: "hsl(20,40%,35%)",
                  overflow: "hidden",
                }}
              />
            </div>
          </div>

          {dots.map((dot, i) => (
            <motion.div
              key={i}
              style={{
                position: "absolute",
                top: dot.top,
                left: dot.left,
                transform: "translate(-50%, -50%)",
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: "#fff",
              }}
              animate={{ scale: [0, 1.3, 1], opacity: [0, 1, 0.7] }}
              transition={{ delay: dot.delay, duration: 0.6, repeat: Infinity, repeatDelay: 1.5 }}
            />
          ))}
        </div>
      </div>

      <motion.div
        style={{
          position: "absolute",
          left: 8,
          right: 8,
          height: 1,
          background: "rgba(255,255,255,0.7)",
          borderRadius: 999,
          boxShadow: "0 0 6px 2px rgba(255,255,255,0.5)",
        }}
        animate={{ top: ["20%", "85%", "20%"] }}
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
      />

      <div
        style={{
          position: "absolute",
          top: 10,
          left: "50%",
          transform: "translateX(-50%)",
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: "rgba(255,255,255,0.6)",
        }}
      />
      <div
        style={{
          position: "absolute",
          bottom: 8,
          left: "50%",
          transform: "translateX(-50%)",
          width: 20,
          height: 2,
          borderRadius: 999,
          background: "rgba(255,255,255,0.4)",
        }}
      />
    </div>
  );
}
