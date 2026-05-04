"use client";
import { Bell, MapPin, Wheat, Thermometer, Settings, Flame, ChevronDown } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { FarmContext, Lang, Notification } from "@/lib/mockData";
import ConnectionStatus from "./ConnectionStatus";

import { translations } from "@/lib/translations";

interface FarmContextBarProps {
  context: FarmContext;
  lang: Lang;
  onChangeLang: (lang: Lang) => void;
  onOpenSettings: () => void;
  onOpenNotifications: () => void;
  notifications: Notification[];
}

const LANG_OPTIONS: { code: Lang; label: string; flag: string }[] = [
  { code: "en",  label: "English",  flag: "🇬🇧" },
  { code: "ur",  label: "اردو",      flag: "🇵🇰" },
  { code: "pa",  label: "پنجابی",    flag: "🌾" },
  { code: "skr", label: "سرائیکی",   flag: "🌻" },
];

export default function FarmContextBar({
  context, lang, onChangeLang, onOpenSettings, onOpenNotifications, notifications,
}: FarmContextBarProps) {
  const isRtl = lang === "ur" || lang === "pa" || lang === "skr";
  const isNonLatin = lang !== "en";
  const [langOpen, setLangOpen] = useState(false);
  const langRef = useRef<HTMLDivElement>(null);

  const unreadCount = notifications.filter((n) => n.unread).length;
  const hasCritical = notifications.some((n) => n.severity === "critical" && n.unread);
  const current = LANG_OPTIONS.find((l) => l.code === lang) ?? LANG_OPTIONS[0];

  const cropDisplay =
    (isNonLatin
      ? context.crop.split(" — ")[1]
      : context.crop.split(" — ")[0]) ?? context.crop;

  const tempColor =
    context.tempC >= 44 ? "text-red-300" :
    context.tempC >= 38 ? "text-orange-300" :
    "text-green-200";

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (langRef.current && !langRef.current.contains(e.target as Node)) {
        setLangOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  return (
    <header className="flex-shrink-0 bg-[#154621] text-white shadow-md z-40">
      {/* Top bar */}
      <div className={`flex items-center justify-between px-6 py-4 max-w-[1400px] mx-auto w-full ${isRtl ? "flex-row-reverse" : ""}`}>
        {/* Brand */}
        <div className={`flex items-center gap-3 ${isRtl ? "flex-row-reverse" : ""}`}>
          <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-xl font-black">
            🌾
          </div>
          <div className={`flex items-center gap-2 ${isRtl ? "flex-row-reverse" : ""}`}>
            <h1 className="text-white font-black text-xl tracking-tight">AgroBot</h1>
            <span className="bg-red-500/20 text-red-400 text-[10px] font-black px-2 py-0.5 rounded-full border border-red-500/30 uppercase tracking-tighter">AI</span>
          </div>
        </div>

        {/* Right actions */}
        <div className={`flex items-center gap-4 ${isRtl ? "flex-row-reverse" : ""}`}>
          {/* Language Picker */}
          <div ref={langRef} className="relative">
            <button
              onClick={() => setLangOpen((v) => !v)}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-sm font-bold transition-all active:scale-95"
            >
              <div className="w-5 h-5 flex items-center justify-center grayscale brightness-125">
                <span className="text-lg leading-none mt-[-2px]">🌐</span>
              </div>
              <span className={`text-sm font-bold ${isNonLatin ? "urdu-text" : ""}`}>{current.label}</span>
              <ChevronDown size={14} className={`opacity-50 transition-transform ${langOpen ? "rotate-180" : ""}`} />
            </button>

            {langOpen && (
              <div className="absolute right-0 top-full mt-2 w-48 bg-white rounded-2xl shadow-2xl border border-stone-100 overflow-hidden z-50 py-2">
                {LANG_OPTIONS.map((opt) => (
                  <button
                    key={opt.code}
                    onClick={() => { onChangeLang(opt.code); setLangOpen(false); }}
                    className={`w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-emerald-50 transition-colors ${
                      opt.code === lang ? "bg-emerald-50 text-emerald-800 font-bold" : "text-stone-700"
                    }`}
                  >
                    <span className="text-xl">{opt.flag}</span>
                    <span className={`text-sm ${opt.code !== "en" ? "urdu-text text-base" : ""}`}>{opt.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* User Profile */}
          <div className="w-10 h-10 rounded-full bg-orange-600 border-2 border-orange-400/50 flex items-center justify-center text-xs font-bold shadow-lg shadow-orange-900/20">
            AK
          </div>
          
          <button
            onClick={onOpenSettings}
            className="p-2 rounded-xl hover:bg-white/10 transition-colors text-emerald-200"
          >
            <Settings size={20} />
          </button>
        </div>
      </div>
    </header>
  );
}
