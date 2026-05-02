"use client";
import { MapPin, Thermometer, Sprout, Bell, Settings, Globe, CloudSync } from "lucide-react";
import { useState } from "react";
import { Lang } from "@/lib/mockData";

interface FarmContextHeaderProps {
  city: string;
  crop: string;
  temp: number;
  lang: Lang;
  onLanguageChange: (lang: Lang) => void;
  onOpenNotifications: () => void;
  onOpenSettings: () => void;
  unreadCount: number;
}

export default function FarmContextHeader({
  city,
  crop,
  temp,
  lang,
  onLanguageChange,
  onOpenNotifications,
  onOpenSettings,
  unreadCount
}: FarmContextHeaderProps) {
  const [isLanguageOpen, setIsLanguageOpen] = useState(false);
  const isRtl = lang === "ur" || lang === "pa" || lang === "skr";

  const languages = [
    { code: "en", label: "English" },
    { code: "ur", label: "اردو" },
    { code: "pa", label: "پنجابی" },
    { code: "skr", label: "سرائیکی" },
  ];

  return (
    <header className="sticky top-0 z-50 w-full h-16 bg-[#FDFBF7]/90 backdrop-blur-md border-b border-[#3A9878]/10 px-4 flex items-center justify-between">
      <div className="flex items-center gap-6">
        {/* Logo Section */}
        <div className="flex items-center gap-2">
          <div className="w-9 h-9 bg-[#3A9878] rounded-xl flex items-center justify-center shadow-lg shadow-[#3A9878]/20">
            <Sprout className="text-white w-5 h-5" />
          </div>
          <span className="text-lg font-black text-stone-800 tracking-tight hidden sm:block">
            AgroBot <span className="text-[#3A9878]">AI</span>
          </span>
        </div>

        {/* Farm Context Indicators */}
        <div className="hidden md:flex items-center gap-1 p-1 bg-stone-50 rounded-2xl border border-stone-100">
          <div className="flex items-center gap-1.5 px-3 py-1.5">
            <MapPin size={14} className="text-stone-400" />
            <span className="text-xs font-bold text-stone-700">{city}</span>
          </div>
          <div className="w-px h-3 bg-stone-200"></div>
          <div className="flex items-center gap-1.5 px-3 py-1.5">
            <Sprout size={14} className="text-emerald-500" />
            <span className="text-xs font-bold text-stone-700">{crop}</span>
          </div>
          <div className="w-px h-3 bg-stone-200"></div>
          <div className="flex items-center gap-1.5 px-3 py-1.5">
            <Thermometer size={14} className="text-orange-500" />
            <span className="text-xs font-bold text-stone-700">{temp}°C</span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-4">
        {/* Sync Status */}
        <div className="hidden lg:flex items-center gap-1.5 px-3 py-1.5 bg-[#C8EEE0]/50 text-[#1B4332] rounded-full border border-[#3A9878]/20">
          <CloudSync size={14} className="animate-pulse" />
          <span className="text-[10px] font-bold uppercase tracking-wider">Syncing Data</span>
        </div>

        {/* Language Selector */}
        <div className="relative">
          <button
            onClick={() => setIsLanguageOpen(!isLanguageOpen)}
            className="flex items-center gap-2 px-3 py-2 bg-stone-50 hover:bg-stone-100 rounded-xl transition-colors border border-stone-100"
          >
            <Globe size={18} className="text-stone-500" />
            <span className="text-sm font-bold text-stone-700 uppercase hidden sm:block">{lang}</span>
          </button>

          {isLanguageOpen && (
            <div className={`absolute top-12 right-0 w-32 bg-white rounded-2xl shadow-xl border border-stone-100 p-2 overflow-hidden animate-in fade-in zoom-in duration-200`}>
              {languages.map((l) => (
                <button
                  key={l.code}
                  onClick={() => {
                    onLanguageChange(l.code as Lang);
                    setIsLanguageOpen(false);
                  }}
                  className={`w-full text-left px-3 py-2 text-sm font-medium rounded-xl transition-colors ${lang === l.code ? "bg-emerald-50 text-emerald-700" : "hover:bg-stone-50 text-stone-600"
                    }`}
                >
                  {l.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Notification Bell */}
        <button
          onClick={onOpenNotifications}
          className="relative p-2.5 bg-stone-50 hover:bg-stone-100 rounded-xl transition-colors border border-stone-100 group"
        >
          <Bell size={20} className="text-stone-500 group-hover:text-stone-700" />
          {unreadCount > 0 && (
            <span className="absolute top-1.5 right-1.5 w-4 h-4 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center border-2 border-white">
              {unreadCount}
            </span>
          )}
        </button>

        {/* User / Settings */}
        <button
          onClick={onOpenSettings}
          className="w-10 h-10 bg-gradient-to-tr from-amber-500 to-orange-400 rounded-xl flex items-center justify-center shadow-lg shadow-orange-100 border-2 border-white active:scale-95 transition-transform"
        >
          <Settings size={20} className="text-white" />
        </button>
      </div>
    </header>
  );
}
