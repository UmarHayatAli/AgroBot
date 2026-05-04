"use client";
import { X, MapPin, Wheat, Save, ChevronDown } from "lucide-react";
import { useState } from "react";
import { CITIES, CROPS, Lang, FarmContext } from "@/lib/mockData";
import { translations } from "@/lib/translations";

interface FarmSettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  context: FarmContext;
  onSave: (ctx: FarmContext) => void;
  lang: Lang;
}

function Dropdown({ value, options, onChange, isRtl }: { value: string, options: string[], onChange: (v: string) => void, isRtl: boolean }) {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full flex items-center justify-between border-2 border-stone-100 rounded-2xl px-4 py-3.5 text-stone-800 font-bold bg-stone-50/50 hover:bg-stone-50 hover:border-emerald-500/30 transition-all ${isRtl ? "flex-row-reverse urdu-text" : ""}`}
      >
        <span className="truncate">{value}</span>
        <ChevronDown size={18} className={`text-stone-400 transition-transform duration-300 ${isOpen ? "rotate-180" : ""}`} />
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-[60]" onClick={() => setIsOpen(false)} />
          <ul className={`absolute z-[70] mt-2 w-full bg-white rounded-2xl shadow-2xl border border-stone-100 p-2 max-h-60 overflow-y-auto animate-in fade-in slide-in-from-top-2 duration-200 ${isRtl ? "text-right urdu-text" : ""}`}>
            {options.map((opt) => (
              <li key={opt}>
                <button
                  onClick={() => {
                    onChange(opt);
                    setIsOpen(false);
                  }}
                  className={`w-full px-4 py-3 text-sm font-bold rounded-xl text-left transition-colors ${
                    value === opt ? "bg-emerald-50 text-emerald-700" : "hover:bg-stone-50 text-stone-600"
                  } ${isRtl ? "text-right" : ""}`}
                >
                  {opt}
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

export default function FarmSettingsModal({
  isOpen, onClose, context, onSave, lang,
}: FarmSettingsModalProps) {
  const t = translations[lang];
  const isRtl = lang === "ur" || lang === "pa" || lang === "skr";
  const [city, setCity] = useState(context.city);
  const [crop, setCrop] = useState(context.crop);

  if (!isOpen) return null;

  const handleSave = () => {
    onSave({ ...context, city, crop });
    onClose();
  };

  return (
    <>
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-end sm:items-center justify-center p-4">
        <div className="bg-white rounded-2xl w-full max-w-md shadow-2xl fade-up overflow-hidden">
          {/* Header */}
          <div className="bg-[#1a5c2e] px-6 py-5 flex items-center justify-between">
            <div className={isRtl ? "text-right" : ""}>
              <h2 className={`text-white font-bold text-lg ${isRtl ? "urdu-text" : ""}`}>
                {t.settings}
              </h2>
              <p className={`text-green-200 text-xs mt-0.5 ${isRtl ? "urdu-text" : ""}`}>
                {t.settingsSub}
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-full hover:bg-white/20 transition-colors text-white"
            >
              <X size={20} />
            </button>
          </div>

          <div className="p-6 space-y-5">
            {/* City Selector */}
            <div>
              <label className={`flex items-center gap-2 text-sm font-semibold text-stone-700 mb-2 ${isRtl ? "flex-row-reverse justify-end urdu-text text-base" : ""}`}>
                <MapPin size={15} className="text-green-700" />
                {t.city}
              </label>
              <Dropdown
                value={city}
                options={CITIES}
                onChange={setCity}
                isRtl={isRtl}
              />
            </div>

            {/* Crop Selector */}
            <div>
              <label className={`flex items-center gap-2 text-sm font-semibold text-stone-700 mb-2 ${isRtl ? "flex-row-reverse justify-end urdu-text text-base" : ""}`}>
                <Wheat size={15} className="text-green-700" />
                {t.crop}
              </label>
              <Dropdown
                value={crop}
                options={CROPS}
                onChange={setCrop}
                isRtl={isRtl}
              />
            </div>

            {/* Info box */}
            <div className={`bg-amber-50 border border-amber-200 rounded-xl p-3.5 ${isRtl ? "text-right" : ""}`}>
              <p className={`text-amber-800 text-xs leading-relaxed ${isRtl ? "urdu-text text-sm" : ""}`}>
                {t.settingsInfo}
              </p>
            </div>

            {/* Save Button */}
            <button
              onClick={handleSave}
              className={`w-full flex items-center justify-center gap-2.5 bg-[#1a5c2e] hover:bg-[#166534] active:scale-[0.98] text-white font-bold py-3.5 rounded-xl transition-all shadow-lg shadow-green-900/20 ${isRtl ? "flex-row-reverse urdu-text text-lg" : ""}`}
            >
              <Save size={17} />
              {t.save}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
