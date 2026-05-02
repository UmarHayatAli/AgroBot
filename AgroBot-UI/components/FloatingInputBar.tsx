"use client";
import { Mic, MicOff, Send, ImagePlus, X, Loader2 } from "lucide-react";
import { useRef, useState } from "react";
import { Lang } from "@/lib/mockData";
import { translations } from "@/lib/translations";

interface FloatingInputBarProps {
  lang: Lang;
  input: string;
  setInput: (val: string) => void;
  onSend: () => void;
  isLoading: boolean;
  isRecording: boolean;
  onToggleRecording: () => void;
  onImageUpload: (file: File) => void;
  previewImage: string | null;
  onClearImage: () => void;
  sttError: string | null;
  onClearSttError: () => void;
}

export default function FloatingInputBar({
  lang,
  input,
  setInput,
  onSend,
  isLoading,
  isRecording,
  onToggleRecording,
  onImageUpload,
  previewImage,
  onClearImage,
  sttError,
  onClearSttError
}: FloatingInputBarProps) {
  const t = translations[lang];
  const isRtl = lang === "ur" || lang === "pa" || lang === "skr";
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="absolute bottom-0 left-0 right-0 px-4 pb-8 pt-12 bg-gradient-to-t from-white via-white to-transparent pointer-events-none z-30">
      <div className="max-w-3xl mx-auto pointer-events-auto">
        {/* Alerts & Previews */}
        <div className="mb-4 space-y-2">
          {previewImage && (
            <div className="flex items-center gap-3 bg-white/90 backdrop-blur-xl border border-emerald-100 rounded-2xl p-2 shadow-xl shadow-emerald-900/5 animate-in slide-in-from-bottom-4 duration-300">
              <div className="relative">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={previewImage} alt="Preview" className="w-14 h-14 object-cover rounded-xl border-2 border-emerald-500/20" />
                <button
                  onClick={onClearImage}
                  className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 shadow-lg hover:bg-red-600 transition-all scale-90"
                >
                  <X size={12} />
                </button>
              </div>
              <div className="flex-1">
                <p className="text-stone-800 text-[11px] font-bold">Image ready for analysis</p>
                <p className="text-stone-400 text-[10px]">AgroBot will scan for pests and diseases</p>
              </div>
            </div>
          )}

          {sttError && (
            <div className="px-4 py-2 bg-red-50/90 backdrop-blur border border-red-100 rounded-xl text-red-700 text-xs flex items-center justify-between shadow-sm">
              <span>{sttError}</span>
              <button onClick={onClearSttError} className="p-1 hover:bg-red-100 rounded-full">
                <X size={12} />
              </button>
            </div>
          )}
        </div>

        {/* Main Input Pill */}
        <div className={`flex items-center gap-3 bg-[#FDFBF7] rounded-full border-2 border-[#3A9878]/20 focus-within:border-[#3A9878] focus-within:shadow-[0_0_20px_rgba(58,152,120,0.15)] shadow-2xl shadow-[#1B4332]/5 transition-all px-4 py-3 ${isRtl ? "flex-row-reverse" : ""}`}>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex-shrink-0 p-3.5 rounded-2xl text-[#3A9878] bg-[#C8EEE0]/50 hover:bg-[#C8EEE0] hover:scale-105 transition-all"
            title={t.uploadSoil}
          >
            <ImagePlus size={24} strokeWidth={2.5} />
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onImageUpload(file);
              e.target.value = "";
            }}
          />

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            dir={isRtl ? "rtl" : "ltr"}
            placeholder={t.placeholder}
            className={`flex-1 resize-none bg-transparent text-stone-900 placeholder-stone-500 font-medium text-base sm:text-lg border-none outline-none focus:outline-none focus-visible:outline-none ring-0 focus:ring-0 py-3.5 max-h-40 min-h-[52px] self-center ${isRtl ? "urdu-text" : ""}`}
            style={{ lineHeight: "1.2" }}
          />

          <div className="flex items-center gap-2.5">
            <button
              onClick={onToggleRecording}
              className={`flex-shrink-0 p-3.5 rounded-2xl transition-all hover:scale-105 ${
                isRecording
                  ? "bg-red-500 text-white shadow-lg shadow-red-200 mic-recording"
                  : "text-[#3A9878] bg-[#C8EEE0]/50 hover:bg-[#C8EEE0]"
              }`}
            >
              {isRecording ? <MicOff size={24} strokeWidth={2.5} /> : <Mic size={24} strokeWidth={2.5} />}
            </button>

            <button
              onClick={onSend}
              disabled={isLoading || (!input.trim() && !previewImage)}
              className="flex-shrink-0 p-4 rounded-2xl bg-gradient-to-br from-[#3A9878] to-[#1B4332] text-white hover:from-[#2D7A60] hover:to-[#1B4332] disabled:opacity-20 disabled:scale-95 transition-all shadow-xl shadow-[#3A9878]/20 active:scale-90 flex items-center justify-center border border-[#3A9878]/20"
            >
              {isLoading ? <Loader2 size={24} className="animate-spin" /> : <Send size={24} strokeWidth={2.5} />}
            </button>
          </div>
        </div>

        {/* Footer Hint */}
        <div className="flex items-center justify-center gap-2.5 mt-4 opacity-80 hover:opacity-100 transition-opacity">
          <div className="w-1.5 h-1.5 rounded-full bg-[#3A9878] animate-pulse shadow-[0_0_8px_rgba(58,152,120,0.5)]"></div>
          <p className="text-[10px] font-black text-[#1B4332] uppercase tracking-[0.15em] drop-shadow-sm">
            {isRecording ? "Listening..." : t.hint}
          </p>
        </div>
      </div>
    </div>
  );
}
