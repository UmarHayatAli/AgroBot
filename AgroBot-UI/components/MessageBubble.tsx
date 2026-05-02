"use client";
import { Volume2, User, Bot, Leaf, Droplets, CloudRain, AlertTriangle } from "lucide-react";
import { useState, useEffect } from "react";
import { Message, Lang } from "@/lib/mockData";
import { translations } from "@/lib/translations";

interface MessageBubbleProps {
  message: Message;
  lang: Lang;
  onSpeak: (text: string, lang: Lang) => void;
}

const intentConfig = {
  weather: { icon: <CloudRain size={13} />, color: "text-orange-700 bg-orange-100", bubbleBg: "bg-[#FFF9F2]", label: "Weather", labelUr: "موسم" },
  disease: { icon: <Leaf size={13} />, color: "text-red-700 bg-red-100", bubbleBg: "bg-[#FDFBF7]", label: "Disease", labelUr: "بیماری" },
  soil: { icon: <Droplets size={13} />, color: "text-amber-700 bg-amber-100", bubbleBg: "bg-[#FDFBF7]", label: "Soil", labelUr: "مٹی" },
  flood: { icon: <AlertTriangle size={13} />, color: "text-orange-700 bg-orange-100", bubbleBg: "bg-[#FDFBF7]", label: "Flood", labelUr: "سیلاب" },
  general: { icon: <Bot size={13} />, color: "text-[#3A9878] bg-[#C8EEE0]/30", bubbleBg: "bg-[#FDFBF7]", label: "General", labelUr: "عمومی" },
};

function formatText(text: string): React.ReactNode[] {
  const lines = text.split("\n");
  const parsedLines: React.ReactNode[] = [];

  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    let isListItem = false;

    // Detect list items
    if (/^[\-\*]\s/.test(line)) {
      isListItem = true;
      line = line.replace(/^[\-\*]\s/, "");
    } else if (/^\d+\.\s/.test(line)) {
      isListItem = true;
    }

    // Process bolding
    const parts = line.split(/(\*\*.*?\*\*)/g);
    const formattedLine = parts.map((part, j) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={j} className="font-bold">{part.slice(2, -2)}</strong>;
      }
      return part;
    });

    if (isListItem) {
      parsedLines.push(
        <div key={i} className="flex items-start gap-2 my-1">
          <span className="text-emerald-500 mt-1 flex-shrink-0">•</span>
          <span>{formattedLine}</span>
        </div>
      );
    } else {
      parsedLines.push(
        <span key={i} className="block mb-2 last:mb-0">
          {formattedLine}
        </span>
      );
    }
  }

  return parsedLines;
}

export default function MessageBubble({ message, lang, onSpeak }: MessageBubbleProps) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const t = translations[lang];
  const isRtl = lang === "ur" || lang === "pa" || lang === "skr";
  const isUrdu = lang !== "en";
  const isUser = message.role === "user";
  const isTyping = message.role === "typing";
  const displayText =
    lang === "ur" ? (message.textUr || message.text) :
      lang === "pa" ? (message.textPa || message.textUr || message.text) :
        lang === "skr" ? (message.textSkr || message.textUr || message.text) :
          message.text;
  const intent = message.intent ? intentConfig[message.intent] : null;
  const time = mounted ? message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: true }) : "";

  if (isTyping) {
    return (
      <div className="flex items-end gap-3 fade-up">
        <div className="flex-shrink-0 w-10 h-10 rounded-full overflow-hidden border-2 border-emerald-500/20 shadow-md">
          <img src="/mascot.png" alt="AgroBot" className="w-full h-full object-cover" />
        </div>
        <div className="bg-[#FDFBF7] rounded-2xl rounded-bl-sm px-5 py-3 shadow-sm border border-[#3A9878]/10">
          <div className="flex items-center gap-1.5 h-6">
            {[0, 1, 2].map((i) => (
              <div key={i} className="typing-dot w-2 h-2 rounded-full bg-emerald-400" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex items-end gap-3 fade-up ${isUser ? "flex-row-reverse" : ""}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center shadow-md text-white border-2 overflow-hidden ${isUser
            ? "bg-[#FFB87A] border-[#FFB87A]/50"
            : "bg-white border-[#3A9878]/20"
          }`}
      >
        {isUser ? (
          <span className="text-[10px] font-bold">AK</span>
        ) : (
          <img src="/mascot.png" alt="AgroBot" className="w-full h-full object-cover" />
        )}
      </div>

      {/* Bubble */}
      <div className={`max-w-[78%] flex flex-col gap-1 ${isUser ? "items-end" : "items-start"}`}>
        {/* Intent tag */}
        {!isUser && intent && (
          <span className={`flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full w-fit ${intent.color}`}>
            {intent.icon}
            {isUrdu ? intent.labelUr : intent.label}
          </span>
        )}

        {/* Image preview */}
        {message.imageUrl && (
          <div className="rounded-xl overflow-hidden border-2 border-green-200 shadow-sm mb-1">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={message.imageUrl} alt="Uploaded crop" className="max-w-full max-h-48 object-cover" />
          </div>
        )}

        {/* Message body */}
        <div
          className={`relative rounded-[24px] px-6 py-4 shadow-md border ${isUser
              ? "bg-gradient-to-br from-[#3A9878] to-[#1B4332] text-white rounded-br-none border-[#1B4332]/20"
              : `${intent?.bubbleBg || "bg-[#FDFBF7]"} text-stone-800 rounded-bl-none border-[#3A9878]/10 shadow-stone-200/20`
            }`}
        >
          <div
            dir={isRtl ? "rtl" : "ltr"}
            className={`text-sm leading-relaxed whitespace-pre-line ${isUrdu ? "urdu-text text-base" : ""
              }`}
          >
            {formatText(displayText)}
          </div>

          {/* TTS Button for bot messages - KEEPING PREMIUM DESIGN */}
          {!isUser && (
            <div className="flex justify-center w-full mt-4">
              <button
                onClick={() => onSpeak(displayText, lang)}
                title={t.listenAudio}
                className="flex items-center gap-2.5 text-xs text-[#1B4332] bg-[#C8EEE0]/80 backdrop-blur-sm px-4 py-2 rounded-xl hover:bg-[#C8EEE0] font-black transition-all group/btn w-fit shadow-sm border border-[#C8EEE0]/50 hover:shadow-md hover:-translate-y-0.5 active:translate-y-0"
              >
                <div className="w-6 h-6 rounded-lg bg-[#3A9878] text-white flex items-center justify-center shadow-md group-hover/btn:scale-110 transition-transform">
                  <Volume2 size={14} />
                </div>
                <span className={isUrdu ? "urdu-text text-base leading-none pt-1" : "tracking-tight"}>
                  {t.listenAudio}
                </span>
              </button>
            </div>
          )}
        </div>

        {/* Timestamp */}
        <span className={`text-[10px] text-stone-400 px-1 ${isUser ? "text-right self-end" : ""}`}>
          {time}
        </span>
      </div>
    </div>
  );
}
