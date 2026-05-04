"use client";
import {
  Mic, MicOff, Send, ImagePlus, X, Loader2, CloudRain, FlaskConical, ScanSearch, Map, Calendar, Leaf, CloudSun, Wind, Droplets, Thermometer
} from "lucide-react";
import { useState, useRef, useEffect, useCallback } from "react";
import { Message, Lang, initialMessages } from "@/lib/mockData";
import { translations } from "@/lib/translations";
import MessageBubble from "./MessageBubble";
import FloatingInputBar from "./FloatingInputBar";

// ── Language config for Web APIs ─────────────────────────────────────────────
const STT_LANG: Record<Lang, string> = {
  en: "en-US",
  ur: "ur-PK",
  pa: "pa-PK",
  skr: "ur-PK", // Saraiki closest to Urdu for STT
};

// TTS: find the best available system voice for each language
function getBestVoice(lang: Lang): SpeechSynthesisVoice | null {
  if (typeof window === "undefined") return null;
  const voices = window.speechSynthesis.getVoices();
  const prefs: Record<Lang, string[]> = {
    en: ["en-US", "en-GB", "en"],
    ur: ["ur-PK", "ur", "ar-SA", "ar"],
    pa: ["pa-PK", "pa-IN", "pa", "ur-PK", "ur"],
    skr: ["ur-PK", "ur", "ar"],
  };
  for (const code of prefs[lang]) {
    const v = voices.find((v) => v.lang.startsWith(code));
    if (v) return v;
  }
  return voices[0] ?? null; // last resort: any voice
}

// TTS logic moved inside component for state management


// ── Component ─────────────────────────────────────────────────────────────────
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface ChatInterfaceProps {
  lang: Lang;
  onContextChange?: (location: string | null, crop: string | null) => void;
}

export default function ChatInterface({ lang, onContextChange }: ChatInterfaceProps) {
  const t = translations[lang];
  const isRtl = lang === "ur" || lang === "pa" || lang === "skr";
  const isNonEn = lang !== "en";

  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [sttError, setSttError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string>(""); // ← memory key

  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recognitionRef = useRef<any>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const currentlyPlayingTextRef = useRef<string | null>(null);

  // ── TTS Logic ─────────────────────────────────────────────────────────────
  const handleSpeak = useCallback(async (text: string, lang: Lang) => {
    const clean = text.replace(/\*\*/g, "").replace(/•/g, "").trim();
    if (!clean) return;

    // TOGGLE: If same text is playing, stop it
    if (currentAudioRef.current && currentlyPlayingTextRef.current === clean) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
      currentlyPlayingTextRef.current = null;
      window.speechSynthesis.cancel();
      return;
    }

    // Stop existing playback (either HTML audio or browser TTS)
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    window.speechSynthesis.cancel();

    try {
      // INSTANT PLAYBACK: Set src directly to streaming URL to avoid fetch delay
      const url = `${API_BASE_URL}/api/tts?text=${encodeURIComponent(clean)}&lang=${lang}`;
      const audio = new Audio(url);

      currentAudioRef.current = audio;
      currentlyPlayingTextRef.current = clean;

      audio.onended = () => {
        if (currentlyPlayingTextRef.current === clean) {
          currentlyPlayingTextRef.current = null;
          currentAudioRef.current = null;
        }
      };

      await audio.play();
    } catch (err) {
      console.error("Backend TTS failed, falling back to browser TTS", err);
      if (typeof window === "undefined" || !("speechSynthesis" in window)) return;

      const doSpeak = () => {
        const utterance = new SpeechSynthesisUtterance(clean);
        utterance.lang = STT_LANG[lang];
        utterance.rate = lang === "en" ? 1 : 0.85;
        utterance.pitch = 1;
        const voice = getBestVoice(lang);
        if (voice) utterance.voice = voice;

        currentlyPlayingTextRef.current = clean;
        utterance.onend = () => {
          if (currentlyPlayingTextRef.current === clean) {
            currentlyPlayingTextRef.current = null;
          }
        };

        window.speechSynthesis.speak(utterance);
      };

      if (window.speechSynthesis.getVoices().length === 0) {
        window.speechSynthesis.onvoiceschanged = () => { doSpeak(); };
      } else {
        doSpeak();
      }
    }
  }, [lang]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── STT via Backend ───────────────────────────────────────────────────────
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecording(false);
  }, []);

  // Stop recording when language changes
  useEffect(() => {
    if (isRecording) stopRecording();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lang]);

  const startRecording = useCallback(async () => {
    setSttError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        setIsLoading(true);

        const formData = new FormData();
        formData.append("file", audioBlob, "recording.webm");
        formData.append("lang", lang);

        try {
          const res = await fetch(`${API_BASE_URL}/api/stt`, {
            method: "POST",
            body: formData,
          });
          const data = await res.json();
          if (data.text) {
            setInput(data.text);
          } else {
            setSttError(t.sttError);
          }
        } catch (err) {
          setSttError(t.backendError);
        } finally {
          setIsLoading(false);
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      setSttError(isNonEn ? "مائیکروفون تک رسائی نہیں" : "Microphone access denied");
      setIsRecording(false);
    }
  }, [lang, isNonEn]);

  const toggleRecording = useCallback(() => {
    if (isRecording) { stopRecording(); } else { startRecording(); }
  }, [isRecording, startRecording, stopRecording]);

  // ── Send message ──────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (text: string, imgUrl?: string) => {
    if (!text.trim() && !imgUrl) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      text: text.trim() || (isNonEn ? t.attachImage : "Image uploaded"),
      timestamp: new Date(),
      imageUrl: imgUrl,
    };
    const typingMsg: Message = { id: "typing", role: "typing", text: "", timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg, typingMsg]);
    setInput("");
    setPreviewImage(null);
    setIsLoading(true);

    try {
      // Create form data for the real backend
      const formData = new FormData();
      formData.append("query", text);
      formData.append("lang", lang);
      formData.append("session_id", sessionId); // ← send session for memory
      if (imageFile) {
        console.log("Appending file to formData:", imageFile.name);
        formData.append("file", imageFile);
      }

      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      let botText = "Error: Invalid response from backend.";
      let intent = "general";
      if (data && data.result) {
        botText = data.result.text || "No response text.";
        intent = data.result.intent || "general";
        // Store session_id so next message uses same memory
        if (data.session_id && !sessionId) setSessionId(data.session_id);
      } else if (data && data.error) {
        botText = `Error: ${data.error}`;
      }

      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        text: botText,
        intent: intent,
        timestamp: new Date(),
      };

      setMessages((prev) => prev.filter((m) => m.id !== "typing").concat(botMsg));
    } catch (err) {
      console.error(err);
      const errorMsg: Message = {
        id: (Date.now() + 1).toString(),
        role: "bot",
        text: t.genericError,
        intent: "general",
        timestamp: new Date(),
      };
      setMessages((prev) => prev.filter((m) => m.id !== "typing").concat(errorMsg));
    } finally {
      setImageFile(null);
      setIsLoading(false);
    }
  }, [lang, isNonEn, imageFile, onContextChange]);

  const handleSend = useCallback(() => {
    sendMessage(input, previewImage ?? undefined);
  }, [sendMessage, input, previewImage]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    console.log("Selected file:", file.name, file.type, file.size);
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (ev) => {
      setPreviewImage(ev.target?.result as string);
    };
    reader.readAsDataURL(file);
    // Reset the input so the same file can be uploaded again if cleared
    e.target.value = "";
  };

  // ── Placeholder text ──────────────────────────────────────────────────────
  const placeholder = t.placeholder;
  const hintText = t.hint;

  const micTooltip =
    isRecording
      ? t.stopRecording
      : `${t.speak} (${STT_LANG[lang]})`;

  return (
    <div className="flex flex-col h-full relative bg-[#F1F8F6]">
      {/* Top Action Cards (Redesigned) */}
      <div className="flex-shrink-0 grid grid-cols-3 gap-4 p-5 bg-gradient-to-b from-[#E8F5E9]/80 to-[#F1F8F6] border-b border-[#3A9878]/10">
        <button className="h-[140px] relative overflow-hidden flex flex-col items-center justify-center gap-3 p-5 bg-[#F1F8F6] rounded-[28px] border border-[#3A9878]/20 shadow-sm hover:shadow-xl hover:-translate-y-1 hover:border-[#3A9878] transition-all group">
          <div className="absolute -top-6 -right-6 w-16 h-16 bg-[#C8EEE0]/30 rounded-full opacity-50 group-hover:scale-150 transition-transform duration-500" />
          <div className="w-14 h-14 rounded-2xl bg-[#3A9878] text-white flex items-center justify-center shadow-lg shadow-[#3A9878]/20 relative z-10 group-hover:bg-[#2D7A60] transition-colors">
            <div className="relative">
              <ScanSearch size={28} />
              <Leaf size={14} className="absolute -bottom-1 -right-1 text-[#C8EEE0] bg-[#1B4332] rounded-full p-0.5" />
            </div>
          </div>
          <span className={`text-[13px] font-black text-stone-700 text-center leading-tight relative z-10 ${isNonEn ? "urdu-text" : ""}`}>
            {t.scanDisease}
          </span>
        </button>

        <button className="h-[140px] relative overflow-hidden flex flex-col items-center justify-center gap-3 p-5 bg-[#FFF9F2] rounded-[28px] border border-[#FFB87A]/20 shadow-sm hover:shadow-xl hover:-translate-y-1 hover:border-[#FFB87A] transition-all group">
          <div className="absolute -top-6 -right-6 w-16 h-16 bg-[#FFB87A]/10 rounded-full opacity-50 group-hover:scale-150 transition-transform duration-500" />
          <div className="w-14 h-14 rounded-2xl bg-[#FFB87A] text-white flex items-center justify-center shadow-lg shadow-[#FFB87A]/20 relative z-10 group-hover:bg-[#FF9F4D] transition-colors">
            <FlaskConical size={28} />
          </div>
          <span className={`text-[13px] font-black text-stone-700 text-center leading-tight relative z-10 ${isNonEn ? "urdu-text" : ""}`}>
            {t.uploadSoil}
          </span>
        </button>

        <button className="h-[140px] relative overflow-hidden flex flex-col items-center justify-center gap-3 p-5 bg-[#E8F5E9]/50 rounded-[28px] border border-[#C8EEE0] shadow-sm hover:shadow-xl hover:-translate-y-1 hover:border-[#3A9878]/30 transition-all group">
          <div className="absolute -top-6 -right-6 w-16 h-16 bg-[#E8F5E9] rounded-full opacity-50 group-hover:scale-150 transition-transform duration-500" />
          <div className="w-14 h-14 rounded-2xl bg-[#C8EEE0] text-[#1B4332] flex items-center justify-center shadow-lg shadow-[#C8EEE0]/20 relative z-10 group-hover:bg-[#A8E6CF] transition-colors">
            <div className="relative group-hover:scale-110 transition-transform duration-300">
              <CloudSun size={32} strokeWidth={2.5} className="text-[#3A9878]" />
              <Thermometer size={14} className="absolute -bottom-1 -right-1 text-white bg-[#3A9878] rounded-full p-0.5 shadow-sm" />
            </div>
          </div>
          <span className={`text-[13px] font-black text-stone-700 text-center leading-tight relative z-10 ${isNonEn ? "urdu-text" : ""}`}>
            {t.viewWeather}
          </span>
        </button>
      </div>

      {/* Chat feed */}
      <div className="flex-1 overflow-y-auto chat-scroll px-6 pt-6 pb-40 space-y-6 bg-[radial-gradient(#FFB87A_1px,transparent_1px)] [background-size:24px:24px] bg-[#F1F8F6]">
        <div className="flex justify-center">
          <span className="px-3 py-1 bg-stone-100 text-[10px] font-black text-stone-500 uppercase tracking-widest rounded-full border border-stone-200">
            Today, 10:42 AM
          </span>
        </div>
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} lang={lang} onSpeak={handleSpeak} />
        ))}

        <div ref={chatEndRef} />
      </div>

      {/* Input Area (Floating) */}
      <FloatingInputBar
        lang={lang}
        input={input}
        setInput={setInput}
        onSend={handleSend}
        isLoading={isLoading}
        isRecording={isRecording}
        onToggleRecording={toggleRecording}
        onImageUpload={(file) => {
          setImageFile(file);
          const reader = new FileReader();
          reader.onload = (ev) => setPreviewImage(ev.target?.result as string);
          reader.readAsDataURL(file);
        }}
        previewImage={previewImage}
        onClearImage={() => {
          setPreviewImage(null);
          setImageFile(null);
        }}
        sttError={sttError}
        onClearSttError={() => setSttError(null)}
      />
    </div>
  );
}
