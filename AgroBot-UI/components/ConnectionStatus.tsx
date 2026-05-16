"use client";
import { Wifi, WifiOff, RefreshCw } from "lucide-react";
import { useState, useEffect } from "react";

import { Lang } from "@/lib/mockData";
import { translations } from "@/lib/translations";

interface ConnectionStatusProps {
  lang: Lang;
}

export default function ConnectionStatus({ lang }: ConnectionStatusProps) {
  const t = translations[lang];
  const isNonLatin = lang !== "en";
  const [isOnline, setIsOnline] = useState(true);
  const [syncing, setSyncing] = useState(false);

  useEffect(() => {

    
    const checkConnection = async () => {
      setIsOnline(window.navigator.onLine);
    };

    checkConnection();
    const interval = setInterval(checkConnection, 10000); // Check every 10s

    const handleOnline = () => checkConnection();
    const handleOffline = () => setIsOnline(false);
    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);
    
    return () => {
      clearInterval(interval);
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);

  const handleSync = async () => {
    if (!window.navigator.onLine) return;
    setSyncing(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 500));
      setIsOnline(true);
    } catch {
      setIsOnline(false);
    } finally {
      setTimeout(() => setSyncing(false), 500);
    }
  };

  if (isOnline) {
    return (
      <button
        onClick={handleSync}
        title={t.statusOnline}
        className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-green-100 border border-green-300 text-green-800 text-xs font-medium hover:bg-green-200 transition-colors"
      >
        {syncing ? (
          <RefreshCw size={11} className="animate-spin text-green-600" />
        ) : (
          <Wifi size={11} className="text-green-600" />
        )}
        <span className={`hidden sm:inline ${isNonLatin ? "urdu-text" : ""}`}>
          {t.statusOnline}
        </span>
      </button>
    );
  }

  return (
    <button
      title={t.statusOffline}
      className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-red-100 border border-red-300 text-red-800 text-xs font-medium animate-pulse"
    >
      <WifiOff size={11} className="text-red-600" />
      <span className={`hidden sm:inline ${isNonLatin ? "urdu-text" : ""}`}>
        {t.statusOffline}
      </span>
    </button>
  );
}
