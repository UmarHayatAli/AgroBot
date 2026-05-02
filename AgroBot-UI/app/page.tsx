"use client";
import { useState } from "react";
import FarmContextHeader from "@/components/FarmContextHeader";
import ChatInterface from "@/components/ChatInterface";
import WeatherWarningDrawer from "@/components/WeatherWarningDrawer";
import WeatherSidebar from "@/components/WeatherSidebar";
import FarmSettingsModal from "@/components/FarmSettingsModal";
import { initialNotifications, Lang, FarmContext } from "@/lib/mockData";

export default function AgroBotDashboard() {
  const [lang, setLang] = useState<Lang>("en");
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [notifications, setNotifications] = useState(initialNotifications);
  const [farmContext, setFarmContext] = useState<FarmContext>({
    city: "Multan",
    crop: "Cotton — کپاس",
    tempC: 41.1,
    condition: "Clear sky",
  });

  const markAllRead = () =>
    setNotifications((prev) => prev.map((n) => ({ ...n, unread: false })));

  const handleContextChange = (location: string | null, crop: string | null) => {
    if (!location && !crop) return;
    setFarmContext((prev) => ({
      ...prev,
      city: location ? location : prev.city,
      crop: crop ? crop.charAt(0).toUpperCase() + crop.slice(1) : prev.crop,
    }));
  };

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-[#E8F5E9] via-[#C8EEE0] to-[#FFB87A] overflow-hidden">
      {/* Top Bar / Header */}
      <FarmContextHeader
        city={farmContext.city}
        crop={farmContext.crop.split(" — ")[0]}
        temp={Math.round(farmContext.tempC)}
        lang={lang}
        onLanguageChange={setLang}
        onOpenNotifications={() => setNotificationsOpen(true)}
        onOpenSettings={() => setSettingsOpen(true)}
        unreadCount={notifications.filter(n => n.unread).length}
      />

      <div className="flex-1 flex overflow-hidden w-full">
        {/* Main Chat Column */}
        <main className="flex-1 flex flex-col min-w-0 bg-white shadow-sm relative border-r border-stone-100">
          <ChatInterface lang={lang} onContextChange={handleContextChange} />
        </main>

        {/* Sidebar Weather Column */}
        <WeatherSidebar lang={lang} city={farmContext.city} />
      </div>

      {/* Overlays */}
      <WeatherWarningDrawer
        isOpen={notificationsOpen}
        onClose={() => setNotificationsOpen(false)}
        lang={lang}
      />

      <FarmSettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        context={farmContext}
        onSave={setFarmContext}
        lang={lang}
      />
    </div>
  );
}
