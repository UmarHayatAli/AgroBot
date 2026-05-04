"use client";
import { X, AlertTriangle, Flame, CheckCircle, Info, Bell } from "lucide-react";
import { Notification, Lang, Severity } from "@/lib/mockData";
import { translations } from "@/lib/translations";

interface NotificationDrawerProps {
  notifications: Notification[];
  isOpen: boolean;
  onClose: () => void;
  lang: Lang;
  onMarkAllRead: () => void;
}

const severityConfig: Record<Severity, { icon: React.ReactNode; bgClass: string; key: string }> = {
  critical: {
    icon: <Flame size={16} className="text-red-600" />,
    bgClass: "alert-critical",
    key: "critical",
  },
  warning: {
    icon: <AlertTriangle size={16} className="text-amber-600" />,
    bgClass: "alert-warning",
    key: "warning",
  },
  info: {
    icon: <Info size={16} className="text-blue-600" />,
    bgClass: "alert-info",
    key: "info",
  },
  ok: {
    icon: <CheckCircle size={16} className="text-green-600" />,
    bgClass: "alert-ok",
    key: "ok",
  },
};

export default function NotificationDrawer({
  notifications, isOpen, onClose, lang, onMarkAllRead,
}: NotificationDrawerProps) {
  const t = translations[lang];
  const isRtl = lang === "ur" || lang === "pa" || lang === "skr";
  const unreadCount = notifications.filter((n) => n.unread).length;

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
        onClick={onClose}
      />

      {/* Drawer */}
      <aside className={`fixed top-0 ${isRtl ? "left-0 drawer-enter-left" : "right-0 drawer-enter-right"} h-full w-full max-w-sm bg-[#faf7f2] shadow-2xl z-50 flex flex-col`}>
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 bg-[#1a5c2e] text-white flex-shrink-0">
          <div className={`flex items-center gap-2 ${isRtl ? "flex-row-reverse" : ""}`}>
            <Bell size={20} />
            <div className={isRtl ? "text-right" : ""}>
              <h2 className={`font-bold text-base leading-tight ${isRtl ? "urdu-text" : ""}`}>
                {t.notifications}
              </h2>
              {unreadCount > 0 && (
                <p className={`text-green-200 text-sm ${isRtl ? "urdu-text" : ""}`}>
                  {unreadCount} {t.newNotifications}
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-3">
            {unreadCount > 0 && (
              <button
                onClick={onMarkAllRead}
                className={`text-green-200 text-xs hover:text-white transition-colors ${isRtl ? "urdu-text" : ""}`}
              >
                {t.markAllRead}
              </button>
            )}
            <button
              onClick={onClose}
              className="p-1.5 rounded-full hover:bg-white/20 transition-colors"
              aria-label="Close"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Subtitle */}
        <div className="px-5 py-2.5 bg-green-50 border-b border-green-200">
          <p className={`text-green-800 text-sm ${isRtl ? "urdu-text" : ""}`}>
            {t.notificationsSub}
          </p>
        </div>

        {/* Notification List */}
        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
          {notifications.map((notif) => {
            const config = severityConfig[notif.severity];
            return (
              <div
                key={notif.id}
                className={`${config.bgClass} rounded-xl p-4 relative fade-up ${
                  notif.unread ? "ring-2 ring-offset-1 ring-current" : "opacity-80"
                }`}
              >
                {notif.unread && (
                  <span className="absolute top-3 right-3 w-2.5 h-2.5 rounded-full bg-green-500 ring-2 ring-white" />
                )}
                <div className={`flex items-start gap-2.5 ${isRtl ? "flex-row-reverse" : ""}`}>
                  <div className="flex-shrink-0 mt-0.5">{config.icon}</div>
                  <div className={`flex-1 ${isRtl ? "text-right" : ""}`}>
                    <div className={`flex items-center gap-2 mb-1 ${isRtl ? "flex-row-reverse justify-end" : ""}`}>
                      <span className="text-xs font-bold tracking-wider opacity-60">
                        {lang === "en" ? config.key.toUpperCase() : (lang === "ur" ? "انتباہ" : "اطلاع")}
                      </span>
                      <span className="text-xs opacity-50">• {notif.time}</span>
                    </div>
                    <h4 className={`text-sm font-bold text-stone-900 ${isRtl ? "urdu-text" : ""}`}>
                      {lang === "ur" ? notif.titleUr : lang === "pa" ? notif.titlePa : lang === "skr" ? notif.titleSkr : notif.title}
                    </h4>
                    <p className={`text-sm text-stone-600 mt-0.5 leading-relaxed ${isRtl ? "urdu-text" : ""}`}>
                      {lang === "ur" ? notif.bodyUr : lang === "pa" ? notif.bodyPa : lang === "skr" ? notif.bodySkr : notif.body}
                    </p>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-stone-200 flex-shrink-0">
          <p className={`text-center text-xs text-stone-400 ${isRtl ? "urdu-text" : ""}`}>
            {t.systemName}
          </p>
        </div>
      </aside>
    </>
  );
}
