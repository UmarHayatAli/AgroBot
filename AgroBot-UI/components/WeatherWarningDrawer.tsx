import { X, CloudRain, ThermometerSun, Wind, AlertTriangle } from "lucide-react";
import { useEffect, useState } from "react";
import ProtectionPlanDialog from "./ProtectionPlanDialog";
import { translations } from "@/lib/translations";

interface Alert {
  id: string;
  title: string;
  titleUr: string;
  titlePa: string;
  titleSkr: string;
  time: string;
  timeUr: string;
  timePa: string;
  timeSkr: string;
  severity: "critical" | "warning" | "info";
  type: "heat" | "rain" | "wind" | "general";
  planMarkdown: string;
  planMarkdownUr: string;
  planMarkdownPa: string;
  planMarkdownSkr: string;
}

interface WeatherWarningDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  lang: "en" | "ur" | "pa" | "skr";
}

const mockAlerts: Alert[] = [
  {
    id: "1",
    title: "Extreme Heat Expected",
    titleUr: "شدید گرمی کی توقع",
    titlePa: "شدید گرمی دی توقع",
    titleSkr: "شدید گرمی دی توقع",
    time: "Tomorrow, 2:00 PM",
    timeUr: "کل، دوپہر ۲:۰۰ بجے",
    timePa: "کل، دوپہر ۲:۰۰ بجے",
    timeSkr: "کل، دوپہر ۲:۰۰ بجے",
    severity: "critical",
    type: "heat",
    planMarkdown: "## Heatwave Alert\n\nTemperatures will reach 42°C tomorrow. This poses a severe risk of heat stress for your cotton crop.\n\n### Immediate Actions\n- **Irrigate Immediately:** Water your fields tonight or before 7 AM tomorrow.\n- **Delay Fertilizers:** Do not apply nitrogen fertilizers during peak heat.\n- **Monitor for Wilting:** Check plants for drooping leaves in the late afternoon.",
    planMarkdownUr: "## ہیٹ ویو الرٹ\n\nکل درجہ حرارت ۴۲ ڈگری تک پہنچ جائے گا۔ یہ آپ کی کپاس کی فصل کے لیے شدید خطرہ ہے۔\n\n### فوری اقدامات\n- **فوری آبپاشی:** اپنے کھیتوں کو آج رات یا کل صبح ۷ بجے سے پہلے پانی دیں۔\n- **کھاد میں تاخیر:** شدید گرمی کے دوران نائٹروجن کھاد نہ ڈالیں۔\n- **مرجھانے کی نگرانی:** سہ پہر میں پودوں کے پتوں کے جھکنے کی جانچ کریں۔",
    planMarkdownPa: "## ہیٹ ویو الرٹ\n\nکل درجہ حرارت ۴۲ ڈگری تیکر پہنچ جائے گا۔ اے تہاڈی کپاس دی فصل لئی شدید خطرہ اے۔\n\n### فوری اقدامات\n- **فوری آبپاشی:** اپنے کھیتاں نوں اج رات یا کل سویرے ۷ بجے توں پہلاں پانی دیو۔\n- **کھاد وچ تاخیر:** شدید گرمی دے دوران نائٹروجن کھاد نہ پاؤ۔\n- **مرجھان دی نگرانی:** سہ پہر وچ بوٹیاں دے پتیاں دے جھکن دی جانچ کرو۔",
    planMarkdownSkr: "## ہیٹ ویو الرٹ\n\nکل درجہ حرارت ۴۲ ڈگری تئیں پہنچ ویسی۔ اے تہاڈی کپاس دی فصل سانگے شدید خطرہ ہے۔\n\n### فوری اقدامات\n- **فوری آبپاشی:** اپنے کھیتاں کوں اج رات یا کل سویرے ۷ بجے توں پہلے پاݨی ڈیو۔\n- **کھاد وچ تاخیر:** شدید گرمی دے دوران نائٹروجن کھاد نہ پاؤ۔\n- **مرجھاوݨ دی نگرانی:** سہ پہر وچ بوٹیاں دے پتیاں دے جھکݨ دی جانچ کرو۔",
  },
  {
    id: "2",
    title: "Light Rain Showers",
    titleUr: "ہلکی بارش",
    titlePa: "ہلکی بارش",
    titleSkr: "ہلکی بارش",
    time: "Friday, 6:00 AM",
    timeUr: "جمعہ، صبح ۶:۰۰ بجے",
    timePa: "جمعہ، سویرے ۶:۰۰ بجے",
    timeSkr: "جمعہ، سویرے ۶:۰۰ بجے",
    severity: "info",
    type: "rain",
    planMarkdown: "## Rain Advisory\n\nLight rain is expected (approx 5mm).\n\n### Actions\n- This is highly beneficial for the current growth stage.\n- No irrigation is required for the next 2 days.",
    planMarkdownUr: "## بارش کی ایڈوائزری\n\nہلکی بارش متوقع ہے (تقریباً ۵ ملی میٹر)۔\n\n### اقدامات\n- یہ موجودہ ترقیاتی مرحلے کے لیے انتہائی فائدہ مند ہے۔\n- اگلے ۲ دنوں تک کسی آبپاشی کی ضرورت نہیں ہے۔",
    planMarkdownPa: "## بارش دی ایڈوائزری\n\nہلکی بارش متوقع اے (تقریباً ۵ ملی میٹر)۔\n\n### اقدامات\n- اے موجودہ ترقیاتی مرحلے لئی انتہائی فائدہ مند اے۔\n- اگلے ۲ دناں تیکر کسی آبپاشی دی لوڑ نہیں۔",
    planMarkdownSkr: "## بارش دی ایڈوائزری\n\nہلکی بارش متوقع ہے (تقریباً ۵ ملی میٹر)۔\n\n### اقدامات\n- اے موجودہ ترقیاتی مرحلے سانگے انتہائی فائدہ مند ہے۔\n- اگلے ۲ ڈیہانہہ تئیں کہیں آبپاشی دی لوڑ کائنی۔",
  }
];

export default function WeatherWarningDrawer({ isOpen, onClose, lang }: WeatherWarningDrawerProps) {
  const t = translations[lang];
  const isRtl = lang === "ur" || lang === "pa" || lang === "skr";
  const isNonLatin = lang !== "en";
  
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  useEffect(() => {
    if (isOpen) document.body.style.overflow = "hidden";
    else document.body.style.overflow = "unset";
    return () => { document.body.style.overflow = "unset"; };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <>
      <div 
        className="fixed inset-0 bg-stone-900/40 backdrop-blur-sm z-40 transition-opacity"
        onClick={onClose}
      />
      
      <div 
        className={`fixed top-0 ${isRtl ? "left-0 drawer-enter-left" : "right-0 drawer-enter-right"} h-full w-full max-w-sm bg-[#fcfaf8] shadow-2xl z-50 flex flex-col`}
        dir={isRtl ? "rtl" : "ltr"}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-stone-200 bg-white">
          <div className={`flex items-center gap-2 ${isRtl ? "flex-row-reverse" : ""}`}>
            <AlertTriangle className="text-orange-500" size={20} />
            <h2 className={`font-black text-lg text-stone-800 ${isNonLatin ? "urdu-text" : ""}`}>
              {t.weatherAlert}
            </h2>
          </div>
          <button onClick={onClose} className="p-2 rounded-full hover:bg-stone-100 text-stone-500">
            <X size={20} />
          </button>
        </div>

        {/* Alerts List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {mockAlerts.map(alert => (
            <button 
              key={alert.id}
              onClick={() => setSelectedAlert(alert)}
              className={`w-full text-left rounded-2xl p-4 border transition-all active:scale-95 shadow-sm hover:shadow-md
                ${alert.severity === "critical" ? "bg-red-50 border-red-200" : 
                  alert.severity === "warning" ? "bg-orange-50 border-orange-200" : 
                  "bg-blue-50 border-blue-200"}
              `}
            >
              <div className={`flex items-start gap-3 ${isRtl ? "flex-row-reverse text-right" : ""}`}>
                <div className={`p-2 rounded-full flex-shrink-0
                  ${alert.severity === "critical" ? "bg-red-100 text-red-600" : 
                    alert.severity === "warning" ? "bg-orange-100 text-orange-600" : 
                    "bg-blue-100 text-blue-600"}
                `}>
                  {alert.type === "heat" ? <ThermometerSun size={20} /> : 
                   alert.type === "rain" ? <CloudRain size={20} /> : <Wind size={20} />}
                </div>
                <div>
                  <h3 className={`font-bold text-stone-900 group-hover:text-emerald-700 transition-colors ${isNonLatin ? "urdu-text" : ""}`}>
                    {lang === "ur" ? alert.titleUr : lang === "pa" ? alert.titlePa : lang === "skr" ? alert.titleSkr : alert.title}
                  </h3>
                  <p className={`text-sm text-stone-500 ${isNonLatin ? "urdu-text mt-[-6px]" : ""}`}>
                    {lang === "ur" ? alert.timeUr : lang === "pa" ? alert.timePa : lang === "skr" ? alert.timeSkr : alert.time}
                  </p>
                  <p className={`text-xs font-bold mt-2 uppercase tracking-wider opacity-60 ${isNonLatin ? "urdu-text" : ""}`}>
                    {t.tapForPlan}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {selectedAlert && (
        <ProtectionPlanDialog 
          isOpen={!!selectedAlert} 
          onClose={() => setSelectedAlert(null)}
          title={
            lang === "ur" ? selectedAlert.titleUr : 
            lang === "pa" ? selectedAlert.titlePa : 
            lang === "skr" ? selectedAlert.titleSkr : 
            selectedAlert.title
          }
          markdownContent={
            lang === "ur" ? selectedAlert.planMarkdownUr : 
            lang === "pa" ? selectedAlert.planMarkdownPa : 
            lang === "skr" ? selectedAlert.planMarkdownSkr : 
            selectedAlert.planMarkdown
          }
          lang={lang}
        />
      )}
    </>
  );
}
