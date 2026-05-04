import * as Dialog from "@radix-ui/react-dialog";
import { X, Calendar, Droplets, ShieldCheck, Sun } from "lucide-react";
import { ReactNode } from "react";
import { Lang } from "@/lib/mockData";
import { translations } from "@/lib/translations";

interface ProtectionPlanDialogProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  markdownContent: string;
  lang: Lang;
}

// Very simple markdown parser for the specific dialog styling
function parseMarkdown(md: string) {
  const lines = md.split("\n");
  const parsed: ReactNode[] = [];
  
  lines.forEach((line, i) => {
    if (line.startsWith("## ")) {
      parsed.push(<h2 key={i} className="text-xl font-black text-emerald-900 mt-6 mb-3">{line.replace("## ", "")}</h2>);
    } else if (line.startsWith("### ")) {
      parsed.push(<h3 key={i} className="text-lg font-bold text-emerald-800 mt-4 mb-2">{line.replace("### ", "")}</h3>);
    } else if (line.startsWith("- ")) {
      parsed.push(
        <li key={i} className="flex items-start gap-2 text-stone-700 text-sm mb-2">
          <span className="text-emerald-500 mt-1">•</span>
          <span dangerouslySetInnerHTML={{ __html: line.replace("- ", "").replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") }} />
        </li>
      );
    } else if (line.trim().length > 0) {
      parsed.push(
        <p key={i} className="text-stone-700 text-sm mb-3 leading-relaxed" 
           dangerouslySetInnerHTML={{ __html: line.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") }} />
      );
    }
  });
  
  return parsed;
}

export default function ProtectionPlanDialog({ isOpen, onClose, title, markdownContent, lang }: ProtectionPlanDialogProps) {
  const t = translations[lang];
  const isRtl = lang === "ur" || lang === "pa" || lang === "skr";
  return (
    <Dialog.Root open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="DialogOverlay" />
        <Dialog.Content className={`DialogContent ${isRtl ? "text-right urdu-text" : "text-left"}`} dir={isRtl ? "rtl" : "ltr"}>
          
          <div className="flex justify-between items-start mb-4">
            <div className={`flex items-center gap-3 ${isRtl ? "flex-row-reverse" : ""}`}>
              <div className="w-10 h-10 rounded-full bg-emerald-100 flex items-center justify-center text-emerald-600">
                <ShieldCheck size={20} />
              </div>
              <Dialog.Title className="text-xl font-black text-stone-900">
                {title}
              </Dialog.Title>
            </div>
            
            <Dialog.Close asChild>
              <button className="p-2 rounded-full hover:bg-stone-100 text-stone-400 hover:text-stone-700 transition-colors">
                <X size={20} />
              </button>
            </Dialog.Close>
          </div>

          <Dialog.Description className="sr-only">
            7-Day Crop Protection Plan
          </Dialog.Description>

          {/* Quick Stats Header */}
          <div className="grid grid-cols-3 gap-2 mb-6 mt-4">
            <div className="bg-orange-50 rounded-xl p-3 border border-orange-100 flex flex-col items-center justify-center">
              <Sun size={18} className="text-orange-500 mb-1" />
              <span className={`text-xs font-bold text-orange-900 ${isRtl ? "urdu-text" : ""}`}>{t.heatAlert}</span>
            </div>
            <div className="bg-blue-50 rounded-xl p-3 border border-blue-100 flex flex-col items-center justify-center">
              <Droplets size={18} className="text-blue-500 mb-1" />
              <span className={`text-xs font-bold text-blue-900 ${isRtl ? "urdu-text" : ""}`}>{t.irrigate}</span>
            </div>
            <div className="bg-emerald-50 rounded-xl p-3 border border-emerald-100 flex flex-col items-center justify-center">
              <Calendar size={18} className="text-emerald-500 mb-1" />
              <span className={`text-xs font-bold text-emerald-900 ${isRtl ? "urdu-text" : ""}`}>{t.sevenDays}</span>
            </div>
          </div>

          <div className="bg-[#faf7f2] border border-stone-200 rounded-2xl p-4 sm:p-6 shadow-inner">
            <ul className="space-y-1">
              {parseMarkdown(markdownContent)}
            </ul>
          </div>
          
          <div className="mt-6">
            <Dialog.Close asChild>
              <button className={`w-full flex items-center justify-center bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3.5 rounded-xl transition-all shadow-md active:scale-95 ${isRtl ? "urdu-text" : ""}`}>
                {t.understood}
              </button>
            </Dialog.Close>
          </div>

        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
