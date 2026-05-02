// Mock data for AgroBot demo

export type Severity = "critical" | "warning" | "info" | "ok";
export type Lang = "en" | "ur" | "pa" | "skr";
export type MessageRole = "user" | "bot" | "typing";

export interface Notification {
  id: string;
  title: string;
  titleUr: string;
  titlePa: string;
  titleSkr: string;
  body: string;
  bodyUr: string;
  bodyPa: string;
  bodySkr: string;
  severity: Severity;
  icon: string;
  time: string;
  unread: boolean;
}

export interface Message {
  id: string;
  role: MessageRole;
  text: string;
  textUr?: string;
  textPa?: string;
  textSkr?: string;
  intent?: "weather" | "disease" | "soil" | "flood" | "general";
  timestamp: Date;
  imageUrl?: string;
}

export interface FarmContext {
  city: string;
  crop: string;
  tempC: number;
  condition: string;
}

export const CROPS = [
  "Wheat — گندم",
  "Cotton — کپاس",
  "Rice — چاول",
  "Sugarcane — گنا",
  "Maize — مکئی",
  "Mango — آم",
  "Citrus — کینو",
  "Potato — آلو",
];

export const CITIES = [
  "Lahore", "Karachi", "Islamabad", "Multan",
  "Faisalabad", "Peshawar", "Quetta", "Hyderabad",
  "Rawalpindi", "Bahawalpur",
];

export const initialNotifications: Notification[] = [
  {
    id: "1",
    title: "🚨 Pest Outbreak Alert — Cotton",
    titleUr: "کپاس پر کیڑے مکوڑوں کا حملہ",
    titlePa: "کپاس تے کیڑے مکوڑیاں دا حملہ",
    titleSkr: "کپاس تے کیڑے مکوڑیاں دا حملہ",
    body: "Armyworm detected in nearby farms. Monitor fields daily and apply Imidacloprid if threshold reached.",
    bodyUr: "قریبی فارمز میں آرمی ورم پایا گیا ہے۔ روزانہ نگرانی کریں اور ضرورت پڑنے پر امیڈاکلوپریڈ کا استعمال کریں۔",
    bodyPa: "نیڑے دے فارماں وچ آرمی ورم پایا گیا اے۔ روزانہ نگرانی کرو تے لوڑ پین تے امیڈاکلوپریڈ دا استعمال کرو۔",
    bodySkr: "نیڑے دے فارماں وچ آرمی ورم پایا گیا ہے۔ روزانہ نگرانی کرو تے ضرورت پاوݨ تے امیڈاکلوپریڈ دا استعمال کرو۔",
    severity: "critical",
    icon: "🐛",
    time: "25 mins ago",
    unread: true,
  },
  {
    id: "2",
    title: "🌧️ Heavy Rain Alert — 3 Days",
    titleUr: "بھاری بارش کا انتباہ — ۳ دن",
    titlePa: "بھاری بارش دی اطلاع — ۳ دن",
    titleSkr: "بھاری بارش دی اطلاع — ۳ دن",
    body: "80mm rainfall expected over 3 days. Check drainage channels and delay fertilizer application.",
    bodyUr: "۳ دنوں میں ۸۰ ملی میٹر بارش متوقع ہے۔ نالیوں کی جانچ کریں اور کھاد کا استعمال ملتوی کریں۔",
    bodyPa: "۳ دناں وچ ۸۰ ملی میٹر بارش متوقع اے۔ نالیاں دی جانچ کرو تے کھاد دا استعمال ملتوی کرو۔",
    bodySkr: "۳ ڈیہانہہ وچ ۸۰ ملی میٹر بارش متوقع ہے۔ نالیاں دی جانچ کرو تے کھاد دا استعمال ملتوی کرو۔",
    severity: "warning",
    icon: "🌧️",
    time: "1 hr ago",
    unread: true,
  },
  {
    id: "3",
    title: "⛈️ Monsoon Arrives in 4 Days",
    titleUr: "مون سون ۴ دن میں آئے گا",
    titlePa: "مون سون ۴ دن وچ آئے گا",
    titleSkr: "مون سون ۴ ڈیہانہہ وچ آسی",
    body: "Monsoon season begins July 15. Prepare drainage and consider early harvest for sensitive crops.",
    bodyUr: "مون سون ۱۵ جولائی کو شروع ہوگا۔ نکاسی آب کا انتظام کریں۔",
    bodyPa: "مون سون ۱۵ جولائی نوں شروع ہوئے گا۔ نکاسی آب دا انتظام کرو۔",
    bodySkr: "مون سون ۱۵ جولائی کوں شروع تھیسی۔ نکاسی آب دا انتظام کرو۔",
    severity: "warning",
    icon: "⛈️",
    time: "3 hrs ago",
    unread: false,
  },
  {
    id: "4",
    title: "✅ Soil Moisture — Optimal",
    titleUr: "مٹی کی نمی — بہترین",
    titlePa: "مٹی دی نمی — بہترین",
    titleSkr: "مٹی دی نمی — بہترین",
    body: "Soil moisture levels in Lahore are within the optimal range for wheat growth. No irrigation needed today.",
    bodyUr: "لاہور میں مٹی کی نمی گندم کی نشوونما کے لیے بہترین سطح پر ہے۔ آج پانی کی ضرورت نہیں۔",
    bodyPa: "لاہور وچ مٹی دی نمی گندم دی نشوونما لئی بہترین سطح تے اے۔ اج پانی دی لوڑ نہیں۔",
    bodySkr: "لاہور وچ مٹی دی نمی گندم دی نشوونما سانگے بہترین سطح تے ہے۔ اج پاݨی دی لوڑ کائنی۔",
    severity: "ok",
    icon: "✅",
    time: "6 hrs ago",
    unread: false,
  },
  {
    id: "5",
    title: "ℹ️ Market Price Update",
    titleUr: "منڈی کی قیمتوں کی تازہ کاری",
    titlePa: "منڈی دی قیمتاں دی تازہ کاری",
    titleSkr: "منڈی دی قیمتاں دی تازہ کاری",
    body: "Wheat price: PKR 3,200/40kg. Cotton: PKR 8,500/maund. Mango: PKR 120/kg.",
    bodyUr: "گندم: ۳۲۰۰ روپے/۴۰کلو۔ کپاس: ۸۵۰۰ روپے/من۔ آم: ۱۲۰ روپے/کلو۔",
    bodyPa: "گندم: ۳۲۰۰ روپے/۴۰کلو۔ کپاس: ۸۵۰۰ روپے/من۔ امب: ۱۲۰ روپے/کلو۔",
    bodySkr: "گندم: ۳۲۰۰ روپے/۴۰کلو۔ کپاس: ۸۵۰۰ روپے/من۔ امب: ۱۲۰ روپے/کلو۔",
    severity: "info",
    icon: "📊",
    time: "12 hrs ago",
    unread: false,
  },
];

export const initialMessages: Message[] = [
  {
    id: "1",
    role: "bot",
    intent: "general",
    text: "Assalam-o-Alaikum! 🌾 I am AgroBot, your AI agricultural advisor. I can help you with:\n\n• 🌱 **Crop & Soil** recommendations\n• 🌤️ **Weather** forecasts & heatwave alerts\n• 🔬 **Disease** diagnosis from photos\n• 🌊 **Flood** risk warnings\n\nAsk me anything in English, Urdu, Punjabi, or Saraiki!",
    textUr: "السلام علیکم! 🌾 میں AgroBot ہوں، آپ کا زرعی مشیر۔ میں آپ کی مدد کر سکتا ہوں:\n\n• 🌱 فصل اور مٹی کی سفارشات\n• 🌤️ موسم کی پیش گوئی اور گرمی کی لہر کی اطلاعات\n• 🔬 تصاویر سے بیماری کی تشخیص\n• 🌊 سیلاب کے خطرے کی اطلاعات\n\nاردو، انگریزی، پنجابی یا سرائیکی میں کچھ بھی پوچھیں!",
    textPa: "السلام علیکم! 🌾 میں ایگرو بوٹ آں، تہاڈا زرعی مشیر۔ میں تہاڈی مدد کر سکدا آں:\n\n• 🌱 فصل تے مٹی دیاں سفارشاں\n• 🌤️ موسم دی پیش گوئی تے گرمی دی لہر دی اطلاع\n• 🔬 تصویراں توں بیماری دی تشخیص\n• 🌊 ہڑھ دے خطرے دی اطلاع\n\nاردو، انگریزی، پنجابی یا سرائیکی وچ کجھ وی پچھو!",
    textSkr: "السلام علیکم! 🌾 میں ایگرو بوٹ ہاں، تہاڈا زرعی صلاحکار۔ میں تہاڈی مدد کر سگدا ہاں:\n\n• 🌱 فصل تے مٹی دیاں سفارشاں\n• 🌤️ موسم دی پیش گوئی تے گرمی دی لہر دی اطلاع\n• 🔬 تصویراں توں بیماری دی تشخیص\n• 🌊 ہڑھ دے خطرے دی اطلاع\n\nاردو، انگریزی، پنجابی یا سرائیکی وچ کجھ وی پچھو!",
    timestamp: new Date(Date.now() - 60000),
  },
  {
    id: "2",
    role: "user",
    text: "ملتان میں اس ہفتے موسم کیسا رہے گا؟",
    timestamp: new Date(Date.now() - 30000),
  },
  {
    id: "3",
    role: "bot",
    intent: "weather",
    text: "🔥 **HEATWAVE ALERT — Multan**\n\nCurrent temperature: **41.1°C** — Clear sky\n\nThis week's outlook:\n- Max temperature will reach **44°C** on May 4\n- **Zero rainfall** expected for 7 days\n- Heat stress risk is **HIGH** for cotton & sugarcane\n\n**🚜 Action for Today:** Irrigate your fields before 7 AM and after 6 PM. Apply potassium-based foliar spray to reduce heat stress on crops.",
    textUr: "🔥 **گرمی کی لہر کا انتباہ — ملتان**\n\nموجودہ درجہ حرارت: **۴۱.۱ ڈگری** — صاف آسمان\n\nاس ہفتے کا جائزہ:\n- زیادہ سے زیادہ درجہ حرارت ۴ مئی کو **۴۴ ڈگری** تک پہنچے گا\n- ۷ دنوں میں **کوئی بارش** متوقع نہیں\n- کپاس اور گنے کے لیے گرمی کا دباؤ **زیادہ** ہے\n\n**🚜 آج کا کام:** صبح ۷ بجے سے پہلے اور شام ۶ بجے کے بعد پانی دیں۔",
    textPa: "🔥 **گرمی دی لہر دا الرٹ — ملتان**\n\nموجودہ درجہ حرارت: **۴۱.۱ ڈگری** — صاف اسمان\n\nاس ہفتے دا جائزہ:\n- ودھ توں ودھ درجہ حرارت ۴ مئی نوں **۴۴ ڈگری** تیکر پہنچے گا\n- ۷ دناں وچ **کوئی بارش** متوقع نہیں\n- کپاس تے گنے لئی گرمی دا دباؤ **ودھ** اے\n\n**🚜 اج دا کم:** سویرے ۷ بجے توں پہلاں تے شام ۶ بجے توں بعد پانی دیو۔",
    textSkr: "🔥 **گرمی دی لہر دا الرٹ — ملتان**\n\nموجودہ درجہ حرارت: **۴۱.۱ ڈگری** — صاف اسمان\n\nاس ہفتے دا جائزہ:\n- ودھ توں ودھ درجہ حرارت ۴ مئی کوں **۴۴ ڈگری** تئیں پہنچسی\n- ۷ ڈیہانہہ وچ **کوئی بارش** متوقع کائنی\n- کپاس تے گنے سانگے گرمی دا دباؤ **ودھ** ہے\n\n**🚜 اج دا کم:** سویرے ۷ بجے توں پہلے تے شام ۶ بجے توں بعد پاݨی ڈیو۔",
    timestamp: new Date(Date.now() - 15000),
  },
];
