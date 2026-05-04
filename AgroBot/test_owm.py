import os
import requests
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

def test_owm():
    owm_key = os.getenv("OPENWEATHERMAP_API_KEY", "").strip()
    if not owm_key:
        print("NO OWM KEY")
        return
        
    lat, lon = 31.5497, 74.3436 # Lahore
    
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={owm_key}"
    resp = requests.get(url, timeout=10)
    print("STATUS", resp.status_code)
    
    if resp.status_code == 200:
        raw = resp.json()
        list_data = raw.get("list", [])
        
        daily_agg = defaultdict(lambda: {
            "max_temp_c": -999,
            "min_temp_c": 999,
            "rain_mm": 0,
            "conditions": [],
            "rain_probs": []
        })

        for item in list_data:
            dt_txt = item.get("dt_txt", "").split(" ")[0]
            temp = item["main"]["temp"]
            rain = item.get("rain", {}).get("3h", 0)
            cond = item["weather"][0]["description"].capitalize()
            pop  = item.get("pop", 0) * 100
            
            day = daily_agg[dt_txt]
            day["max_temp_c"] = max(day["max_temp_c"], temp)
            day["min_temp_c"] = min(day["min_temp_c"], temp)
            day["rain_mm"] += rain
            day["conditions"].append(cond)
            day["rain_probs"].append(pop)

        days = []
        for dt_txt, day in sorted(daily_agg.items()):
            from collections import Counter
            most_common_cond = Counter(day["conditions"]).most_common(1)[0][0] if day["conditions"] else "Unknown"
            days.append({
                "date": dt_txt,
                "max_temp_c": round(day["max_temp_c"], 1),
                "min_temp_c": round(day["min_temp_c"], 1),
                "rain_mm": round(day["rain_mm"], 1),
                "condition": most_common_cond,
                "rain_prob_%": round(max(day["rain_probs"] or [0])),
            })

        print("DAYS PARSED:", len(days))
        for d in days:
            print(d)

if __name__ == "__main__":
    test_owm()
