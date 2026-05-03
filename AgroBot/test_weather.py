import requests

def test():
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":         31.5497,
            "longitude":        74.3436,
            "current":          "temperature_2m,precipitation,weathercode,relative_humidity_2m",
            "daily":            "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,precipitation_probability_max",
            "timezone":         "Asia/Karachi",
            "forecast_days":    7,
            "wind_speed_unit":  "kmh",
        },
        timeout=10,
    )
    with open("weather_output.txt", "w") as f:
        f.write(f"STATUS {resp.status_code}\nRESPONSE {resp.text}\n")

if __name__ == "__main__":
    test()
