import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features.weather_insights import WeatherInsightsTool

def test():
    tool = WeatherInsightsTool()
    print("Testing weather for Lahore...")
    res = tool.run("weather in lahore", location="Lahore")
    print("RESULT:", res)

if __name__ == "__main__":
    test()
