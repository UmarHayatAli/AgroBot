# Clone repo
!git clone https://github.com/UmarHayatAli/AgroBot.git

# Move into repo
%cd AgroBot

# Install dependencies
!pip install -r requirements.txt

# Move into repo
%cd Backend


#  Run This Cell
import os

os.environ["GROQ_API_KEY"] = "your_key_here"
os.environ["OPENWEATHER_API_KEY"] = "your_key_here"

#  Run Chatbot
!python MultiLingualChatBot.py


