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

os.environ["GROQ_API_KEY"] = "your_key_here";

os.environ["OPENWEATHER_API_KEY"] = "your_key_here"
# Prepare RAG.
!python ingest_knowledge_base.py
#  Run My final improved  Chatbot
!python MultiLingualAgenticRAGImageDiseasePredictionWithWeatherForecasting.py


# To Test Image Disease detection
--scan   → open image picker for leaf disease detection
Add --scan at the end of prompt

# To Instruct Chatbot to provide audio output
 --audio  → enable voice output for this message
 the audios will be saved in audio_outputs folder.On google colab click on folder icon on the left bar.Then Go to Agrobot and then to Backend and then to audio_outputs folder .there you  will see chatbott output.

 # Implemented Weather forecasting
 It will give critical warnings regarding flood and heatwave If There are any.
# Improved ChatBot response Through Prompt Engineering

