############
# Analyse de l'amplitude sonore dans un discours
# Stéphane Meurisse
# www.codeandcortex.fr
# Date : 19-04-2024
############

### installation des ibrairies ###
pip install numpy
pip install pandas
pip install openai-whisper
pip install plotly
pip install streamlit 
pip install soundfile

### installation ffmpeg ###
Pour que Whisper (et l’ensemble de la pipeline audio) fonctionne, ffmpeg doit être disponible en local (Mac ou PC)
macOS, dans le terminal : brew install ffmpeg
