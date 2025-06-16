# Whisper 
#!pip install -U openai-whisper --quiet 
import whisper 
# Transcrire avec Whisper 
def transcribe_audio_with_whisper(audio_file, model_name="base"): 
    # Charger le modèle Whisper 
    model = whisper.load_model(model_name) 
    # Transcrire le fichier audio 
    result = model.transcribe(audio_file) 
    # Afficher la transcription 
    print("Transcription :") 
    print(result["text"]) 

# Transcrire l'audio avec Whisper 
transcribe_audio_with_whisper("./audio.opus") 
