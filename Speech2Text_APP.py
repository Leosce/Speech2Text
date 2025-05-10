import streamlit as st
import librosa
from transformers import pipeline, AutoProcessor
from st_audiorec import st_audiorec
from io import BytesIO
import torch



# Load the processor (once for caching)
@st.cache_resource
def load_processor():
        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")  # Using Whisper tiny
        return processor

processor = load_processor()

# Set up the Streamlit app
st.title("Whisper Tiny Transcription")
st.write("Record or upload an audio file for transcription.")

# Audio recording option
audio_data = st_audiorec()

# Audio upload option
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Process the audio
with st.spinner("Give it a sec..."):
    if audio_data:
        # Convert audio data to the correct format
        audio_bytes = BytesIO(audio_data)
        audio_array, sr = librosa.load(audio_bytes, sr=processor.feature_extractor.sampling_rate)
    elif uploaded_file:
        # Load audio from the uploaded file
        audio_bytes = BytesIO(uploaded_file.read())
        audio_array, sr = librosa.load(audio_bytes, sr=processor.feature_extractor.sampling_rate)
    else:
        audio_array = None

# Transcribe the audio if available
if audio_array is not None:
    with st.spinner("Transcribing..."):
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",  # Using Whisper tiny
            processor=processor,  # Use the cached processor
        )
        transcription = pipe(audio_array)["text"]
    
        st.write("**Transcription:**")
        st.write(transcription)
