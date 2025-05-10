import streamlit as st
import librosa
from transformers import pipeline, AutoProcessor
from st_audiorec import st_audiorec
from io import BytesIO

# Load the processor (once for caching)
@st.cache_resource
def load_processor():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    return processor

processor = load_processor()

# Set up the Streamlit app
st.title("Qwen2-Audio Transcription")
st.write("Record or upload an audio file for transcription.")

# Audio recording option
audio_data = st_audiorec(max_seconds=30)

# Audio upload option
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Process the audio
if audio_data:
    # Convert audio data to the correct format
    audio_bytes = BytesIO(audio_data.getvalue())
    audio_array, sr = librosa.load(audio_bytes, sr=processor.feature_extractor.sampling_rate)
elif uploaded_file:
    # Load audio from the uploaded file
    audio_bytes = BytesIO(uploaded_file.read())
    audio_array, sr = librosa.load(audio_bytes, sr=processor.feature_extractor.sampling_rate)
else:
    audio_array = None

# Transcribe the audio if available
if audio_array is not None:
    pipe = pipeline(
        "automatic-speech-recognition",
        model="Qwen/Qwen2-Audio-7B-Instruct",
        processor=processor,  # Use the cached processor
    )
    transcription = pipe(audio_array, sampling_rate=sr)["text"]

    st.write("**Transcription:**")
    st.write(transcription)
