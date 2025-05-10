import streamlit as st
import librosa
import numpy as np
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from io import BytesIO
import soundfile as sf

# Load the model and processor
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")
    return processor, model

processor, model = load_model()

# Set up the Streamlit app
st.title("Qwen2-Audio Transcription")
st.write("Record or upload an audio file for transcription.")

# Audio recording option
audio_data = st.audio_recorder("Record audio (max 30 seconds)", max_seconds=30)

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
    inputs = processor(audio=audio_array, return_tensors="pt", sampling_rate=sr)
    inputs.input_values = inputs.input_values.to(model.device)

    generated_ids = model.generate(**inputs, max_length=256)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.write("**Transcription:**")
    st.write(transcription)
