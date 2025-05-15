import flask, socketio, openai, pyaudio, wave, pydub
import langchain, langchain_unstructured, langchain_openai, langchain_huggingface, langchain_community
from sentence_transformers import SentenceTransformer
import speech_recognition, langdetect, engineio, faiss

print("✅ All imports succeeded! Your environment is ready.")
