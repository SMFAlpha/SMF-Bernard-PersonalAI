#!/usr/bin/env python
# coding: utf-8

"""
Flask-based web application for Bernard conversation assistant.
It provides two conversation flows:
1. "Start Conversation": The text in the textbox is spoken via TTS (as Bernard's text)
   and recorded in chat history. The server then auto-records audio and returns its transcript.
2. "Send LLM": The textbox content (Subject's text) is sent to OpenAI along with a system prompt
   and retrieved context. Four completions are returned.
When a reply is selected, it is spoken via TTS and the exchange is recorded.
"""

# Move all imports to the very top
import os
import json
from dotenv import load_dotenv
import glob
import openai
import requests
import time
import shutil
import uuid
import re
import threading
import traceback
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from flask_socketio import SocketIO, emit, join_room, leave_room
import pyaudio
import audioop
import wave
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Queue
from typing import Optional, AsyncGenerator, List, Dict, Any, Callable
import atexit
import signal
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import speech_recognition as sr
from pydub import AudioSegment
from werkzeug.exceptions import RequestEntityTooLarge

SETTINGS_FILE = 'settings.json'

def load_settings_from_file():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
            print(f"Loaded settings from {SETTINGS_FILE}: {settings}")
            return settings
        else:
            print(f"Settings file {SETTINGS_FILE} does not exist.")
            return None
    except Exception as e:
        print(f"Error loading settings from file: {e}")
        return None

def save_settings_to_file(settings):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"Settings saved to {SETTINGS_FILE}: {settings}")
        return True
    except Exception as e:
        print(f"Error saving settings to file: {e}")
        return False

def create_default_settings():
    settings = {
        "voice": {
            "apiKey": "",
            "voiceId": "CwhRBWXzGAHq8TQ4Fs17",
            "model": "eleven_multilingual_v2",
            "outputFormat": "mp3_44100_192",
            "seed": 8675309,
            "voiceSettings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": 0.9,
                "style": 0.3
            }
        },
        "llm": {
            "openai": {
                "apiKey": "",
                "model": "gpt-4.1-mini"
            },
            "groq": {
                "apiKey": "",
                "model": "llama-3.3-70b-specdec"
            },
            "perplexity": {
                "apiKey": "",
                "model": "sonar"
            },
            "provider": "openai",
            "temperature": 0.9,
            "maxTokens": 150,
            "completionsCount": 4
        },
        "recorder": {
            "silenceThreshold": 500,
            "silenceDuration": 5.0,
            "minRecordingDuration": 1.0,
            "maxRecordingDuration": 60.0,
            "useNoiseReduction": True
        },
        "system": {
            "systemPrompt": "You are the persona Bernard Muller...",
            "fileLocations": {
                "kbDir": "data/kb",
                "vectorStoreDir": "data/vector_store",
                "chatVectorStoreDir": "data/chat_vector_store",
                "chatHistoryFile": "data/chat_history.json",
                "phrasesFile": "data/phrases.json"
            },
            "language": {
                "defaultLanguage": "auto",
                "useLanguageDetection": True
            },
            "debugMode": False
        }
    }
    print(f"Created default settings: {settings}")
    return settings

def ensure_settings_json():
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(create_default_settings(), f, indent=2)
        print(f"Created default settings.json at {SETTINGS_FILE}")

ensure_settings_json()
settings = load_settings_from_file()

# Load environment variables from .env file
load_dotenv()

# For knowledge-base text processing
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
# Fixed import for FAISS
from langchain_community.vectorstores import FAISS
# Import simpler PDF loaders that don't depend on onnxruntime
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# For local embeddings 
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# For server-side audio recording and playback
import speech_recognition as sr
from pydub import AudioSegment

# Set ffmpeg and ffprobe paths explicitly – update these to your installation:
AudioSegment.converter = r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.exe"

# Optional language detection
try:
    from langdetect import detect
except ImportError:
    detect = None

##############################################################################
# CONFIGURATION
##############################################################################

# Load API keys from environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# Verify required API keys
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
if not ELEVENLABS_API_KEY:
    raise ValueError("ELEVENLABS_API_KEY environment variable is not set")

# Directory configuration (now from settings.json)
settings = load_settings_from_file()
KB_DIR = settings['system']['fileLocations']['kbDir']
VECTOR_STORE_DIR = settings['system']['fileLocations']['vectorStoreDir']
CHAT_HISTORY_JSON = settings['system']['fileLocations']['chatHistoryFile']
PHRASES_FILE = settings['system']['fileLocations']['phrasesFile']
CHAT_VECTOR_STORE_DIR = settings['system']['fileLocations'].get('chatVectorStoreDir', 'data/chat_vector_store')

# Add environment variable defaults
SUBJECTS_FILE = os.getenv("SUBJECTS_FILE", os.path.join(os.path.dirname(__file__), "data", "subjects.json"))

# Add this after imports and before any functions that use CURRENT_SUBJECT
CURRENT_SUBJECT = ["Default Subject"]

##############################################################################
# CLASSES
##############################################################################
@dataclass
class SimpleDoc:
    page_content: str
    metadata: dict

class SmartAudioRecorder:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.thread = None
        
        # Voice detection parameters
        self.silence_threshold = 500  # Adjust based on your microphone/environment default = 700
        self.silence_duration = 5.0   # Seconds of silence to consider end of speech
        self.min_recording_duration = 1.0  # Minimum seconds to record
        self.max_recording_duration = 60.0  # Maximum seconds to record
        
        # State variables
        self.last_audio_time = 0
        self.recording_start_time = 0
        self.stopped_automatically = False
    
    def start_recording(self):
        if self.is_recording:
            return False
        
        self.frames = []
        self.is_recording = True
        self.stopped_automatically = False
        self.recording_start_time = time.time()
        self.last_audio_time = time.time()  # Reset the last audio time
        self.thread = threading.Thread(target=self._record)
        self.thread.start()
        return True
        
    def _record(self):
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        while self.is_recording:
            try:
                data = stream.read(self.chunk_size)
                self.frames.append(data)

                # Calculate audio level
                rms = audioop.rms(data, 2)  # Width=2 for paInt16

                # Check if this is audible speech
                if rms > self.silence_threshold:
                    self.last_audio_time = time.time()

                # Check if we've exceeded the maximum recording time
                current_duration = time.time() - self.recording_start_time
                if current_duration > self.max_recording_duration:
                    print("Max recording duration reached, stopping automatically")
                    self.is_recording = False
                    self.stopped_automatically = True
                    break

                # Check if we've been silent for too long (but only if minimum duration reached)
                silence_time = time.time() - self.last_audio_time
                if (current_duration > self.min_recording_duration and 
                    silence_time > self.silence_duration):
                    print(f"Silence detected for {silence_time:.1f} seconds, stopping automatically")
                    self.is_recording = False
                    self.stopped_automatically = True
                    break
            except Exception as e:
                print(f"Error during recording: {e}")
                break

        try:
            stream.stop_stream()
            stream.close()
            print(f"Recording stopped with {len(self.frames)} frames")
        except Exception as e:
            print(f"Error closing stream: {e}")
        
    def stop_recording(self):
        if not self.is_recording:
            return None
            
        self.is_recording = False
        if self.thread:
            self.thread.join()
            
        return b''.join(self.frames)
    
    def is_stopped_automatically(self):
        return self.stopped_automatically
    
    def save_wav(self, filename="recording.wav"):
        if not self.frames:
            return False
            
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        return True
        
    def __del__(self):
        self.is_recording = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.audio is not None:  # Check added to ensure self.audio exists
            self.audio.terminate()

class LocalEmbeddings:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        """Initialize a local embeddings model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                Default is "paraphrase-multilingual-MiniLM-L12-v2" which works well for Dutch documents.
                Other options: "distiluse-base-multilingual-cased-v1" (good multilingual model),
                "paraphrase-multilingual-mpnet-base-v2" (higher quality but slower)
        """
        self.model_name = model_name
        
        # Check if CUDA is available
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create HuggingFaceEmbeddings with SentenceTransformers model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},  # Use GPU if available
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"Loaded local embeddings model: {model_name} on {device}")
    
    def embed_documents(self, texts):
        """Embed a list of documents/texts with GPU optimizations."""
        # Process in optimized batches
        import torch
        import gc
        
        batch_size = 64  # Larger batch size for GPU
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Periodic GPU memory cleanup
            if i % (batch_size * 10) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        return all_embeddings
    
    def embed_query(self, text):
        """Embed a single query text."""
        return self.embeddings.embed_query(text)
        
    # Make the object callable to maintain compatibility with FAISS
    def __call__(self, text):
        """Make the embeddings object callable for compatibility with FAISS."""
        # Check if input is a list/array or a single string
        if isinstance(text, list):
            return self.embed_documents(text)
        else:
            return self.embed_query(text)
        

def clean_completions(completions):
    """
    Clean and normalize completions to ensure we have exactly 4 single responses.
    Handles nested lists, numbered responses, and ensures proper formatting.
    """
    # Flatten any nested lists
    flat = []
    for c in completions:
        if isinstance(c, list):
            flat.extend(c)
        elif isinstance(c, str):
            # If the string contains multiple numbered responses, split them
            matches = re.findall(r'(?:^|\n)[1-4][\.:]\s*(.*?)(?=\n[1-4][\.:]|\Z)', c, re.DOTALL)
            if matches and len(matches) > 1:
                flat.extend([m.strip() for m in matches])
            else:
                flat.append(c.strip())
    # Clean each completion
    cleaned = []
    for c in flat:
        # Remove any remaining "Bernard:" prefixes
        if c.lower().startswith("bernard:"):
            c = c[c.find(':')+1:].strip()
        # Remove any numbered prefixes
        c = re.sub(r'^\s*[1-4][\.:]\s*', '', c)
        cleaned.append(c)
    # Only keep the first 4, pad with empty strings if needed
    while len(cleaned) < 4:
        cleaned.append("")
    return cleaned[:4]



##############################################################################
# OPTIMIZED PIPELINE
##############################################################################
class ParallelPipeline:
    """
    Main pipeline class to handle the parallelized workflow:
    1. Start transcription immediately
    2. Begin context retrieval as soon as transcription is ready
    3. Start LLM generation with streaming as soon as context is ready
    4. Begin TTS streaming as soon as LLM chunks start arriving
    """
    def __init__(self, socketio):
        self.socketio = socketio
        self.active_pipelines = {}  # Store active pipelines by session ID

    def _emit_update(self, session_id, event_type, data):
        """
        Emit a Socket.IO event to the client with the provided data.
        
        Args:
            session_id: The Socket.IO session ID to emit to
            event_type: The type of event (e.g., 'status_update', 'transcript_ready')
            data: The data to send with the event
        """
        try:
            # Remove the 'room' parameter
            self.socketio.emit(event_type, data, to=session_id)
            print(f"Emitted {event_type} to {session_id}: {data}")
        except Exception as e:
            print(f"Error emitting {event_type} to {session_id}: {e}")
            
    def process_audio(self, audio_data, session_id, auto_mode=False, user_text=None):
        """
        Synchronous wrapper for the async process_audio method.
        This method is called by socketio.start_background_task.
        
        Parameters:
            audio_data: The audio data to transcribe
            session_id: Unique identifier for this session
            auto_mode: If True, automatically select first completion and speak it
            user_text: Optional pre-transcribed text
        """
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async process
            loop.run_until_complete(self._process_audio_async(audio_data, session_id, auto_mode, user_text))
            loop.close()
        except Exception as e:
            print(f"Error in process_audio: {e}")
            traceback.print_exc()
            # Try to notify the client of the error
            try:
                self._emit_update(session_id, 
                              {"status": f"Error processing audio: {str(e)}"}, 
                              room=session_id)
            except Exception as inner_e:
                print(f"Error sending error notification: {inner_e}")
    
    async def _process_audio_async(self, audio_data, session_id, auto_mode=False, user_text=None):
        """
        Async implementation of the audio processing pipeline
        
        Args:
            audio_data: The audio data to transcribe
            session_id: Unique identifier for this session
            auto_mode: If True, automatically select first completion and speak it
            user_text: Optional pre-transcribed text
        """
        pipeline_id = f"{session_id}_{int(time.time())}"
        self.active_pipelines[pipeline_id] = {
            "status": "transcribing",
            "transcript": "",
            "context": "",
            "llm_responses": [],  # Change to list for multiple completions
            "audio_urls": []
        }
        
        # Step 1: Transcribe audio (async)
        if user_text:
            transcript = user_text
            self._emit_update(session_id, "transcript_ready", {"transcript": transcript})
        else:
            # Start transcription in background
            self._emit_update(session_id, "status_update", {"status": "Transcribing audio..."})
            transcript = await transcribe_audio_async(audio_data)
            self._emit_update(session_id, "transcript_ready", {"transcript": transcript})
        # Access the global lastBernardStatement variable
        global last_user_text, flow_mode
        # Use CURRENT_SUBJECT to access global state
        current_last_bernard = ""

        # Look for lastBernardStatement in global scope
        if 'lastBernardStatement' in globals():
            current_last_bernard = globals()['lastBernardStatement']
        elif hasattr(self, 'lastBernardStatement'):
            current_last_bernard = self.lastBernardStatement

        if current_last_bernard:
            display_text = f"Bernard: {current_last_bernard}\n\nUser: {transcript}"
            self._emit_update(session_id, "update_textbox", {"text": display_text})
        else:
            self._emit_update(session_id, "update_textbox", {"text": transcript})
        
        if not transcript:
            self._emit_update(session_id, "status_update", 
                             {"status": "No transcript obtained from audio"})
            del self.active_pipelines[pipeline_id]
            return
        
        self.active_pipelines[pipeline_id]["transcript"] = transcript
        self.active_pipelines[pipeline_id]["status"] = "retrieving_context"
        
        # Step 2: Begin context retrieval in parallel
        self._emit_update(session_id, "status_update", {"status": "Retrieving context..."})
        
        # Start all retrievals in parallel
        settings = load_settings()
        if not isinstance(settings, dict):
            raise ValueError(f"Settings is not a dictionary! Got: {type(settings)} with value: {settings}")
        use_context = settings.get("useContext", True)
        use_internet = settings.get("useInternet", False)
        
        context_tasks = []
        
        if use_context:
            # KB context
            kb_task = asyncio.create_task(retrieve_context_async(transcript, k=2))
            context_tasks.append(("kb", kb_task))
            
            # Chat history context
            chat_task = asyncio.create_task(retrieve_chat_history_async(
                transcript, 
                subject=CURRENT_SUBJECT[0], 
                max_exchanges=2
            ))
            context_tasks.append(("chat", chat_task))
        
        if use_internet:
            # Internet context
            internet_task = asyncio.create_task(get_perplexity_response_async(transcript))
            context_tasks.append(("internet", internet_task))
        
        # Wait for all context retrieval to complete
        contexts = {}
        for ctx_type, task in context_tasks:
            try:
                contexts[ctx_type] = await task
            except Exception as e:
                print(f"Error retrieving {ctx_type} context: {e}")
                contexts[ctx_type] = ""
        
        # Combine contexts
        combined_context = ""
        if use_context:
            kb_context = contexts.get("kb", "")
            chat_context = contexts.get("chat", "")
            
            if kb_context and chat_context:
                combined_context = f"Knowledge Base Context:\n{kb_context}\n\nChat History Context:\n{chat_context}"
            elif kb_context:
                combined_context = kb_context
            elif chat_context:
                combined_context = f"Chat History Context:\n{chat_context}"
        
        # Add internet info
        internet_info = contexts.get("internet", "")
        if use_internet and internet_info:
            if combined_context:
                combined_context += f"\n\nInternet Information: {internet_info}"
            else:
                combined_context = f"Internet Information: {internet_info}"
        
        self.active_pipelines[pipeline_id]["context"] = combined_context
        self.active_pipelines[pipeline_id]["status"] = "generating_response"
        
        # Step 3: Generate LLM responses (multiple if in manual mode)
        self._emit_update(session_id, "status_update", {"status": "Generating response..."})
        
        # Prepare messages for LLM
        settings_sys_prompt = settings.get("system", {}).get("systemPrompt", "")
        # if not settings_sys_prompt:
            # If not found in settings, use the global function
            # from app_v1_5k import system_prompt as global_system_prompt
            # settings_sys_prompt = global_system_prompt()
        
        # Add subject context if available
        subject_context = ""
        subject_description = get_subject_description(CURRENT_SUBJECT[0])
        if subject_description:
            subject_name = CURRENT_SUBJECT[0]
            subject_context = f"\n\nCurrent conversation is with: {subject_name}\nContext about this person: {subject_description}"
        
        msgs = [{"role": "system", "content": settings_sys_prompt + subject_context}]
        
        # Add context to messages
        if combined_context:
            msgs.append({"role": "system", "content": "Retrieved context:\n" + combined_context})
        
        # Add user message
        msgs.append({"role": "user", "content": transcript})
        
        # Get provider from settings
        provider = settings.get("provider", "openai")
        
        # Modified to get multiple completions if not in auto mode
        if auto_mode:
            n_completions = 1
        else:
            # Get completions count from settings, default to 4
            n_completions = settings.get("completionsCount", 4)
        
        # Generate completions
        completions = []
        try:
            # Choose the provider to use
            if provider == "groq":
                for i in range(n_completions):
                    # For Groq, generate completions with different temperatures
                    temp_adjust = i * 0.1
                    completion = await self._generate_single_completion_groq(
                        msgs, temperature=0.7 + temp_adjust
                    )
                    completions.append(completion)
                    # Emit each completion as it's ready
                    self._emit_update(session_id, "completion_ready", {
                        "index": i,
                        "text": completion
                    })
            else:
                # OpenAI - use streaming API for first completion only in auto mode
                if auto_mode:
                    # Just get one streaming completion
                    llm_queue = asyncio.Queue()
                    completion = ""
                    async for chunk in get_openai_streaming_completion(msgs):
                        if chunk:
                            completion += chunk
                            self._emit_update(session_id, "llm_chunk", {
                                "chunk": chunk,
                                "full_response_so_far": completion
                            })
                            await llm_queue.put(chunk)
                    
                    # Signal end of LLM response
                    await llm_queue.put(None)
                    completions = [completion]
                    
                    # Update pipeline record
                    if pipeline_id in self.active_pipelines:
                        self.active_pipelines[pipeline_id]["llm_responses"] = completions
                else:
                    # Get multiple non-streaming completions for manual selection
                    # Use the regular OpenAI API to get multiple completions
                    temperature = settings.get("temperature", 0.7)
                    max_tokens = settings.get("maxTokens", 150)
                    
                    url = "https://api.openai.com/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {openai.api_key}", "Content-Type": "application/json"}
                    data = {
                        "model": settings.get("openai", {}).get("model", "gpt-4.1-mini"),
                        "messages": msgs,
                        "max_tokens": max_tokens,
                        "n": n_completions,
                        "temperature": temperature,
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, json=data) as resp:
                            if resp.status != 200:
                                error_text = await resp.text()
                                print(f"OpenAI API error: {resp.status} {error_text}")
                                raise Exception(f"OpenAI API error: {error_text}")
                            
                            result = await resp.json()
                            raw_completions = [choice["message"]["content"] for choice in result["choices"]]
                            completions = clean_completions(raw_completions)
                    
                    # Emit all completions at once
                    self._emit_update(session_id, "completions_ready", {
                        "completions": completions,
                        "transcript": transcript 
                    })
                    
                    # Store completions in the pipeline record
                    if pipeline_id in self.active_pipelines:
                        self.active_pipelines[pipeline_id]["llm_responses"] = completions
                        self.active_pipelines[pipeline_id]["transcript"] = transcript
                        print(f"Stored {len(completions)} completions with transcript in pipeline {pipeline_id}")
        except Exception as e:
            print(f"Error generating completions: {e}")
            traceback.print_exc()
            self._emit_update(session_id, "status_update", {
                "status": f"Error generating response: {str(e)}"
            })
            completions = ["Error generating response."]
        
        # Step 4: In auto mode, process TTS; in manual mode, wait for selection
        if auto_mode and completions:
            # Auto mode - Generate TTS for the first completion
            self._emit_update(session_id, "status_update", {"status": "Converting to speech..."})
            
            try:
                # Process the first completion into TTS chunks
                llm_queue = asyncio.Queue()
                for chunk in completions[0].split(". "):
                    if chunk.strip():
                        await llm_queue.put(chunk + ".")
                await llm_queue.put(None)
                
                # Process TTS chunks
                tts_queue = asyncio.Queue()
                audio_urls = await self._stream_tts_audio_manual(session_id, llm_queue, tts_queue)
                
                # Add audio URLs to pipeline record
                if pipeline_id in self.active_pipelines:
                    self.active_pipelines[pipeline_id]["audio_urls"] = audio_urls
                
                # Notify client of completion
                self._emit_update(session_id, "pipeline_complete", {
                    "transcript": transcript,
                    "response": completions[0],
                    "audio_urls": audio_urls
                })
            except Exception as e:
                print(f"Error processing TTS in auto mode: {e}")
                traceback.print_exc()
                self._emit_update(session_id, "status_update", {
                    "status": f"Error converting to speech: {str(e)}"
                })
        else:
            # Manual mode - Just notify that completions are ready to select from
            self._emit_update(session_id, "pipeline_status", {
                "status": "completions_ready",
                "transcript": transcript,
                "completions": completions
            })
        
    async def _stream_llm_response(self, provider, messages, session_id, llm_queue):
        """Stream LLM response chunks to the client and the TTS queue"""
        try:
            full_response = ""
            
            # Choose the appropriate streaming function based on provider
            stream_func = get_openai_streaming_completion
            if provider == "groq":
                stream_func = get_groq_streaming_completion
            
            # Get LLM chunks
            async for chunk in stream_func(messages):
                if chunk:
                    full_response += chunk
                    
                    # Send chunk to client
                    self._emit_update(session_id, "llm_chunk", {
                        "chunk": chunk,
                        "full_response_so_far": full_response
                    })
                    
                    # Add to queue for TTS
                    await llm_queue.put(chunk)
            
            # Signal end of LLM response
            await llm_queue.put(None)
            
            # Make sure we update the pipeline record
            pipeline_id = session_id
            for key in self.active_pipelines:
                if key.startswith(session_id):
                    pipeline_id = key
                    break
                    
            if pipeline_id in self.active_pipelines:
                self.active_pipelines[pipeline_id]["llm_response"] = full_response
            else:
                print(f"Warning: Pipeline {pipeline_id} not found when updating LLM response")
        
        except Exception as e:
            print(f"Error in _stream_llm_response: {e}")
            print(traceback.format_exc())
            
            # Signal error to the client
            self._emit_update(session_id, "status_update", {
                "status": f"Error generating response: {str(e)}"
            })
            
            # Signal end of LLM response to prevent hanging
            await llm_queue.put(None)
    
    async def _stream_tts_audio(self, session_id, llm_queue, tts_queue):
        """Process LLM chunks into TTS audio in real-time"""
        buffer = ""
        audio_urls = []
        chunk_counter = 0
        min_chars_for_tts = 40  # Minimum characters to process for TTS
        max_chars_for_tts = 150  # Maximum characters to process in one TTS request
        
        try:
            # Process LLM chunks as they arrive
            while True:
                chunk = await llm_queue.get()
                if chunk is None:
                    # End of LLM response
                    break
                
                buffer += chunk
                
                # Only process when we have enough text or at good break points
                process_now = (
                    (len(buffer) >= min_chars_for_tts and any(mark in buffer for mark in ['.', '!', '?', ',', ';', ':'])) 
                    or len(buffer) >= max_chars_for_tts
                )
                
                if process_now:
                    # Find a good break point
                    break_points = ['.', '!', '?', ',', ';', ':']
                    break_index = -1
                    
                    for mark in break_points:
                        pos = buffer.rfind(mark)
                        if pos > min_chars_for_tts // 2:  # Make sure we have a reasonable chunk size
                            break_index = pos + 1
                            break
                    
                    # If no good break point found
                    if break_index == -1:
                        if len(buffer) > max_chars_for_tts:
                            # If buffer is too large, break at a space after min_chars
                            spaces = [i for i, c in enumerate(buffer) if c == ' ' and i >= min_chars_for_tts]
                            if spaces:
                                break_index = spaces[0] + 1
                            else:
                                # Worst case: just use the whole buffer
                                break_index = len(buffer)
                        else:
                            # If buffer is still reasonable size, continue collecting
                            continue
                    
                    # Extract text to process
                    text_to_process = buffer[:break_index].strip()
                    buffer = buffer[break_index:].strip()
                    
                    if not text_to_process:
                        continue
                    
                    # Process TTS for this chunk
                    try:
                        # Start TTS generation for this chunk
                        chunk_counter += 1
                        print(f"Processing TTS chunk {chunk_counter}: '{text_to_process[:30]}...'")
                        
                        # Generate TTS but ensure filename is unique for each chunk
                        ext = "mp3"  # Default extension
                        timestamp = int(time.time())
                        import uuid
                        unique_id = f"{chunk_counter}_{uuid.uuid4().hex[:8]}"
                        
                        # Call speak_text_eleven with the chunk
                        url = speak_text_eleven(text_to_process)
                        
                        if url:
                            # Verify URL is not a duplicate before adding
                            if url not in audio_urls:
                                audio_urls.append(url)
                                # Send audio URL to client
                                self._emit_update(session_id, "tts_chunk_ready", {
                                    "audio_url": url,
                                    "chunk_index": chunk_counter,
                                    "text": text_to_process
                                })
                            else:
                                print(f"Warning: Duplicate audio URL detected: {url}")
                    except Exception as e:
                        print(f"Error in TTS processing: {e}")
                        print(traceback.format_exc())
                
            # Process any remaining text
            if buffer and len(buffer.strip()) > 10:  # Only process if there's meaningful text left
                try:
                    chunk_counter += 1
                    print(f"Processing final TTS chunk: '{buffer[:30]}...'")
                    url = speak_text_eleven(buffer)
                    if url:
                        audio_urls.append(url)
                        self._emit_update(session_id, "tts_chunk_ready", {
                            "audio_url": url,
                            "chunk_index": chunk_counter,
                            "text": buffer,
                            "is_last": True
                        })
                except Exception as e:
                    print(f"Error in final TTS processing: {e}")
                    print(traceback.format_exc())
            
            # Update the pipeline with all audio URLs
            pipeline_id = session_id
            for key in self.active_pipelines:
                if key.startswith(session_id):
                    pipeline_id = key
                    break
                    
            if pipeline_id in self.active_pipelines:
                self.active_pipelines[pipeline_id]["audio_urls"] = audio_urls
            else:
                print(f"Warning: Pipeline {pipeline_id} not found when updating audio URLs")
        
        except Exception as e:
            print(f"Error in _stream_tts_audio: {e}")
            print(traceback.format_exc())
            
            # Signal error to the client
            self._emit_update(session_id, "status_update", {
                "status": f"Error processing TTS: {str(e)}"
            })
    async def _stream_tts_audio_manual(self, session_id, llm_queue, tts_queue):
        """Process TTS for a manually selected completion"""
        audio_urls = []
        chunk_counter = 0
        
        try:
            # Process text chunks from the queue
            while True:
                chunk = await llm_queue.get()
                if chunk is None:
                    break
                    
                if not chunk.strip():
                    continue
                    
                # Process TTS for this chunk
                try:
                    chunk_counter += 1
                    print(f"Processing TTS chunk {chunk_counter}: '{chunk[:30]}...'")
                    
                    # Generate TTS for the chunk
                    url = speak_text_eleven(chunk)
                    
                    if url:
                        audio_urls.append(url)
                        # Send audio URL to client
                        self._emit_update(session_id, "tts_chunk_ready", {
                            "audio_url": url,
                            "chunk_index": chunk_counter,
                            "text": chunk
                        })
                except Exception as e:
                    print(f"Error in TTS processing: {e}")
                    print(traceback.format_exc())
                    
        except Exception as e:
            print(f"Error in _stream_tts_audio_manual: {e}")
            print(traceback.format_exc())
            
        return audio_urls
        
##############################################################################
# GLOBAL STATE VARIABLES
##############################################################################

last_user_text = [""]
flow_mode = ["subject"]  # "bernard" if Bernard spoke last, "subject" otherwise
## Global audio recorder instance ##
recorder = SmartAudioRecorder()

##############################################################################
# START REVERSE PROXY
##############################################################################

def start_docker_reverse_proxy():
    print("⚠️  Skipping reverse proxy setup - not required for basic functionality")

def is_container_running(name):
    return False

# Skip reverse proxy setup
print("⚠️  Skipping reverse proxy setup - not required for basic functionality")

##############################################################################
# FLASK APP SETUP
##############################################################################

from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
AUDIO_FOLDER = os.path.join(STATIC_FOLDER, "temp_audio")
os.makedirs(AUDIO_FOLDER, exist_ok=True)  # Ensure audio folder exists

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path="")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Print MAX_CONTENT_LENGTH at startup
print("MAX_CONTENT_LENGTH:", app.config.get('MAX_CONTENT_LENGTH'))

# Initialize SocketIO with your Flask app
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variable to track active clients
active_clients = set()

##############################################################################
# SYSTEM PROMPT
##############################################################################

def system_prompt():
    return (
        "You are the persona Bernard Muller\n\n"
        "1. Personal background:\n"
        "   - Born August 16, 1968 in Dordrecht\n"
        "   - Married to Antoinette Bunt\n"
        "   - Has two sons named Joost and Emiel, ages 28 and 27 both living in London\n"
        "   - Lives in Willemstad at Landpoortstraat 22, with Antoinette still in Portugal\n"
        "   - Wheelchair bound (since 2016), uses a feeding tube and depending on a ventilator (since 2022)\n\n"
        "2. Professional background:\n"
        "   - Entrepreneur who was diagnosed with ALS in 2010\n"
        "   - Before diagnosis chairman of Burando Maritime Services in the port of Rotterdam\n"
        "   - Founded Treeway, a biotech company Together with Robbert Jan Stuit\n"
        "   - Co-founder of Project MinE (genome research project)\n"
        "   - Co-founder of ALS Investment Fund\n"
        "   - Received the Officers Order of Orange-Nassau (royal distinction) in November 2024\n\n"
        "3. Personality and interests:\n"
        "   - Curious and creative\n"
        "   - Passionate about technology, art, business, sports\n"
        "   - Specific interests include Feyenoord (football), golf, tennis, Formula 1\n"
        "   - Hobbies: stock market, creating art, artificial intelligence\n\n"
        "4. Communication style:\n"
        "   - Bernard Uses eye-tracking technology to communicate\n"
        "   - respond Friendly, sometimes sarcastic and precise with humor\n"
        "   - Use an accessible, informal tone\n"
        "   - May swear from time to time \n"
        "   - Give direct, factual answers\n"
        "   - Avoid excessive formality\n\n"
        "5. Prompt and chat completion instructions\n"
        "  - Please keep your responses to a maximum of one sentence\n"
        "  - For each completion in a set of multiple completions, adopt a distinctly different tone, perspective, or approach\n"
        "  - Make sure each response offers unique information or a unique angle on the same information\n"
        "  - When answering binary questions (e.g., yes/no), Ensure that your answers include both 'yes' and 'no' options, maintaining diversity in the responses."
    )

##############################################################################
# SUBJECT MANAGEMENT FUNCTIONS - UPDATED FOR DESCRIPTIONS
##############################################################################

def load_subjects():
    """Load subjects from the subjects file."""
    try:
        if os.path.exists(SUBJECTS_FILE):
            with open(SUBJECTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("subjects", ["Default Subject"])
        return ["Default Subject"]
    except Exception as e:
        print(f"Error loading subjects: {e}")
        return ["Default Subject"]

def save_subjects(subjects):
    """Save the list of subject objects to the subjects file."""
    try:
        os.makedirs(os.path.dirname(SUBJECTS_FILE), exist_ok=True)
        with open(SUBJECTS_FILE, "w", encoding="utf-8") as f:
            json.dump({"subjects": subjects}, f, indent=2)
        print(f"Saved subjects to {SUBJECTS_FILE}")
    except Exception as e:
        print(f"Error saving subjects: {e}")

def get_subject_description(subject_name):
    """
    Get the description for a specific subject.
    Returns empty string if the subject is not found or has no description.
    """
    subjects = load_subjects()
    for subject in subjects:
        if isinstance(subject, dict):
            if subject.get("name") == subject_name:
                return subject.get("description", "")
        elif isinstance(subject, str):
            if subject == subject_name:
                return ""
    return ""

def update_current_subject(subject_name, description=None):
    """
    Update the current subject and optionally its description.
    If the subject doesn't exist, it will be created.
    """
    subjects = load_subjects()
    
    # Check if subject exists
    subject_exists = False
    for subject in subjects:
        if subject.get("name") == subject_name:
            subject_exists = True
            # Update description if provided
            if description is not None:
                subject["description"] = description
            break
    
    # Add new subject if it doesn't exist
    if not subject_exists:
        subjects.append({"name": subject_name, "description": description or ""})
    
    # Save changes
    save_subjects(subjects)
    
    # Update global current subject
    CURRENT_SUBJECT[0] = subject_name
    
    return True

##############################################################################
# CONVERSATION STARTERS & ENDERS MANAGEMENT FUNCTIONS
##############################################################################

def load_phrases(phrase_type='all'):
    """Load conversation phrases from the phrases file."""
    try:
        if os.path.exists(PHRASES_FILE):
            with open(PHRASES_FILE, "r", encoding="utf-8") as f:
                phrases = json.load(f)
                if phrase_type == 'all':
                    return phrases
                return phrases.get(phrase_type, [])
        return []
    except Exception as e:
        print(f"Error loading phrases: {e}")
        return []

def save_phrases(phrases):
    """Save the phrases dictionary to the phrases file."""
    try:
        os.makedirs(os.path.dirname(PHRASES_FILE), exist_ok=True)
        with open(PHRASES_FILE, "w", encoding="utf-8") as f:
            json.dump(phrases, f, indent=2)
        print(f"Saved phrases to {PHRASES_FILE}")
    except Exception as e:
        print(f"Error saving phrases: {e}")

def update_phrase(phrase_type, old_text, new_text):
    """
    Update or add a conversation phrase.
    
    Parameters:
    - phrase_type: 'starter' or 'ender'
    - old_text: Original text (empty string if adding new)
    - new_text: New text to save
    
    Returns:
    - Boolean indicating success
    """
    if phrase_type not in ['starter', 'ender']:
        print(f"Invalid phrase type: {phrase_type}")
        return False
        
    # Load current phrases
    phrases = load_phrases()
    
    # Determine the list to modify
    phrase_list = phrases["starters"] if phrase_type == 'starter' else phrases["enders"]
    
    # If old_text is empty, this is a new addition
    if not old_text:
        # Check if it already exists to avoid duplicates
        if new_text not in phrase_list:
            phrase_list.append(new_text)
    else:
        # Find and replace the existing text
        try:
            index = phrase_list.index(old_text)
            phrase_list[index] = new_text
        except ValueError:
            # If not found, add as new
            phrase_list.append(new_text)
    
    # Update the dictionary
    if phrase_type == 'starter':
        phrases["starters"] = phrase_list
    else:
        phrases["enders"] = phrase_list
    
    # Save to file
    return save_phrases(phrases)
        
##############################################################################
# KNOWLEDGE BASE & VECTOR STORE FUNCTIONS
##############################################################################
    
def load_documents():
    """
    Load documents from the KB directory.
    This function handles memories.json, JSONL files (which combine user and assistant messages into one document),
    regular text files (excluding chat_history.txt), PDF files (using PyPDFLoader with fallback), and other files.
    """
    docs = []
    loaded_files = []
    
    for f in glob.glob(os.path.join(KB_DIR, "*")):
        try:
            # Special handling for memories.json
            if f.lower().endswith(".json") and os.path.basename(f).lower() == "memories.json":
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        memories_data = json.load(file)
                        memories = memories_data.get("memories", [])
                        for memory in memories:
                            title = memory.get("title", "")
                            text = memory.get("text", "")
                            date = memory.get("date", "")
                            tags = memory.get("tags", [])
                            content = f"Memory - Title: {title}\nDate: {date}\nTags: {', '.join(tags)}\n\n{text}"
                            if content.strip():
                                docs.append(SimpleDoc(
                                    page_content=content,
                                    metadata={"source": f, "memory_id": str(id(memory)), "date": date}
                                ))
                                loaded_files.append(f"memory: {title} ({date})")
                except Exception as e:
                    print(f"Error loading memories.json: {e}")
            
            # Special handling for JSONL files
            elif f.lower().endswith(".jsonl"):
                try:
                    with open(f, "r", encoding="utf-8") as file:
                        line_count = 0
                        for line in file:
                            line_count += 1
                            try:
                                conversation = json.loads(line)
                                if "messages" in conversation:
                                    messages = conversation["messages"]
                                    content_parts = []
                                    for msg in messages:
                                        role = msg.get("role", "").lower()
                                        # Skip system messages (which might contain internal instructions)
                                        if role == "system":
                                            continue
                                        elif role == "user":
                                            content_parts.append("User: " + msg.get("content", ""))
                                        elif role == "assistant":
                                            content_parts.append("Assistant: " + msg.get("content", ""))
                                        else:
                                            content_parts.append(msg.get("content", ""))
                                    if content_parts:
                                        # Join with double newlines so that Q&A remain together
                                        combined = "\n\n".join(content_parts)
                                        doc_id = f"jsonl_{os.path.basename(f)}_{uuid.uuid4().hex[:8]}"
                                        docs.append(SimpleDoc(
                                            page_content=combined,
                                            metadata={"source": f, "doc_id": doc_id, "type": "conversation"}
                                        ))
                            except json.JSONDecodeError as je:
                                print(f"Error parsing JSON line {line_count} in {f}: {je}")
                                continue
                    loaded_files.append(f"{os.path.basename(f)} ({line_count} conversations)")
                    print(f"Successfully loaded JSONL file: {os.path.basename(f)} with {line_count} entries")
                except Exception as e:
                    print(f"Error loading JSONL file {f}: {e}")
                    print(traceback.format_exc())
            
            # Regular text files (excluding chat_history.txt)
            elif f.lower().endswith(".txt") and not f.lower().endswith("chat_history.txt"):
                with open(f, "r", encoding="utf-8") as file:
                    content = file.read()
                    if content.strip():
                        docs.append(SimpleDoc(page_content=content, metadata={"source": f}))
                        loaded_files.append(os.path.basename(f))
            
            # PDF files
            elif f.lower().endswith(".pdf"):
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(f)
                    pdf_docs = loader.load()
                    if pdf_docs:
                        docs.extend(pdf_docs)
                        loaded_files.append(os.path.basename(f))
                        print(f"Successfully loaded PDF: {os.path.basename(f)} with PyPDFLoader")
                except Exception as pdf_error:
                    print(f"Error loading PDF with PyPDFLoader: {pdf_error}")
                    print(traceback.format_exc())
                    try:
                        with open(f, "rb") as pdf_file:
                            content = pdf_file.read(10240).decode('utf-8', errors='ignore')
                            docs.append(SimpleDoc(page_content=content, metadata={"source": f}))
                            loaded_files.append(os.path.basename(f) + " (text fallback)")
                            print(f"Loaded PDF as text fallback: {os.path.basename(f)}")
                    except Exception as fallback_error:
                        print(f"Failed fallback for PDF: {fallback_error}")
            
            # Fallback for other document types not matching PDF, TXT, JSON, or JSONL
            else:
                if not f.lower().endswith((".txt", ".pdf", ".json", ".jsonl")):
                    from langchain_unstructured import UnstructuredLoader
                    other_docs = UnstructuredLoader(f).load()
                    if other_docs:
                        docs.extend(other_docs)
                        loaded_files.append(os.path.basename(f))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    print(f"Loaded {len(docs)} documents from {len(loaded_files)} files: {', '.join(loaded_files)}")
    return docs

    
def build_vector_store():
    # Instead of importing from a separate module, directly call the locally defined load_documents()
    docs = load_documents()
    if not docs:
        print("No documents in KB.")
        return None

    # Separate conversation documents (from JSONL) from other documents.
    conversation_docs = [doc for doc in docs if doc.metadata.get("type") == "conversation"]
    other_docs = [doc for doc in docs if doc.metadata.get("type") != "conversation"]

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # For conversation documents, use a larger chunk size
    conv_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    # For other documents, use default splitting parameters
    default_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    conv_chunks = conv_splitter.split_documents(conversation_docs) if conversation_docs else []
    other_chunks = default_splitter.split_documents(other_docs) if other_docs else []
    all_chunks = conv_chunks + other_chunks

    print(f"Processing {len(all_chunks)} chunks from {len(docs)} documents")

    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    api_key = os.environ.get("OPENAI_API_KEY") or openai.api_key
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set; please set the environment variable or pass the API key.")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    store = FAISS.from_documents(all_chunks, embeddings)

    if store.index.ntotal == 0:
        print("Vector store empty.")
        return None

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    store.save_local(VECTOR_STORE_DIR)
    print(f"Vector store built with {store.index.ntotal} vectors")
    return store

# Replace the load_vector_store function in app v1_5b with this version:
def load_vector_store():
    """
    Load the vector store from disk or build it if it doesn't exist.
    This function uses OpenAI embeddings for the main KB vector store.
    """
    if os.path.exists(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR):
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
            store = FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded existing vector store with {store.index.ntotal} vectors")
            return store
        except Exception as e:
            print("Error loading vector store:", e)
            print("Rebuilding vector store from scratch...")
            return build_vector_store()
    else:
        print("Vector store not found. Building from scratch...")
        return build_vector_store()

def retrieve_context(query, k=2, similarity_threshold=0.1):
    """
    Get relevant context from vector store using the query.
    Now includes a similarity threshold to filter out low-relevance results.
    
    Parameters:
    - query: The search query text
    - k: Number of documents to retrieve
    - similarity_threshold: Minimum similarity score (0-1) to include a document
    
    Returns:
    - String of relevant context, or empty string if nothing meets the threshold
    """
    if not query.strip():
        print("Empty query, no context retrieved")
        return ""
        
    store = load_vector_store()
    if not store:
        print("No vector store available.")
        return ""
    
    try:
        # Use FAISS's scored retriever to get similarity scores
        # This returns (document, score) tuples
        docs_with_scores = store.similarity_search_with_score(query, k=k)
        
        # Filter out results below the threshold
        filtered_docs = [(doc, score) for doc, score in docs_with_scores if score >= similarity_threshold]
        
        if not filtered_docs:
            print(f"No documents met the similarity threshold of {similarity_threshold}")
            return ""
            
        # Log what we're retrieving with scores for debugging
        for i, (doc, score) in enumerate(filtered_docs):
            source = doc.metadata.get("source", "unknown")
            print(f"Retrieved doc {i+1}: Score={score:.4f}, Source={source}")
            print(f"  Preview: {doc.page_content[:50]}...")
        
        # Format context from the filtered documents
        context = "\n\n".join(doc.page_content for doc, _ in filtered_docs)
        print(f"Retrieved context for '{query[:30]}...': {len(context)} chars, {len(filtered_docs)} documents")
        return context
    except Exception as e:
        print(f"Error retrieving context: {e}")
        print(traceback.format_exc())
        return ""

##############################################################################
# CHAT VECTOR STORE FUNCTIONS
##############################################################################

def build_chat_vector_store(incremental=True):
    """
    Build or update the vector store for chat history using local embeddings.
    
    Parameters:
    - incremental: If True, adds only new exchanges. If False, rebuilds from scratch.
    
    Returns:
    - FAISS vector store object
    """
    # Initialize the JSON file if needed
    initialize_chat_history()
    
    # Load existing store if we're doing an incremental update
    chat_store = None
    if incremental and os.path.exists(CHAT_VECTOR_STORE_DIR):
        try:
            # Use local embeddings instead of OpenAI
            embeddings = LocalEmbeddings()
            chat_store = FAISS.load_local(CHAT_VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded existing chat vector store with {chat_store.index.ntotal} vectors")
        except Exception as e:
            print(f"Error loading chat vector store: {e}")
            print("Will rebuild the chat vector store from scratch")
            incremental = False
    
    # Get chat history from JSON
    with open(CHAT_HISTORY_JSON, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    # Process all exchanges, or just new ones if incremental
    new_docs = []
    updated = False
    
    # We'll track which exchanges we've processed
    for conv in history.get("conversations", []):
        subject = conv.get("subject", "Unknown")
        for exchange in conv.get("exchanges", []):
            # Check if this exchange has been vectorized already
            if incremental and exchange.get("vectorized", False):
                continue
                
            # Combine the user and bernard text for embedding
            combined_text = f"User: {exchange.get('user', '')}\nBernard: {exchange.get('bernard', '')}"
            
            # Create a unique ID for this exchange if it doesn't have one
            exchange_id = exchange.get("id", str(uuid.uuid4()))
            if "id" not in exchange:
                exchange["id"] = exchange_id
            
            # Create a document for the vector store
            from langchain.schema import Document
            doc = Document(
                page_content=combined_text,
                metadata={
                    "subject": subject,
                    "timestamp": exchange.get("timestamp", ""),
                    "exchange_id": exchange_id,
                    "type": "chat_history"
                }
            )
            new_docs.append(doc)
            
            # Mark as vectorized in the original data
            exchange["vectorized"] = True
            updated = True
    
    # If no new documents to add, just return the existing store
    if not new_docs:
        print("No new chat history to vectorize")
        return chat_store
    
    print(f"Adding {len(new_docs)} new chat exchanges to vector store")
    
    # Create embeddings for the new documents using local model
    embeddings = LocalEmbeddings()
    
    if chat_store and incremental:
        # Add to existing store
        chat_store.add_documents(new_docs)
    else:
        # Create new store from scratch
        chat_store = FAISS.from_documents(new_docs, embeddings)
    
    # Save the updated store
    os.makedirs(CHAT_VECTOR_STORE_DIR, exist_ok=True)
    chat_store.save_local(CHAT_VECTOR_STORE_DIR)
    
    # Save the updated history with vectorized flags
    if updated:
        with open(CHAT_HISTORY_JSON, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    print(f"Chat vector store built/updated with {chat_store.index.ntotal} vectors")
    return chat_store

def load_chat_vector_store():
    """
    Load the chat history vector store or build it if it doesn't exist.
    Uses local embeddings instead of OpenAI.
    """
    if os.path.exists(CHAT_VECTOR_STORE_DIR) and os.listdir(CHAT_VECTOR_STORE_DIR):
        try:
            # Use local embeddings instead of OpenAI
            embeddings = LocalEmbeddings()
            store = FAISS.load_local(CHAT_VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded chat history vector store with {store.index.ntotal} vectors")
            return store
        except Exception as e:
            print("Error loading chat history vector store:", e)
            print("Rebuilding chat history vector store...")
            return build_chat_vector_store(incremental=False)
    else:
        print("Chat history vector store not found. Building from scratch...")
        return build_chat_vector_store(incremental=False)

def retrieve_chat_history(query, subject=None, days=None, max_exchanges=2, similarity_threshold=0.3, search_all_subjects=False):
    """
    Unified function to retrieve chat history using vector search.
    
    Parameters:
    - query: The search query text
    - subject: Specific subject to filter by (uses current subject if None)
    - days: If provided, only return exchanges from the last X days
    - max_exchanges: Maximum number of exchanges to return
    - similarity_threshold: Minimum similarity score (0-1) to include an exchange
    - search_all_subjects: If True, search across all subjects, ignoring the subject parameter
    
    Returns:
    - String of relevant chat history, or empty string if nothing found
    """
    # Load the chat vector store
    store = load_chat_vector_store()
    if not store:
        print("No chat vector store available")
        return ""
    
    # Use current subject if not specified and not searching all subjects
    if not search_all_subjects:
        if not subject:
            subject = CURRENT_SUBJECT[0]
    
    # Create filter parameters
    filter_dict = {}
    
    # Filter by subject only if not searching all subjects
    if not search_all_subjects and subject != "Default Subject":
        filter_dict["subject"] = subject
    
    # Filter by date if requested
    if days:
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        filter_dict["timestamp"] = {"$gte": cutoff_date}
    
    try:
        # Get documents with similarity scores
        docs_with_scores = store.similarity_search_with_score(
            query, 
            k=max_exchanges * 2,  # Get more, then filter
            filter=filter_dict
        )
        
        # Filter by similarity threshold
        filtered_docs = [(doc, score) for doc, score in docs_with_scores if score >= similarity_threshold]
        
        if not filtered_docs:
            print(f"No chat history met the similarity threshold of {similarity_threshold}")
            return ""
        
        # Sort by similarity (highest first) and limit to max_exchanges
        filtered_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)[:max_exchanges]
        
        # Log what we're retrieving
        for i, (doc, score) in enumerate(filtered_docs):
            subject = doc.metadata.get("subject", "unknown")
            timestamp = doc.metadata.get("timestamp", "unknown time")
            print(f"Retrieved chat {i+1}: Score={score:.4f}, Subject={subject}, Time={timestamp}")
            print(f"  Preview: {doc.page_content[:50]}...")
        
        # Format context from the documents
        context = "\n\n".join(doc.page_content for doc, _ in filtered_docs)
        print(f"Retrieved chat history for '{query[:30]}...': {len(context)} chars, {len(filtered_docs)} exchanges")
        return context
        
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        print(traceback.format_exc())
        return ""

##############################################################################
# CHAT HISTORY FUNCTIONS
##############################################################################

def initialize_chat_history():
    """
    Initialize the chat history JSON file if it doesn't exist
    """
    if not os.path.exists(CHAT_HISTORY_JSON):
        history = {
            "conversations": []
        }
        with open(CHAT_HISTORY_JSON, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"Created new chat history file at {CHAT_HISTORY_JSON}")
    else:
        # Validate the existing file
        try:
            with open(CHAT_HISTORY_JSON, 'r', encoding='utf-8') as f:
                history = json.load(f)
            if "conversations" not in history:
                history["conversations"] = []
                with open(CHAT_HISTORY_JSON, 'w', encoding='utf-8') as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
                print(f"Fixed chat history file structure at {CHAT_HISTORY_JSON}")
        except Exception as e:
            print(f"Error validating chat history file: {e}")
            # Create a new file if there's an error
            history = {
                "conversations": []
            }
            with open(CHAT_HISTORY_JSON, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            print(f"Created new chat history file at {CHAT_HISTORY_JSON}")
    
    return True

def save_chat_exchange_json(user_text, bernard_text):
    """
    Save the chat exchange to the JSON history file
    """
    try:
        # Ensure the chat history file exists
        initialize_chat_history()
        
        # Load current history
        with open(CHAT_HISTORY_JSON, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        # Get current subject - make sure we're using the global variable
        current_subject = CURRENT_SUBJECT[0]
        print(f"Saving chat exchange with subject: '{current_subject}'")
        current_time = datetime.now(timezone.utc).isoformat()
        
        # Get subject description
        subject_description = get_subject_description(current_subject)
        
        # Check if user_text contains Bernard: and User: prefixes, and extract just the user part
        if "Bernard:" in user_text and "User:" in user_text:
            # Extract just the user's text
            user_parts = user_text.split("User:")
            if len(user_parts) > 1:
                user_text = user_parts[1].strip()
        
        # Check if we have an active conversation for this subject
        # We'll consider a conversation active if it's the same subject and less than 1 hour old
        current_conversation = None
        for conv in history["conversations"]:
            if conv["subject"] == current_subject:
                # Check if the last exchange was recent (within 1 hour)
                if conv["exchanges"] and len(conv["exchanges"]) > 0:
                    last_exchange_time = conv["exchanges"][-1]["timestamp"]
                    # Simple time comparison - could be more sophisticated
                    if current_time.split('T')[0] == last_exchange_time.split('T')[0]:  # Same day
                        current_conversation = conv
                        break
        
        # If no active conversation found, create a new one
        if not current_conversation:
            current_conversation = {
                "id": f"conv_{uuid.uuid4().hex[:8]}",
                "subject": current_subject,  # Make sure this is the correct subject
                "subject_description": subject_description,  # Add subject description
                "timestamp": current_time,
                "exchanges": []
            }
            history["conversations"].append(current_conversation)
        
        # Add the new exchange with a vectorized flag set to False
        exchange = {
            "user": user_text,
            "bernard": bernard_text,
            "timestamp": current_time,
            "vectorized": False  # Add this flag to track which exchanges are in the vector store
        }
        current_conversation["exchanges"].append(exchange)
        
        # Update the conversation timestamp
        current_conversation["timestamp"] = current_time
        
        # Save the updated history
        with open(CHAT_HISTORY_JSON, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        print(f"Saved chat exchange to JSON history, subject: {current_subject}")
        return True
    
    except Exception as e:
        print(f"Error saving chat exchange to JSON: {e}")
        return False
    
##############################################################################
# AUDIO & TTS FUNCTIONS (SERVER-SIDE)
##############################################################################

def transcribe_audio_eleven(audio_data):
    if not audio_data:
        return ""
        
    # Handle different types of audio data
    if isinstance(audio_data, bytes):
        # Create a properly formatted WAV file in memory
        import io
        import wave
        
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for paInt16
            wf.setframerate(16000)  # Assumes 16kHz sample rate
            wf.writeframes(audio_data)
        
        # Get the WAV bytes
        wav_io.seek(0)
        wav_bytes = wav_io.read()
    else:
        # Assume it's an AudioData object from speech_recognition
        wav_bytes = audio_data.get_wav_data()
        
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {"model_id": "scribe_v1", "language_code": None}
    
    try:
        resp = requests.post(url, headers=headers, data=data, files=files)
        resp.raise_for_status()
        return resp.json().get("text", "")
    except Exception as e:
        print(f"Error in transcribe_audio_eleven: {e}")
        return ""

def speak_text_eleven(text):
    """
    Calls ElevenLabs TTS API, writes the resulting MP3 to a permanent folder under the static folder,
    and returns the URL to the audio file. This lets the client play the audio asynchronously.
    """
    if not text.strip():
        return ""
    settings = load_settings_from_file()
    voice_settings = settings['voice']
    voice_id = voice_settings.get('voiceId', 'CwhRBWXzGAHq8TQ4Fs17')
    model_id = voice_settings.get('model', 'eleven_multilingual_v2')
    output_format = voice_settings.get('outputFormat', 'mp3_44100_192')
    seed = int(voice_settings.get('seed', 8675309))
    voice_params = voice_settings.get('voiceSettings', {
        'stability': 0.5,
        'similarity_boost': 0.75,
        'speed': 0.9,
        'style': 0.3
    })
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": model_id,
        "output_format": output_format,
        "seed": seed,
        "voice_settings": voice_params
    }
    print(f"Using TTS model: {model_id}")
    response = requests.post(tts_url, headers=headers, json=payload)
    if response.status_code == 200:
        ext = "mp3" if output_format.startswith("mp3") else "wav"
        timestamp = int(time.time())
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        filename = f"tts_audio_{timestamp}_{unique_id}.{ext}"
        filepath = os.path.join(AUDIO_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(response.content)
        print("TTS file saved at:", filepath)
        return f"/temp_audio/{filename}"
    else:
        raise Exception(f"TTS error: {response.status_code} {response.text}")

      
##############################################################################
# CHAT COMPLETIONS & HISTORY FUNCTIONS
##############################################################################

def get_perplexity_response(query):
    """
    Send a query to Perplexity API to get internet-based information
    """
    # System prompt for Perplexity
    system_prompt = (
        "Je bent de co-assistent van Bernard en helpt hem om actuele zaken op internet op te zoeken. "
        "Als er in de prompt query irrelevante niet zoekbare dingen staan vermeld je dat in de chat completion. "
        "Antwoord kort, bondig en accuraat in maximaal 1 zin."
    )
    
    # Parameters for the Perplexity API
    llm_params = {
        "model": "sonar",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.95,
        "search_domain_filter": None,
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": None,
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "response_format": None,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending query to Perplexity: {query[:50]}...")
        response = requests.post(PERPLEXITY_API_URL, json=llm_params, headers=headers)
        response.raise_for_status()
        data = response.json()
        # Extract the response content
        internet_info = data.get("choices", [{}])[0].get("message", {}).get("content", "No information found.")
        print(f"Received Perplexity response: {internet_info[:50]}...")
        return internet_info
    except Exception as e:
        print(f"Error querying Perplexity API: {str(e)}")
        return f"Error retrieving information from internet: {str(e)}"

def get_chat_completion(messages, max_tokens=None, n=None, temperature=None):
    try:
        # Load settings
        settings = load_settings_from_file()
        if not isinstance(settings, dict):
            raise ValueError(f"Settings is not a dictionary! Got: {type(settings)} with value: {settings}")
        llm_settings = settings.get("llm", {})
        system_settings = settings.get("system", {})  # Add this line to get system settings
        
        # Get the system prompt from settings
        prompt_from_settings = system_settings.get("systemPrompt", system_prompt())  # Use hardcoded as fallback
        
        # Replace the system message with the one from settings
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] = prompt_from_settings  # Replace with settings prompt
        else:
            # If no system message exists, add one
            messages.insert(0, {"role": "system", "content": prompt_from_settings})
        
        # Get provider from settings
        provider = settings.get("provider", "openai")
        
        # Get common settings or use defaults/parameters
        max_tokens = max_tokens or settings.get("maxTokens", 150)
        n = n or settings.get("completionsCount", 4)
        temperature = temperature or settings.get("temperature", 0.9)
        
        # Get subject context if available
        subject_context = ""
        subject_description = get_subject_description(CURRENT_SUBJECT[0])
        
        # Detect language of the user query (last message)
        input_language = "English"  # Default
        if messages and len(messages) > 0:
            user_messages = [msg for msg in messages if msg.get('role') == 'user']
            if user_messages:
                last_user_message = user_messages[-1].get('content', '')
                try:
                    if detect:  # Check if langdetect is available
                        detected_lang = detect(last_user_message)
                        if detected_lang == "nl":
                            input_language = "Dutch"
                except:
                    pass  # Fall back to English if detection fails
        
        # Add subject description to the system prompt if available
        if subject_description and messages and messages[0]['role'] == 'system':
            subject_name = CURRENT_SUBJECT[0]
            subject_context = f"\n\nCurrent conversation is with: {subject_name}\nContext about this person: {subject_description}"
            # Add as part of system context
            messages[0]['content'] += subject_context
            print(f"Added subject context to system prompt: {subject_context}")
        
        # Add instruction to not include "Bernard:" prefix in responses
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] += "\n\nIMPORTANT: Do not include 'Bernard:' or any other speaker prefix in your responses. Just provide the direct response as Bernard."
            messages[0]['content'] += f"\n\nIMPORTANT: The user is speaking in {input_language}. You MUST respond in the SAME LANGUAGE ({input_language})."
        
        # Get completions based on selected provider
        if provider == "openai":
            return get_openai_completion(messages, max_tokens, n, temperature, settings)
        elif provider == "groq":
            return get_groq_completion(messages, max_tokens, n, temperature, settings)
        else:
            print(f"Unknown provider: {provider}, falling back to OpenAI")
            return get_openai_completion(messages, max_tokens, n, temperature, settings)
    except Exception as e:
        print(f"Error in get_chat_completion: {e}")
        print(traceback.format_exc())
        # Always return a list of error messages, not a string
        n = n or 4
        return [f"Error getting response: {str(e)}"] * n

def get_openai_completion(messages, max_tokens, n, temperature, settings):
    """
    Get completions from the OpenAI API.
    """
    # Get OpenAI settings
    openai_settings = settings.get("openai", {})
    model = openai_settings.get("model", "gpt-4.1-mini")
    api_key = openai_settings.get("apiKey", openai.api_key)
    
    # Call OpenAI API
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "n": n,
        "temperature": temperature,
        "top_p": 0.95,
        "presence_penalty": 0.8,
        "frequency_penalty": 0.8
    }
    
    print(f"Sending request to OpenAI with model {model}")
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()
    
    # Extract completions
    completions = [choice["message"]["content"] for choice in result["choices"]]
    
    # Post-process completions to remove any "Bernard:" prefix
    processed_completions = []
    for completion in completions:
        # Check for and remove "Bernard:" prefix (case insensitive)
        if completion.lower().startswith("bernard:"):
            completion = completion[completion.find(':')+1:].strip()
        processed_completions.append(completion)
        
    print(f"Received {len(processed_completions)} completions from OpenAI")
    return processed_completions

def get_groq_completion(messages, max_tokens, n, temperature, settings):
    """
    Get completions from the Groq API.
    Since Groq doesn't support multiple completions in a single request (n>1),
    we make multiple requests with slightly varied temperatures to get diversity.
    """
    # Get Groq settings
    groq_settings = settings.get("groq", {})
    model = groq_settings.get("model", "llama-3.3-70b-specdec")
    api_key = groq_settings.get("apiKey", GROQ_API_KEY)  # Use the global variable as fallback
    
    # Set up Groq API details
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    # For diversity, we'll make n separate requests with slightly different temperatures
    completions = []
    print(f"Requesting {n} completions from Groq with model {model}...")
    
    # For Groq, we need to make separate requests instead of using n>1
    for i in range(n):
        # For each iteration, slightly adjust the temperature to get variety
        adjusted_temp = min(0.99, temperature + (i * 0.10))
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": adjusted_temp,
            "presence_penalty": 0.8,
            "frequency_penalty": 0.8
        }
        
        print(f"Sending request {i+1}/{n} to Groq with temperature {adjusted_temp}")
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        result = resp.json()
        
        # Extract content from response
        content = result["choices"][0]["message"]["content"]
        completions.append(content)
        print(f"Received completion {i+1}/{n} from Groq")
        
        # Brief pause between requests to avoid rate limiting
        if i < n-1:  # Don't pause after the last request
            time.sleep(0.5)
    
    # Post-process completions to remove any "Bernard:" prefix
    processed_completions = []
    for completion in completions:
        # Check for and remove "Bernard:" prefix (case insensitive)
        if completion.lower().startswith("bernard:"):
            completion = completion[completion.find(':')+1:].strip()
        processed_completions.append(completion)
        
    print(f"Received all {len(processed_completions)} completions from Groq")
    return processed_completions

# Function to load settings
def load_settings():
    """
    Load settings from environment variables only.
    """
    settings = {
        "voice": {
            "apiKey": os.getenv("ELEVENLABS_API_KEY", ""),
            "voiceId": os.getenv("ELEVENLABS_VOICE_ID", "CwhRBWXzGAHq8TQ4Fs17"),
            "model": os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
            "outputFormat": os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_192"),
            "seed": int(os.getenv("ELEVENLABS_SEED", "8675309")),
            "voiceSettings": {
                "stability": float(os.getenv("ELEVENLABS_STABILITY", "0.5")),
                "similarity_boost": float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75")),
                "speed": float(os.getenv("ELEVENLABS_SPEED", "0.9")),
                "style": float(os.getenv("ELEVENLABS_STYLE", "0.3")),
            },
        },
        "llm": {
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "openai": {
                "apiKey": os.getenv("OPENAI_API_KEY", ""),
                "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            },
            "groq": {
                "apiKey": os.getenv("GROQ_API_KEY", ""),
                "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-specdec"),
            },
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.9")),
            "maxTokens": int(os.getenv("LLM_MAX_TOKENS", "150")),
            "completionsCount": int(os.getenv("LLM_COMPLETIONS_COUNT", "4")),
            "perplexity": {
                "apiKey": os.getenv("PERPLEXITY_API_KEY", ""),
                "model": os.getenv("PERPLEXITY_MODEL", "sonar"),
            },
        },
        "recorder": {
            "silenceThreshold": int(os.getenv("RECORDER_SILENCE_THRESHOLD", "500")),
            "silenceDuration": float(os.getenv("RECORDER_SILENCE_DURATION", "5.0")),
            "minRecordingDuration": float(os.getenv("RECORDER_MIN_DURATION", "1.0")),
            "maxRecordingDuration": float(os.getenv("RECORDER_MAX_DURATION", "60.0")),
            "useNoiseReduction": os.getenv("RECORDER_USE_NOISE_REDUCTION", "True") == "True",
        },
        "system": {
            "systemPrompt": os.getenv("SYSTEM_PROMPT", system_prompt()),
            "fileLocations": {
                "kbDir": os.getenv("KB_DIR", "data/kb"),
                "vectorStoreDir": os.getenv("VECTOR_STORE_DIR", "data/vector_store"),
                "chatVectorStoreDir": os.getenv("CHAT_VECTOR_STORE_DIR", "data/chat_vector_store"),
                "chatHistoryFile": os.getenv("CHAT_HISTORY_JSON", "data/chat_history.json"),
                "phrasesFile": os.getenv("PHRASES_FILE", "data/phrases.json"),
            },
            "language": {
                "defaultLanguage": os.getenv("DEFAULT_LANGUAGE", "auto"),
                "useLanguageDetection": os.getenv("USE_LANGUAGE_DETECTION", "True") == "True",
            },
            "debugMode": os.getenv("DEBUG_MODE", "False") == "True",
        },
    }
    return settings

def save_settings(settings):
    try:
        # Ensure settings.json exists by creating it if it doesn't
        if not os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        else:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        print(f"Settings saved to {SETTINGS_FILE}: {settings}")
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

def create_default_settings():
    settings = {
        "voice": {
            "apiKey": "",
            "voiceId": "CwhRBWXzGAHq8TQ4Fs17",
            "model": "eleven_multilingual_v2",
            "outputFormat": "mp3_44100_192",
            "seed": 8675309,
            "voiceSettings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": 0.9,
                "style": 0.3
            }
        },
        "llm": {
            "openai": {
                "apiKey": "",
                "model": "gpt-4.1-mini"
            },
            "groq": {
                "apiKey": "",
                "model": "llama-3.3-70b-specdec"
            },
            "perplexity": {
                "apiKey": "",
                "model": "sonar"
            },
            "provider": "openai",
            "temperature": 0.9,
            "maxTokens": 150,
            "completionsCount": 4
        },
        "recorder": {
            "silenceThreshold": 500,
            "silenceDuration": 5.0,
            "minRecordingDuration": 1.0,
            "maxRecordingDuration": 60.0,
            "useNoiseReduction": True
        },
        "system": {
            "systemPrompt": "You are the persona Bernard Muller...",
            "fileLocations": {
                "kbDir": "data/kb",
                "vectorStoreDir": "data/vector_store",
                "chatVectorStoreDir": "data/chat_vector_store",
                "chatHistoryFile": "data/chat_history.json",
                "phrasesFile": "data/phrases.json"
            },
            "language": {
                "defaultLanguage": "auto",
                "useLanguageDetection": True
            },
            "debugMode": False
        }
    }
    print(f"Created default settings: {settings}")
    return settings

##############################################################################
# ASYNC TRANSCRIPTION
##############################################################################
async def transcribe_audio_async(audio_data):
    """
    Asynchronous version of transcribe_audio_eleven.
    Uses aiohttp instead of requests for non-blocking HTTP calls.
    """
    if not audio_data:
        print("Error: Empty audio data in transcribe_audio_async")
        return ""  # Return empty string instead of None
        
    # Handle different types of audio data
    try:
        if isinstance(audio_data, bytes):
            # Use the original bytes
            wav_bytes = audio_data
        else:
            # Assume it's an AudioData object from speech_recognition
            wav_bytes = audio_data.get_wav_data()
        
        # Validate the audio data before sending
        if not wav_bytes or len(wav_bytes) == 0:
            print("Error: No valid audio bytes to transcribe")
            return ""
            
        # ElevenLabs API setup
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": ELEVENLABS_API_KEY}
        
        # Use aiohttp ClientSession for async HTTP
        async with aiohttp.ClientSession() as session:
            # Match the same structure as the synchronous version
            # Instead of using FormData, use multipart/form-data manually
            data = aiohttp.FormData()
            data.add_field('model_id', 'scribe_v1')
            # Omit language_code completely instead of setting to None or empty string
            
            # Add file as a separate field
            data.add_field('file', wav_bytes, filename='audio.wav', content_type='audio/wav')
            
            try:
                async with session.post(url, headers=headers, data=data) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        print(f"Error in transcribe_audio_async: {resp.status} {error_text}")
                        return ""  # Return empty string on error
                    
                    result = await resp.json()
                    transcript = result.get("text", "")
                    print(f"Transcription successful: {transcript[:50]}...")
                    return transcript
            except Exception as e:
                print(f"Network error in transcribe_audio_async: {e}")
                return ""  # Return empty string on exception
    except Exception as e:
        print(f"General error in transcribe_audio_async: {e}")
        return ""  # Return empty string on exception

##############################################################################
# CONCURRENT CONTEXT RETRIEVAL
##############################################################################
async def retrieve_context_async(query, k=2, similarity_threshold=0.1):
    """
    Asynchronous version of retrieve_context.
    Uses a thread pool to handle the FAISS operations which are CPU-bound.
    """
    if not query.strip():
        print("Empty query, no context retrieved")
        return ""
        
    # Run the CPU-bound FAISS operations in a thread pool
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        context = await loop.run_in_executor(
            pool, 
            partial(retrieve_context, query, k, similarity_threshold)
        )
    
    return context

async def retrieve_chat_history_async(query, subject=None, days=None, max_exchanges=2, 
                                      similarity_threshold=0.3, search_all_subjects=False):
    """
    Asynchronous version of retrieve_chat_history.
    Uses a thread pool to handle the FAISS operations which are CPU-bound.
    """
    # Run the CPU-bound FAISS operations in a thread pool
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        chat_context = await loop.run_in_executor(
            pool, 
            partial(retrieve_chat_history, query, subject, days, 
                   max_exchanges, similarity_threshold, search_all_subjects)
        )
    
    return chat_context

async def get_perplexity_response_async(query):
    """
    Asynchronous version of get_perplexity_response.
    """
    # System prompt for Perplexity
    system_prompt = (
        "Je bent de co-assistent van Bernard en helpt hem om actuele zaken op internet op te zoeken. "
        "Als er in de prompt query irrelevante niet zoekbare dingen staan vermeld je dat in de chat completion. "
        "Antwoord kort, bondig en accuraat in maximaal 1 zin."
    )
    
    # Parameters for the Perplexity API
    llm_params = {
        "model": "sonar",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.95,
        "search_domain_filter": None,
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": None,
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "response_format": None,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Sending query to Perplexity: {query[:50]}...")
        async with aiohttp.ClientSession() as session:
            async with session.post(PERPLEXITY_API_URL, json=llm_params, headers=headers) as resp:
                if resp.status != 200:
                    print(f"Perplexity API error: {resp.status} {await resp.text()}")
                    return "Error retrieving information from internet"
                
                data = await resp.json()
                # Extract the response content
                internet_info = data.get("choices", [{}])[0].get("message", {}).get("content", "No information found.")
                print(f"Received Perplexity response: {internet_info[:50]}...")
                return internet_info
    except Exception as e:
        print(f"Error querying Perplexity API: {str(e)}")
        return f"Error retrieving information from internet: {str(e)}"

##############################################################################
# STREAMING LLM RESPONSE
##############################################################################
async def get_openai_streaming_completion(messages, temp=0.9, max_tokens=None):
    """
    Get streaming completions from the OpenAI API.
    Yields chunks of text as they become available.
    """
    # Get OpenAI settings
    settings = load_settings()
    if not isinstance(settings, dict):
        raise ValueError(f"Settings is not a dictionary! Got: {type(settings)} with value: {settings}")
    openai_settings = settings.get("llm", {}).get("openai", {})
    model = openai_settings.get("model", "gpt-4.1-mini")
    api_key = openai_settings.get("apiKey", openai.api_key)
    
    # Set up the API call
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "stream": True,  # Enable streaming
    }
    
    if max_tokens:
        data["max_tokens"] = max_tokens
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"OpenAI API error: {resp.status} {error_text}")
                yield f"Error: {error_text}"
                return
                
            # Process the streaming response
            buffer = ""
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line == "":
                    continue
                if line == "data: [DONE]":
                    break
                    
                if line.startswith("data: "):
                    try:
                        json_str = line[6:]  # Remove "data: " prefix
                        chunk = json.loads(json_str)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            buffer += content
                            yield content
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON from OpenAI: {line}")
                        continue

async def get_groq_streaming_completion(messages, temp=0.9, max_tokens=None):
    """
    Get streaming completions from the Groq API.
    Yields chunks of text as they become available.
    """
    # Get Groq settings
    settings = load_settings()
    if not isinstance(settings, dict):
        raise ValueError(f"Settings is not a dictionary! Got: {type(settings)} with value: {settings}")
    groq_settings = settings.get("llm", {}).get("groq", {})
    model = groq_settings.get("model", "llama-3.3-70b-specdec")
    api_key = groq_settings.get("apiKey", GROQ_API_KEY)
    
    # Set up the API call
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "stream": True,  # Enable streaming
    }
    
    if max_tokens:
        data["max_tokens"] = max_tokens
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"Groq API error: {resp.status} {error_text}")
                yield f"Error: {error_text}"
                return
                
            # Process the streaming response
            buffer = ""
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line == "":
                    continue
                if line == "data: [DONE]":
                    break
                    
                if line.startswith("data: "):
                    try:
                        json_str = line[6:]  # Remove "data: " prefix
                        chunk = json.loads(json_str)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            buffer += content
                            yield content
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON from Groq: {line}")
                        continue

async def stream_groq_completions(messages, temp=0.7, max_tokens=None):
    """
    Stream completions from Groq's API.
    
    Args:
        messages: List of message dictionaries to send to Groq
        temp: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        
    Yields:
        Generated text chunks as they arrive
    """
    # Get Groq settings
    settings = load_settings()
    if not isinstance(settings, dict):
        raise ValueError(f"Settings is not a dictionary! Got: {type(settings)} with value: {settings}")
    groq_settings = settings.get("llm", {}).get("groq", {})
    model = groq_settings.get("model", "llama-3.3-70b-specdec")
    api_key = groq_settings.get("apiKey", GROQ_API_KEY)
    
    # Set up the API call
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "stream": True,  # Enable streaming
    }
    
    if max_tokens:
        data["max_tokens"] = max_tokens
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"Groq API error: {resp.status} {error_text}")
                yield f"Error: {error_text}"
                return
                
            # Process the streaming response
            buffer = ""
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line == "":
                    continue
                if line == "data: [DONE]":
                    break
                    
                if line.startswith("data: "):
                    try:
                        json_str = line[6:]  # Remove "data: " prefix
                        chunk = json.loads(json_str)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            buffer += content
                            yield content
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON from Groq: {line}")
                        continue

def process_selected_completion(pipeline_id, session_id, completion_text, transcript=None):
    """
    Process a selected completion for TTS conversion.
    This runs in a background thread.
    """
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async process
        loop.run_until_complete(_process_selected_completion_async(pipeline_id, session_id, completion_text, transcript))
        loop.close()
    except Exception as e:
        print(f"Error in process_selected_completion: {e}")
        traceback.print_exc()

async def _process_selected_completion_async(pipeline_id, session_id, completion_text, transcript=None):
    """
    Async implementation of processing a selected completion.
    """
    try:
        # Update status
        pipeline._emit_update(session_id, "status_update", {"status": "Converting selection to speech..."})
        
        # If transcript not provided, get it from the pipeline
        if transcript is None and pipeline_id in pipeline.active_pipelines:
            transcript = pipeline.active_pipelines[pipeline_id].get("transcript", "")
            print(f"Retrieved transcript from pipeline: {transcript[:50]}...")
        
        # Print detailed debug info about what we're saving
        print(f"---CHAT HISTORY DEBUG---")
        print(f"Pipeline ID: {pipeline_id}")
        print(f"Transcript to use: {transcript[:100]}...")
        print(f"Selected completion: {completion_text[:100]}...")
        
        # Save to chat history with this transcript and completion
        try:
            # Extract user input from transcript (improved extraction)
            user_text = transcript
            if "Bernard:" in user_text and "User:" in user_text:
                # Split on the first occurrence only
                parts = user_text.split("User:", 1)
                if len(parts) > 1:
                    user_text = parts[1].strip()
                    print(f"Extracted user text from transcript: {user_text[:50]}...")
                else:
                    print("Warning: Found 'Bernard:' but couldn't extract after 'User:'")
            elif "Bernard:" not in user_text:
                # If no formatting, assume entire text is user input
                print(f"Using raw transcript as user text: {user_text[:50]}...")
                
            # Now we have the correct user_text and completion_text to save
            result = save_chat_exchange_json(user_text, completion_text)
            if result:
                print("Successfully saved to chat history")
                
                # Update global lastBernardStatement if it exists
                if 'lastBernardStatement' in globals():
                    globals()['lastBernardStatement'] = completion_text
                    print("Updated global lastBernardStatement")
                    
            else:
                print("Failed to save to chat history")
                
        except Exception as chat_error:
            print(f"Error saving to chat history: {chat_error}")
            print(traceback.format_exc())
        
        # Prepare text chunks
        llm_queue = asyncio.Queue()
        
        # Split by sentences for better chunking
        import re
        # Split at sentence boundaries with . ! ?
        chunks = re.split(r'(?<=[.!?]) +', completion_text)
        
        for chunk in chunks:
            if chunk.strip():
                await llm_queue.put(chunk)
        await llm_queue.put(None)
        
        # Process TTS chunks
        tts_queue = asyncio.Queue()
        audio_urls = await pipeline._stream_tts_audio_manual(session_id, llm_queue, tts_queue)
        
        # Update pipeline record
        if pipeline_id in pipeline.active_pipelines:
            pipeline.active_pipelines[pipeline_id]["audio_urls"] = audio_urls
        
        # Notify client of completion
        pipeline._emit_update(session_id, "pipeline_complete", {
            "transcript": transcript,
            "response": completion_text,
            "audio_urls": audio_urls
        })
        
        # Now that processing is complete, we can clean up
        # del pipeline.active_pipelines[pipeline_id]
        
    except Exception as e:
        print(f"Error processing selected completion: {e}")
        print(traceback.format_exc())
        pipeline._emit_update(session_id, "status_update", {
            "status": f"Error converting to speech: {str(e)}"
        })

##############################################################################
# STREAMING TEXT-TO-SPEECH
##############################################################################
async def speak_text_streaming(text, callback_url=None):
    """
    Generate audio in chunks and stream it back to the client.
    
    Args:
        text: The text to convert to speech
        callback_url: Optional URL to notify when a chunk is ready
        
    Returns:
        A list of audio file URLs that can be played in sequence
    """
    # If text is very short, use normal non-streaming TTS
    if len(text) < 50:
        audio_url = speak_text_eleven(text)
        return [audio_url]
    
    # Get voice settings
    settings = load_settings()
    if not isinstance(settings, dict):
        raise ValueError(f"Settings is not a dictionary! Got: {type(settings)} with value: {settings}")
    voice_settings = settings.get("voice", {})
    
    # Use settings values or defaults
    voice_id = voice_settings.get("voiceId", "eSAnsg5EvoUbefcXwEKT") 
    model_id = voice_settings.get("model", "eleven_multilingual_v2")
    output_format = voice_settings.get("outputFormat", "mp3_44100_192")
    seed = voice_settings.get("seed", 8675309)
    voice_params = voice_settings.get("voiceSettings", {
        "stability": 0.5, 
        "similarity_boost": 0.75, 
        "speed": 0.9, 
        "style": 0.3
    })
    
    # Split text into meaningful chunks for parallel processing
    # Try to split at sentence boundaries
    import re
    chunks = re.split(r'(?<=[.!?]) +', text)
    
    # Combine very short chunks
    final_chunks = []
    current_chunk = ""
    for chunk in chunks:
        if len(current_chunk) + len(chunk) < 100:
            current_chunk += " " + chunk
        else:
            if current_chunk:
                final_chunks.append(current_chunk.strip())
            current_chunk = chunk
    if current_chunk:
        final_chunks.append(current_chunk.strip())
    
    # Process chunks in parallel
    audio_urls = []
    tasks = []
    
    # Set up API call parameters
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    
    async with aiohttp.ClientSession() as session:
        # Start processing all chunks in parallel
        for i, chunk in enumerate(final_chunks):
            if not chunk:
                continue
                
            payload = {
                "text": chunk,
                "model_id": model_id,
                "output_format": output_format,
                "seed": seed,
                "voice_settings": voice_params
            }
            
            # Create a task for this chunk
            task = asyncio.create_task(
                process_tts_chunk(session, tts_url, headers, payload, i, callback_url)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete and collect results in order
        results = await asyncio.gather(*tasks)
        
        # Sort results by chunk index and collect URLs
        sorted_results = sorted(results, key=lambda x: x[0])
        audio_urls = [url for _, url in sorted_results if url]
    
    return audio_urls

async def process_tts_chunk(session, url, headers, payload, chunk_index, callback_url=None):
    """Process a single TTS chunk"""
    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                print(f"TTS error for chunk {chunk_index}: {resp.status} {await resp.text()}")
                return chunk_index, ""
            
            # Save the audio chunk
            ext = "mp3" if payload.get("output_format", "").startswith("mp3") else "wav"
            timestamp = int(time.time())
            # Add unique identifier to prevent filename collisions
            import uuid
            unique_id = uuid.uuid4().hex[:8]
            filename = f"tts_chunk_{timestamp}_{chunk_index}_{unique_id}.{ext}"
            filepath = os.path.join(AUDIO_FOLDER, filename)
            
            # Write the audio data to file
            with open(filepath, "wb") as f:
                f.write(await resp.read())
            
            # Generate the URL for the audio file
            audio_url = f"/temp_audio/{filename}"
            
            # Notify the client if a callback URL was provided
            if callback_url:
                async with session.post(callback_url, json={
                    "chunk_index": chunk_index,
                    "audio_url": audio_url,
                    "is_last": False
                }) as _:
                    pass
            
            return chunk_index, audio_url
    except Exception as e:
        print(f"Error processing TTS chunk {chunk_index}: {e}")
        return chunk_index, ""

##############################################################################
# FLASK ROUTES
##############################################################################

@app.route("/")
def index():
    return send_from_directory(STATIC_FOLDER, "index.html")

@app.route("/retrieve_context", methods=["POST"])
def route_retrieve_context():
    data = request.json
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # First, check if the query is asking about chat history with date references
    date_patterns = [
        r'\b(gisteren|yesterday)\b',
        r'\b(vorige week|last week)\b',
        r'\b(afgelopen week|past week)\b',
        r'\b(vorige maand|last month)\b',
        r'\b(eerder|earlier)\b',
        r'\b(\d+) dagen geleden|(\d+) days ago\b'
    ]
    
    days_to_search = None
    
    for pattern in date_patterns:
        if re.search(pattern, query.lower()):
            if 'gisteren' in query.lower() or 'yesterday' in query.lower():
                days_to_search = 1
            elif 'vorige week' in query.lower() or 'last week' in query.lower() or 'afgelopen week' in query.lower() or 'past week' in query.lower():
                days_to_search = 7
            elif 'vorige maand' in query.lower() or 'last month' in query.lower():
                days_to_search = 30
            else:
                # Default to 7 days if we detect date reference but can't determine specific period
                days_to_search = 7
                
            match = re.search(r'(\d+) (dagen|days)', query.lower())
            if match:
                try:
                    days_to_search = int(match.group(1))
                except:
                    pass
                    
            print(f"Date search detected: Searching past {days_to_search} days")
            break
    
    # Get context from vector store
    kb_context = retrieve_context(query, k=2)
    
    # Get chat history context using the new unified function
    chat_context = retrieve_chat_history(
        query, 
        subject=CURRENT_SUBJECT[0], 
        days=days_to_search,
        max_exchanges=2
    )
    
    # Combine contexts
    combined_context = ""
    if kb_context and chat_context:
        combined_context = f"Knowledge Base Context:\n{kb_context}\n\n---\n\nChat History Context:\n{chat_context}"
    elif kb_context:
        combined_context = kb_context
    elif chat_context:
        combined_context = f"Chat History Context:\n{chat_context}"
    
    # Return the context to replace the textbox content
    return jsonify({"context": combined_context})

@app.route("/upload_doc", methods=["POST"])
def route_upload_doc():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400
    filename = file.filename
    os.makedirs(KB_DIR, exist_ok=True)
    dest = os.path.join(KB_DIR, filename)
    file.save(dest)
    return jsonify({"message": f"Saved {filename} to KB folder."})

@app.route("/log_event", methods=["POST"])
def route_log_event():
    data = request.json
    
    # Check if this is the new memory format
    if "memory" in data:
        memory = data.get("memory")
        title = memory.get("title", "")
        tags = memory.get("tags", [])
        date = memory.get("date", datetime.now().strftime("%Y-%m-%d"))
        text = memory.get("text", "")
        
        if not title or not text:
            return jsonify({"error": "Title and text are required"}), 400
        
        # Create memory object with timestamp
        memory_obj = {
            "title": title,
            "tags": tags,
            "date": date,
            "text": text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Define path for the all memories file
        memories_file = os.path.join(KB_DIR, "memories.json")
        
        # Load existing memories file if it exists
        if os.path.exists(memories_file):
            try:
                with open(memories_file, "r", encoding="utf-8") as f:
                    memories = json.load(f)
            except Exception as e:
                print(f"Error loading memories file: {e}")
                memories = {"memories": []}
        else:
            # Create new memories structure
            memories = {"memories": []}
        
        # Add the new memory
        memories["memories"].append(memory_obj)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(memories_file), exist_ok=True)
        
        # Save updated memories file
        try:
            with open(memories_file, "w", encoding="utf-8") as f:
                json.dump(memories, f, ensure_ascii=False, indent=2)
            
            return jsonify({"message": f"Memory '{title}' logged successfully"})
        except Exception as e:
            print(f"Error saving memories file: {e}")
            return jsonify({"error": f"Error saving memory: {str(e)}"}), 500
    
    # Handle the original text-only format
    else:
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {text}\n{'-'*40}\n"
        
        # Create log.txt in the KB directory
        log_file = os.path.join(KB_DIR, "log.txt")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)
        
        return jsonify({"message": "Event logged successfully to log.txt"})

@app.route("/update_vector", methods=["POST"])
def route_update_vector():
    """
    Update both vector stores - main KB and chat history.
    """
    try:
        # 1. Update the main vector store
        if os.path.exists(VECTOR_STORE_DIR):
            shutil.rmtree(VECTOR_STORE_DIR, ignore_errors=True)
            print(f"Removed existing vector store at {VECTOR_STORE_DIR}")
        
        kb_store = build_vector_store()
        
        # 2. Update the chat vector store (incremental)
        chat_store = build_chat_vector_store(incremental=True)
        
        # Success message with combined stats
        kb_vectors = kb_store.index.ntotal if kb_store else 0
        chat_vectors = chat_store.index.ntotal if chat_store else 0
        
        message = f"Vector stores updated: Knowledge base ({kb_vectors} vectors), Chat history ({chat_vectors} vectors)"
        print(message)
        return jsonify({"message": message})
        
    except Exception as e:
        error_msg = f"Error updating vector stores: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route("/get_subjects", methods=["GET"])
def route_get_subjects():
    """Return the list of all available subjects and the currently selected subject"""
    try:
        subjects = load_subjects()
        # Extract just the names for backwards compatibility
        subject_names = [s["name"] if isinstance(s, dict) else s for s in subjects]
        
        print(f"Returning {len(subject_names)} subjects: {subject_names}")
        return jsonify({
            "subjects": subject_names,
            "current_subject": CURRENT_SUBJECT[0]
        })
    except Exception as e:
        print(f"Error in route_get_subjects: {e}")
        return jsonify({
            "subjects": ["Default Subject"], 
            "current_subject": "Default Subject",
            "error": str(e)
        })

@app.route("/get_subject_details", methods=["POST"])
def route_get_subject_details():
    """Get details for a specific subject"""
    try:
        data = request.json
        subject_name = data.get("subject", "")
        
        if not subject_name:
            return jsonify({"error": "No subject name provided"}), 400
        
        subjects = load_subjects()
        for subject in subjects:
            if isinstance(subject, dict) and subject.get("name") == subject_name:
                return jsonify({
                    "name": subject.get("name"),
                    "description": subject.get("description", "")
                })
        
        # If not found
        return jsonify({"name": subject_name, "description": ""})
    except Exception as e:
        print(f"Error in route_get_subject_details: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/select_subject", methods=["POST"])
def route_select_subject():
    data = request.json
    subj = data.get("subject", "")
    if not subj:
        return jsonify({"message": "No subject provided"}), 400
    
    # Get description if provided
    description = data.get("description", None)
    
    try:
        # Update the current subject and optionally its description
        update_current_subject(subj, description)
        
        return jsonify({"message": f"Subject '{subj}' selected."})
    except Exception as e:
        print(f"Error in route_select_subject: {e}")
        return jsonify({"message": f"Error selecting subject: {str(e)}"}), 500

@app.route("/update_subject", methods=["POST"])
def route_update_subject():
    """Update or add a subject with description"""
    data = request.json
    name = data.get("name", "")
    description = data.get("description", "")
    
    if not name:
        return jsonify({"error": "No subject name provided"}), 400
    
    try:
        subjects = load_subjects()
        subject_exists = False
        
        # Update existing subject
        for subject in subjects:
            if isinstance(subject, dict) and subject.get("name") == name:
                subject["description"] = description
                subject_exists = True
                break
                
        # Add new subject if it doesn't exist
        if not subject_exists:
            subjects.append({"name": name, "description": description})
            
        # Save changes
        save_subjects(subjects)
        
        # If this is the current subject, update the global variable
        if CURRENT_SUBJECT[0] == name:
            # No need to update the global variable name, just refresh the subject list
            pass
            
        return jsonify({"message": f"Subject '{name}' updated successfully"})
    except Exception as e:
        print(f"Error in route_update_subject: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/transcribe_audio", methods=["POST"])
def route_transcribe_audio():
    try:
        # Check if the audio file is in the request
        if 'audio' not in request.files:
            return jsonify({
                "error": "No audio file provided",
                "success": False,
                "transcript": ""
            }), 400
            
        audio_file = request.files['audio']
        
        # Save the audio file temporarily
        temp_filename = os.path.join(AUDIO_FOLDER, f"client_recording_{int(time.time())}.webm")
        audio_file.save(temp_filename)
        print(f"Client recording saved to {temp_filename}")
        
        # Send directly to ElevenLabs API
        transcript = ""
        try:
            # Read the file as binary data
            with open(temp_filename, 'rb') as f:
                audio_data = f.read()
                
            url = "https://api.elevenlabs.io/v1/speech-to-text"
            headers = {"xi-api-key": ELEVENLABS_API_KEY}
            files = {"file": ("audio.webm", audio_data, "audio/webm")}
            data = {"model_id": "scribe_v1", "language_code": None}
            
            resp = requests.post(url, headers=headers, data=data, files=files)
            if resp.status_code == 200:
                transcript = resp.json().get("text", "")
                print(f"ElevenLabs transcript: '{transcript}'")
            else:
                print(f"ElevenLabs API error: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"Error with ElevenLabs API: {e}")
            
        # Clean up temporary files
        try:
            os.remove(temp_filename)
        except:
            pass
            
        # Clean up the transcript
        if transcript:
            # Remove noise descriptions
            noise_patterns = [
                r'\(background noise\)', r'\(music\)', r'\(traffic\)', 
                r'\(car\s+[^\)]*\)', r'\(roaring\)', r'\(craowng\)',
                r'\([^\)]*noise[^\)]*\)', r'\([^\)]*sound[^\)]*\)'
            ]
            
            for pattern in noise_patterns:
                transcript = re.sub(pattern, '', transcript, flags=re.IGNORECASE)
            
            # Clean up whitespace
            transcript = re.sub(r'\s+', ' ', transcript).strip()
            
        # Determine if transcript is valid
        is_valid_transcript = (
            transcript and 
            not re.match(r'^\s*\([^)]*\)\s*$', transcript) and 
            len(transcript.strip().split()) >= 2
        )
        
        return jsonify({
            "success": True,
            "transcript": transcript if is_valid_transcript else ""
        })
    except Exception as e:
        print(f"Error transcribing client audio: {e}")
        print(traceback.format_exc())
        
        return jsonify({
            "error": str(e),
            "success": False,
            "transcript": ""
        }), 500

@app.route("/translate", methods=["POST"])
def route_translate():
    """
    Translate text using OpenAI's API.
    Expects JSON with:
    - text: text to translate
    - target_lang: target language code (e.g., 'nl', 'en', 'cs', 'sk', 'pl')
    - source_lang: source language code or 'auto' for automatic detection
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_lang = data.get('target_lang', 'en')
        source_lang = data.get('source_lang', 'auto')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Map language codes to full names for better translation
        lang_map = {
            'nl': 'Dutch',
            'en': 'English',
            'cs': 'Czech',
            'sk': 'Slovak',
            'pl': 'Polish'
        }

        # Create the translation prompt
        if source_lang == 'auto':
            prompt = f"Translate the following text to {lang_map.get(target_lang, target_lang)}. Maintain the original meaning and tone:\n\n{text}"
        else:
            prompt = f"Translate the following text from {lang_map.get(source_lang, source_lang)} to {lang_map.get(target_lang, target_lang)}. Maintain the original meaning and tone:\n\n{text}"

        # Get translation from OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator. Provide only the translation without any additional text or explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        translation = response.choices[0].message.content.strip()
        return jsonify({"translation": translation})

    except Exception as e:
        print(f"Translation error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/speak_text", methods=["POST"])
def route_speak_text():
    """General endpoint for speaking text and recording Bernard's statement in the chat history"""
    try:
        data = request.json
        text = data.get("text", "").strip()
        user_text = data.get("user_text", "").strip()  # Optional, used for select_reply
        bernard_text = data.get("bernard_text", "").strip()  # Optional, used when Bernard initiates
        is_enhanced_prompt = data.get("is_enhanced_prompt", False)  # Flag for enhanced prompts
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Get the TTS audio URL
        audio_url = speak_text_eleven(text)
        
        # Save chat exchange based on the context
        if is_enhanced_prompt:
            # Don't record anything for enhanced prompts
            print("Enhanced prompt detected - not recording in chat history")
            pass
        elif user_text:
            # User spoke, Bernard replied
            save_chat_exchange_json(user_text, text)
            last_user_text[0] = user_text
        elif bernard_text:
            # Bernard initiated the conversation
            save_chat_exchange_json("", bernard_text)  # Empty user text since Bernard initiated
        else:
            # For other cases where neither is specified, don't save to history
            pass
        
        flow_mode[0] = "bernard"
        
        # Return response
        return jsonify({
            "message": "Text converted to speech",
            "audio_url": audio_url
        })
    except Exception as e:
        print(f"Error in speak_text: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/manual_record", methods=["POST"])
def route_manual_record():
    """Endpoint for manually initiating recording after audio playback"""
    global recorder
    
    try:
        # Start a new recording with automatic silence detection
        success = recorder.start_recording()
        
        if not success:
            return jsonify({
                "message": "Recording already in progress.",
                "transcript": ""
            })
        
        # Wait for the recording to complete (with a timeout)
        max_wait = 60  # Maximum seconds to wait
        wait_start = time.time()
        while recorder.is_recording:
            time.sleep(0.1)  # Short sleep to prevent CPU hogging
            if time.time() - wait_start > max_wait:
                recorder.stop_recording()  # Force stop if taking too long
                break
        
        # Get the audio data
        audio_data = b''.join(recorder.frames)
        
        if not audio_data:
            return jsonify({
                "message": "No audio captured.",
                "transcript": ""
            })
        
        if len(audio_data) > 30 * 1024 * 1024:
            return jsonify({"error": "Recording too large (max 30 MB). Please try again.", "message": "Recording too large.", "transcript": ""}), 413
        
        # Transcribe the audio
        transcript = transcribe_audio_eleven(audio_data)
        
        # Process and validate transcript
        is_valid_transcript = True
        if not transcript or not transcript.strip():
            is_valid_transcript = False
        elif re.match(r'^\s*\([^)]*\)\s*$', transcript):
            transcript = ""
            is_valid_transcript = False
        # PATCH: Allow transcripts with at least 1 word (not 2)
        elif len(transcript.strip().split()) < 1:
            is_valid_transcript = False
        # Clean up the transcript
        if transcript:
            noise_patterns = [
                r'\(background noise\)', r'\(music\)', r'\(traffic\)', 
                r'\(car\s+[^\)]*\)', r'\(roaring\)', r'\(crang\)',
                r'\([^\)]*noise[^\)]*\)', r'\([^\)]*sound[^\)]*\)'
            ]
            for pattern in noise_patterns:
                transcript = re.sub(pattern, '', transcript, flags=re.IGNORECASE)
            transcript = re.sub(r'\s+', ' ', transcript).strip()
            # PATCH: Allow transcripts with at least 1 word (not 2)
            if not transcript or len(transcript.strip().split()) < 1:
                is_valid_transcript = False
        print(f"[DEBUG] Transcript length: {len(transcript.strip().split())}, content: '{transcript}'")
        # Update flow mode
        if is_valid_transcript:
            flow_mode[0] = "subject" if flow_mode[0] == "bernard" else "bernard"
        else:
            # PATCH: Always return the raw transcript for debugging
            pass
        return jsonify({
            "message": "Manual recording complete",
            "transcript": transcript
        })
    except Exception as e:
        print(f"Error in manual_record: {e}")
        # Return a more graceful error response
        return jsonify({
            "error": str(e),
            "message": "Error during recording",
            "transcript": ""
        }), 500

@app.route("/start_recording", methods=["POST"])
def route_start_recording():
    global recorder
    
    try:
        # Start a new recording
        success = recorder.start_recording()
        
        if success:
            message = "Recording started. Please speak now."
        else:
            message = "Recording already in progress."
            
        return jsonify({
            "message": message,
            "success": success,
            "transcript": ""  # No transcript yet
        })
    except Exception as e:
        print(f"Error in start_recording: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "message": "Error starting recording",
            "success": False,
            "transcript": ""
        }), 500

@app.route("/stop_recording", methods=["POST"])
def route_stop_recording():
    global recorder
    
    try:
        # Check if recording was automatically stopped
        auto_stopped = recorder.is_stopped_automatically()
        print(f"Recording was {'auto-stopped' if auto_stopped else 'manually stopped'}")
        
        # Get audio data from the recorder
        audio_data = recorder.stop_recording()
        
        if not audio_data:
            print("No audio data captured")
            return jsonify({
                "message": "No audio data captured.",
                "success": False,
                "transcript": ""
            })
        
        # Save the recording to a file for debugging
        temp_filename = os.path.join(AUDIO_FOLDER, f"recording_{int(time.time())}.wav")
        recorder.save_wav(temp_filename)
        print(f"Recording saved to {temp_filename}")
        
        # Check length of audio data for debugging
        print(f"Audio data length: {len(audio_data)} bytes")
        
        # Transcribe using ElevenLabs
        transcript = transcribe_audio_eleven(audio_data)
        print(f"Transcript from ElevenLabs: '{transcript}'")
        
        # If ElevenLabs returns empty or just parenthesized content, try Google
        if not transcript.strip() or re.match(r'^\s*\([^)]*\)\s*$', transcript):
            print("ElevenLabs transcript was inadequate, trying Google...")
            try:
                # Create a properly formatted WAV file for Google
                import io
                import wave
                import speech_recognition as sr
                
                # Create a WAV file in memory
                wav_io = io.BytesIO()
                with wave.open(wav_io, 'wb') as wf:
                    wf.setnchannels(1) 
                    wf.setsampwidth(2)  # 2 bytes for paInt16
                    wf.setframerate(16000)  # Assumes 16kHz sample rate
                    wf.writeframes(audio_data)
                
                # Rewind and read the WAV data
                wav_io.seek(0)
                wav_data = wav_io.read()
                
                # Create a temporary file for Google to read
                google_temp = os.path.join(AUDIO_FOLDER, f"google_temp_{int(time.time())}.wav")
                with open(google_temp, 'wb') as f:
                    f.write(wav_data)
                
                # Use Google Speech Recognition with the file
                r = sr.Recognizer()
                with sr.AudioFile(google_temp) as source:
                    audio = r.record(source)
                    transcript = r.recognize_google(audio)
                    print(f"Google transcript: '{transcript}'")
                    
                # Clean up the temporary file
                try:
                    os.remove(google_temp)
                except:
                    pass
                    
            except Exception as fallback_error:
                print(f"Failed fallback for Google: {fallback_error}")
                import traceback
                print(traceback.format_exc())
        
        # Check if transcript is valid
        is_valid_transcript = True
        if not transcript or not transcript.strip():
            is_valid_transcript = False
        elif re.match(r'^\s*\([^)]*\)\s*$', transcript):  # Only content in parentheses
            transcript = ""
            is_valid_transcript = False
        # PATCH: Allow transcripts with at least 1 word (not 2)
        elif len(transcript.strip().split()) < 1:
            is_valid_transcript = False
        # Post-process the transcript to remove noise descriptions
        if transcript:
            # Remove noise descriptions
            noise_patterns = [
                r'\(background noise\)', r'\(music\)', r'\(traffic\)', 
                r'\(car\s+[^\)]*\)', r'\(roaring\)', r'\(c shirt ng\)',
                r'\([^\)]*noise[^\)]*\)', r'\([^\)]*sound[^\)]*\)'
            ]
            for pattern in noise_patterns:
                transcript = re.sub(pattern, '', transcript, flags=re.IGNORECASE)
            # Clean up whitespace
            transcript = re.sub(r'\s+', ' ', transcript).strip()
            # PATCH: Allow transcripts with at least 1 word (not 2)
            if not transcript or len(transcript.strip().split()) < 1:
                is_valid_transcript = False
        print(f"[DEBUG] Transcript length: {len(transcript.strip().split())}, content: '{transcript}'")
        # Final sanity check
        if is_valid_transcript:
            print(f"Valid transcript: '{transcript}'")
        else:
            print("No valid transcript generated")
            # PATCH: Always return the raw transcript for debugging
            pass
        return jsonify({
            "message": f"Recording stopped {'automatically' if auto_stopped else 'manually' }.",
            "success": True,
            "transcript": transcript,
            "auto_stopped": auto_stopped
        })
    except Exception as e:
        print(f"Error in stop_recording: {e}")
        print(traceback.format_exc())
        
        return jsonify({
            "error": str(e),
            "message": "Error stopping recording",
            "success": False,
            "transcript": ""
        }), 500
    
@app.route("/check_recording_status", methods=["POST"])
def route_check_recording_status():
    global recorder
    
    try:
        # Check if recording is active
        is_recording = recorder.is_recording
        auto_stopped = recorder.is_stopped_automatically()
        
        # If auto-stopped, get the transcript
        transcript = ""
        if auto_stopped and not is_recording:
            # Get audio data from the recorder
            audio_data = b''.join(recorder.frames)
            
            if audio_data:
                # Save the recording to a file for debugging
                temp_filename = os.path.join(AUDIO_FOLDER, f"auto_recording_{int(time.time())}.wav")
                recorder.save_wav(temp_filename)
                print(f"Auto-recording saved to {temp_filename}")
                
                # Transcribe using ElevenLabs
                transcript = transcribe_audio_eleven(audio_data)
                print(f"Auto transcript: '{transcript}'")
                
                # Process transcript for validity
                if transcript and not re.match(r'^\s*\([^)]*\)\s*$', transcript) and len(transcript.strip().split()) >= 2:
                    # Clean up transcript
                    noise_patterns = [
                        r'\(background noise\)', r'\(music\)', r'\(traffic\)', 
                        r'\(car\s+[^\)]*\)', r'\(roaring\)', r'\(crashing\)',
                        r'\([^\)]*noise[^\)]*\)', r'\([^\)]*sound[^\)]*\)'
                    ]
                    
                    for pattern in noise_patterns:
                        transcript = re.sub(pattern, '', transcript, flags=re.IGNORECASE)
                    
                    transcript = re.sub(r'\s+', ' ', transcript).strip()
                else:
                    transcript = ""
        
        return jsonify({
            "is_recording": is_recording,
            "auto_stopped": auto_stopped,
            "transcript": transcript
        })
    except Exception as e:
        print(f"Error checking recording status: {e}")
        print(traceback.format_exc())
        
        return jsonify({
            "error": str(e),
            "is_recording": False,
            "auto_stopped": False,
            "transcript": ""
        }), 500
    
@app.route("/send_llm", methods=["POST"])
def route_send_llm():
    try:
        start_time = time.time()  # Start timer
        data = request.json
        prompt = data.get("prompt", "").strip()
        conversation_context = data.get("conversationContext", [])  # New parameter
        # Get feature toggle states
        use_context = data.get("useContext", True)  # Default to True if not provided
        auto_mode = data.get("autoMode", False)     # Default to False if not provided
        use_internet = data.get("useInternet", False)  # Default to False if not provided
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Store the user text for later use when selecting a reply
        last_user_text[0] = prompt
        
        # IMPORTANT: Extract the transcript from the full prompt
        # This is the key fix - detect language on just the user's input
        user_transcript = prompt
        
        # If there's a "Bernard:" and "User:" format, extract just the user part
        if "Bernard:" in prompt and "User:" in prompt:
            parts = prompt.split("User:")
            if len(parts) > 1:
                user_transcript = parts[1].strip()
        
        # Prepare messages for OpenAI
        settings = load_settings()
        if not isinstance(settings, dict):
            raise ValueError(f"Settings is not a dictionary! Got: {type(settings)} with value: {settings}")
        system_settings = settings.get("system", {})
        prompt_from_settings = system_settings.get("systemPrompt", system_prompt())
        msgs = [{"role": "system", "content": prompt_from_settings}]
        
        # Add conversation history if provided
        if conversation_context:
            msgs.extend(conversation_context)
        # Add context only if the context toggle is enabled
        if use_context:
            # Get KB context
            kb_context = retrieve_context(prompt, k=2)
            
            # Get chat history context using the new unified function
            chat_context = retrieve_chat_history(
                prompt, 
                subject=CURRENT_SUBJECT[0], 
                max_exchanges=2
            )
            
            # Combine contexts
            if kb_context and chat_context:
                combined_context = f"Knowledge Base Context:\n{kb_context}\n\nChat History Context:\n{chat_context}"
            elif kb_context:
                combined_context = kb_context
            elif chat_context:
                combined_context = f"Chat History Context:\n{chat_context}"
            else:
                combined_context = ""
            
            # Add combined context to OpenAI message
            if combined_context:
                print(f"Adding combined context to OpenAI message: {len(combined_context)} chars")
                msgs.append({"role": "system", "content": "Retrieved context:\n" + combined_context})
            else:
                print("No context retrieved for this query")
        else:
            print("Context toggle is off - skipping context retrieval")
        
        # Add internet-based context if internet toggle is enabled
        internet_info = ""
        if use_internet:
            print("Internet toggle is on - querying Perplexity for internet information")
            internet_info = get_perplexity_response(prompt)
            
            if internet_info and not internet_info.startswith("Error"):
                msgs.append({"role": "system", "content": f"Internet Information: {internet_info}"})
                print(f"Added internet information to prompt: {internet_info}")
        
        # Get subject context if available
        subject_context = ""
        subject_description = get_subject_description(CURRENT_SUBJECT[0])
        
        # LANGUAGE DETECTION ON USER TRANSCRIPT ONLY (not full prompt)
        try:
            from langdetect import detect
            lang = detect(user_transcript)
            print(f"Detected language from transcript: {lang}")
            
            # Add language instruction to system prompt
            if lang == "en":
                lang_instruction = "The user is speaking in English. You MUST respond in ENGLISH."
            elif lang == "nl":
                lang_instruction = "The user is speaking in Dutch. You MUST respond in DUTCH."
            else:
                # Default to matching the input language if detected
                lang_instruction = f"The user is speaking in {lang}. You MUST respond in the SAME LANGUAGE."
                
            # Add subject description to the system prompt if available
            if subject_description and msgs and msgs[0]['role'] == 'system':
                subject_name = CURRENT_SUBJECT[0]
                subject_context = f"\n\nCurrent conversation is with: {subject_name}\nContext about this person: {subject_description}"
                # Add as part of system context
                msgs[0]['content'] += subject_context
                print(f"Added subject context to system prompt: {subject_context}")
            
            # Add to system prompt
            msgs[0]["content"] += f"\n\nIMPORTANT: {lang_instruction}"
            print(f"Added language instruction: {lang_instruction}")
            
            # Add instruction to not include "Bernard:" prefix in responses
            msgs[0]['content'] += "\n\nIMPORTANT: Do not include 'Bernard:' or any other speaker prefix in your responses. Just provide the direct response as Bernard."
            
        except Exception as e:
            print(f"Error detecting language: {e}")
            # Continue without language detection if it fails
            
            # Still add subject context even if language detection fails
            if subject_description and msgs and msgs[0]['role'] == 'system':
                subject_name = CURRENT_SUBJECT[0]
                subject_context = f"\n\nCurrent conversation is with: {subject_name}\nContext about this person: {subject_description}"
                msgs[0]['content'] += subject_context
                print(f"Added subject context to system prompt: {subject_context}")
                
            # Add default instruction to not include "Bernard:" prefix
            msgs[0]['content'] += "\n\nIMPORTANT: Do not include 'Bernard:' or any other speaker prefix in your responses. Just provide the direct response as Bernard."
        
        # Add user message
        msgs.append({"role": "user", "content": prompt})
        
        # Get completions from OpenAI - in auto mode, we only need 1 completion
        n_completions = 1 if auto_mode else 4
        completions = get_chat_completion(msgs, max_tokens=150, n=n_completions, temperature=0.9)

        # PATCH: Ensure completions is always a list
        if isinstance(completions, str):
            completions = [completions]
        elif not isinstance(completions, list):
            completions = [f"Error: Unexpected completions type: {type(completions)}"]

        print(f"[DEBUG] Type of completions: {type(completions)}, Value: {completions}")

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total request time: {total_time:.2f} seconds")

        return jsonify({
            "completions": completions,
            "autoMode": auto_mode,
            "internetInfo": internet_info if use_internet else "",
            "responseTime": f"{total_time:.2f}"  # Add this line
        })
    except Exception as e:
        import traceback
        print("[ERROR] Unhandled exception in /send_llm:")
        traceback.print_exc()
        return jsonify({"error": f"[UNHANDLED EXCEPTION] {str(e)}"}), 500

@app.route("/get_phrases", methods=["POST"])
def route_get_phrases():
    """Return the list of conversation phrases based on type"""
    try:
        data = request.json
        phrase_type = data.get("type", "all")
        
        if phrase_type not in ['starter', 'ender', 'all']:
            return jsonify({"error": "Invalid phrase type"}), 400
        
        phrases = load_phrases(phrase_type)
        
        # Format the response based on the requested type
        if phrase_type == 'starter':
            return jsonify({"phrases": phrases})
        elif phrase_type == 'ender':
            return jsonify({"phrases": phrases})
        else:
            return jsonify({
                "starters": phrases["starters"],
                "enders": phrases["enders"]
            })
    except Exception as e:
        print(f"Error in route_get_phrases: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/update_phrase", methods=["POST"])
def route_update_phrase():
    """Update or add a conversation phrase"""
    try:
        data = request.json
        phrase_type = data.get("type", "")
        old_text = data.get("old_text", "")
        new_text = data.get("new_text", "")
        
        if not phrase_type or phrase_type not in ['starter', 'ender']:
            return jsonify({"error": "Invalid phrase type"}), 400
            
        if not new_text:
            return jsonify({"error": "New phrase text is required"}), 400
        
        # Call the update function
        success = update_phrase(phrase_type, old_text, new_text)
        
        if success:
            return jsonify({
                "message": f"{'Added new' if not old_text else 'Updated'} conversation {phrase_type} successfully"
            })
        else:
            return jsonify({"error": "Failed to update phrase"}), 500
            
    except Exception as e:
        print(f"Error in route_update_phrase: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route("/temp_audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

@app.route("/enhance_prompt", methods=["POST"])
def route_enhance_prompt():
    try:
        data = request.json
        prompt = data.get("prompt", "").strip()
        
        # Get feature toggle states
        use_context = data.get("useContext", True)
        use_internet = data.get("useInternet", False)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        # Use our own language detection logic instead of relying on the external library
        # Common Dutch words or patterns
        dutch_patterns = [
            r'\b(ik|je|jij|hij|zij|het|wij|jullie|zij)\b',
            r'\b(een|de|het|over|van|met|voor)\b',
            r'\b(vertel|vraag|zeg|wie|wat|waar|waarom|hoe)\b',
            r'\b(graag|misschien|natuurlijk)\b'
        ]
            
        # Check if the text contains Dutch patterns
        is_likely_dutch = False
        for pattern in dutch_patterns:
            if re.search(pattern, prompt.lower()):
                is_likely_dutch = True
                break
                
        # Force Dutch detection for short prompts that match common patterns
        detected_lang = "nl" if is_likely_dutch else "en"
        print(f"Custom language detection for enhance_prompt: {detected_lang}")
        
        # Prepare messages for OpenAI - reduce system prompt length to speed up inference
        enhance_system_prompt = (
            "Je bent Bernard zijn assistent om zijn prompt te verbeteren van slechts een paar woorden moet jij een zin maken. Antwoord in de taal van de prompt. " +
            ("Geef 4 verschillende versies van hoe je dit zou zeggen." if detected_lang == "nl" else 
             "Give 4 different versions of how you would say this.")  +
            ("\n\nAntwoord ALLEEN in het Nederlands." if detected_lang == "nl" else "\n\nAnswer ONLY in English.")
        )
        
        msgs = [{"role": "system", "content": enhance_system_prompt}]
        
        # Add minimal context if the toggle is enabled - limit to reduce inference time
        if use_context:
            # Reduce context retrieval to minimize tokens
            kb_context = retrieve_context(prompt, k=3)  # Reduced from k=2
            chat_context = retrieve_chat_history(prompt, max_exchanges=3, search_all_subjects=True)
            
            # Keep context concise
            combined_context = ""
            if kb_context and chat_context:
                # Take only the first paragraph of each to reduce token count
                kb_first_paragraph = kb_context.split('\n\n')[0] if '\n\n' in kb_context else kb_context
                chat_first_paragraph = chat_context.split('\n\n')[0] if '\n\n' in chat_context else chat_context
                combined_context = f"Context: {kb_first_paragraph}\n{chat_first_paragraph}"
            elif kb_context:
                combined_context = f"Context: {kb_context}"
            elif chat_context:
                combined_context = f"Context: {chat_context}"
            
            # Add context only if it's not too long
            if combined_context and len(combined_context) < 1000:  # Limit context size
                msgs.append({"role": "system", "content": combined_context})
        
        # Add internet information only if really necessary and enabled
            internet_info = get_perplexity_response(prompt)
            if internet_info and not internet_info.startswith("Error") and len(internet_info) < 300:  # Limit size
                msgs.append({"role": "system", "content": f"Info: {internet_info}"})
        
        # Add user message
        msgs.append({"role": "user", "content": prompt})
        
        # Use a smaller model for faster inference if the prompt is simple
        model = "gpt-4.1-mini"
        
        # Use direct API call for more control
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {openai.api_key}", "Content-Type": "application/json"}
        data = {
            "model": model,
            "messages": msgs,
            "max_tokens": 400,  # Reduced from 600
            "temperature": 0.9,
            "top_p": 0.95
        }
        
        # Time the API call
        start_time = time.time()
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        result = resp.json()
        inference_time = time.time() - start_time
        print(f"OpenAI API call took {inference_time:.2f} seconds")
        
        # Extract the full response
        full_response = result["choices"][0]["message"]["content"]
        print(f"Full response from OpenAI: {full_response}")
        
        # Use simpler extraction logic - split by numbers
        variations = []
        pattern = r'(?:^|\n)([1-4])[\.:]\s*(.*(?:\n(?![1-4][\.:]).*)*)'
        matches = re.findall(pattern, full_response, re.MULTILINE | re.DOTALL)
        
        for num, content in sorted(matches):
            variations.append(content.strip())
            print(f"Extracted variation {num}: {content.strip()}")
            
        # If we don't have 4 variations, use a fallback
        if len(variations) < 4:
            # Split by lines and look for numbered lines
            lines = full_response.split('\n')
            current_variation = ""
            current_num = 0
            
            for line in lines:
                if re.match(r'^\s*[1-4][\.:]\s*', line):
                    if current_variation and current_num > 0:
                        while len(variations) < current_num:
                            variations.append("")
                        if current_num <= 4:
                            variations[current_num-1] = current_variation.strip()
                    
                    current_num = int(re.match(r'^\s*([1-4])[\.:]\s*', line).group(1))
                    current_variation = re.sub(r'^\s*[1-4][\.:]\s*', '', line)
                else:
                    current_variation += " " + line
            
            # Add the last variation
            if current_variation and 0 < current_num <= 4:
                while len(variations) < current_num:
                    variations.append("")
                variations[current_num-1] = current_variation.strip()
        
        # Ensure we have exactly 4 variations
        while len(variations) < 4:
            variations.append(prompt)
        variations = variations[:4]  # Limit to 4 if we have more
        
        return jsonify({
            "completions": variations,
            "inference_time": f"{inference_time:.2f} seconds"
        })
    except Exception as e:
        print(f"Error in enhance_prompt: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "completions": ["Error processing request", prompt, "", ""]
        }), 500

# Route to get settings
@app.route("/get_settings", methods=["GET"])
def route_get_settings():
    try:
        # Load settings from settings.json
        settings = load_settings_from_file()
        print(f"Loaded settings from file: {settings}")
        if not settings:
            # Fall back to environment variables or defaults
            settings = create_default_settings()
            print(f"Using default settings: {settings}")
        return jsonify(settings)
    except Exception as e:
        print(f"Error in route_get_settings: {e}")
        return jsonify({"error": str(e)}), 500

# Update route_save_settings to persist to file
@app.route("/save_settings", methods=["POST"])
def route_save_settings():
    try:
        data = request.get_json()
        print(f"Received settings data: {data}")
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Load current settings
        current_settings = load_settings_from_file() or create_default_settings()

        # If the frontend sends a category and settings, update only that section
        if "category" in data and "settings" in data:
            category = data["category"]
            current_settings[category] = data["settings"]
            save_settings_to_file(current_settings)
        else:
            # Only allow full overwrite if all top-level keys are present
            if all(k in data for k in ["voice", "llm", "recorder", "system"]):
                save_settings_to_file(data)
            else:
                return jsonify({"error": "Invalid settings structure"}), 400

        return jsonify({"message": "Settings saved successfully"})
    except Exception as e:
        print(f"Error in route_save_settings: {e}")
        return jsonify({"error": str(e)}), 500

# Route to test voice settings
@app.route("/test_voice", methods=["POST"])
def route_test_voice():
    """Test voice settings with a sample text"""
    try:
        data = request.json
        text = data.get("text", "This is a test of the voice settings.")
        settings = data.get("settings", {})
        
        # Extract voice settings
        voice_id = settings.get("voiceId", "eSAnsg5EvoUbefcXwEKT")
        api_key = settings.get("apiKey", ELEVENLABS_API_KEY)
        model_id = settings.get("model", "eleven_multilingual_v2")
        output_format = settings.get("outputFormat", "mp3_44100_192")
        seed = settings.get("seed", 8675309)
        voice_settings = settings.get("voiceSettings", {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "speed": 0.9,
            "style": 0.3
        })
        
        # Call ElevenLabs TTS API with the provided settings
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": model_id,
            "output_format": output_format,
            "seed": seed,
            "voice_settings": voice_settings
        }
        
        response = requests.post(tts_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            ext = "mp3" if output_format.startswith("mp3") else "wav"
            # Save audio file in the audio folder
            timestamp = int(time.time())
            filename = f"tts_test_{timestamp}.{ext}"
            filepath = os.path.join(AUDIO_FOLDER, filename)
            with open(filepath, "wb") as f:
                f.write(response.content)
            print("Test TTS file saved at:", filepath)
            # Return the URL path to the audio file
            return jsonify({"audio_url": f"/temp_audio/{filename}"})
        else:
            return jsonify({"error": f"TTS error: {response.status_code} {response.text}"}), 500
    except Exception as e:
        print(f"Error in route_test_voice: {e}")
        return jsonify({"error": str(e)}), 500

# Route to test LLM settings
@app.route("/test_llm", methods=["POST"])
def route_test_llm():
    """Test LLM settings with a sample prompt"""
    try:
        data = request.json
        prompt = data.get("prompt", "This is a test. What's your name?")
        settings = data.get("settings", {})
        
        # Extract OpenAI settings
        openai_settings = settings.get("openai", {})
        api_key = openai_settings.get("apiKey", openai.api_key)
        model = openai_settings.get("model", "gpt-4.1-mini")
        temperature = openai_settings.get("temperature", 0.9)
        max_tokens = openai_settings.get("maxTokens", 150)
        n = openai_settings.get("completionsCount", 1)
        
        # Prepare the system message
        # Load the system prompt from settings
        current_settings = load_settings()
        system_settings = current_settings.get("system", {})
        system_content = system_settings.get("systemPrompt", system_prompt())  # Fallback to hardcoded if not found
        # Prepare messages for API
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        # Call OpenAI API with the provided settings
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "n": n,
            "temperature": temperature
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            completions = [choice["message"]["content"] for choice in result["choices"]]
            return jsonify({"completion": completions[0] if completions else "No response"})
        else:
            return jsonify({"error": f"LLM error: {response.status_code} {response.text}"}), 500
    except Exception as e:
        print(f"Error in route_test_llm: {e}")
        return jsonify({"error": str(e)}), 500

# Route to test recorder settings
@app.route("/test_recorder", methods=["POST"])
def route_test_recorder():
    """Test recorder settings"""
    try:
        global recorder
        data = request.json
        settings = data.get("settings", {})
        
        # Temporarily adjust recorder settings for the test
        original_silence_threshold = recorder.silence_threshold
        original_silence_duration = recorder.silence_duration
        original_min_recording_duration = recorder.min_recording_duration
        original_max_recording_duration = recorder.max_recording_duration
        
        # Update recorder with test settings
        recorder.silence_threshold = settings.get("silenceThreshold", recorder.silence_threshold)
        recorder.silence_duration = settings.get("silenceDuration", recorder.silence_duration)
        recorder.min_recording_duration = settings.get("minRecordingDuration", recorder.min_recording_duration)
        recorder.max_recording_duration = settings.get("maxRecordingDuration", recorder.max_recording_duration)
        
        print(f"Testing recorder with settings: threshold={recorder.silence_threshold}, silence_duration={recorder.silence_duration}")
        
        # Start recording with adjusted settings
        success = recorder.start_recording()
        
        if not success:
            # Restore original settings if start fails
            recorder.silence_threshold = original_silence_threshold
            recorder.silence_duration = original_silence_duration
            recorder.min_recording_duration = original_min_recording_duration
            recorder.max_recording_duration = original_max_recording_duration
            return jsonify({"error": "Recording already in progress."}), 400
        
        return jsonify({"message": "Recording started with test settings. Use the check_recording_status endpoint to monitor."})
    except Exception as e:
        print(f"Error in route_test_recorder: {e}")
        return jsonify({"error": str(e)}), 500
    

@app.route("/translate_text", methods=["POST"])
def route_translate_text():
    """
    Translate text using OpenAI
    
    Parameters in request:
    - text: Text to translate
    - targetLanguage: Target language code (e.g., 'nl', 'en', 'de')
    - sourceLanguage: Source language code (optional)
    - autoDetectSource: Whether to auto-detect the source language (boolean)
    
    Returns:
    - translatedText: The translated text
    - detectedLanguage: The detected source language (if auto-detect was enabled)
    - detectedLanguageName: Full name of the detected source language
    - targetLanguageName: Full name of the target language
    """
    try:
        data = request.json
        text = data.get("text", "").strip()
        target_lang = data.get("targetLanguage", "nl")
        source_lang = data.get("sourceLanguage", "")
        auto_detect = data.get("autoDetectSource", True)
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Language code to full name mapping
        language_names = {
            "nl": "Dutch",
            "en": "English",
            "cs": "Czech",
            "sk": "Slovak",
            "pl": "Polish",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "tr": "Turkish",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "sv": "Swedish",
            "no": "Norwegian",
            "da": "Danish",
            "fi": "Finnish",
            "el": "Greek",
            "he": "Hebrew",
            "hu": "Hungarian",
            "ro": "Romanian",
            "bg": "Bulgarian",
            "uk": "Ukrainian",
            "hi": "Hindi",
            "id": "Indonesian",
            "th": "Thai",
            "vi": "Vietnamese",
            "auto": "Auto-detected"
        }
        
        # Auto-detect source language if requested
        detected_lang = ""
        if auto_detect:
            try:
                if detect:  # Using the global detect function from langdetect
                    detected_lang = detect(text)
                    print(f"Detected language: {detected_lang}")
                else:
                    # Simple language detection fallback
                    # Dutch patterns
                    dutch_patterns = [
                        r'\b(ik|je|hij|zij|het|wij|jullie|zij)\b',
                        r'\b(een|de|het|over|van|met|voor)\b',
                        r'\b(vertel|vraag|zeg|wie|wat|waar|waarom|hoe)\b'
                    ]
                    
                    # English patterns
                    english_patterns = [
                        r'\b(i|you|he|she|it|we|they)\b',
                        r'\b(a|an|the|of|with|for|in|on|at)\b',
                        r'\b(tell|ask|say|who|what|where|why|how)\b'
                    ]
                    
                    # Check patterns
                    dutch_matches = 0
                    english_matches = 0
                    
                    for pattern in dutch_patterns:
                        if re.search(pattern, text.lower()):
                            dutch_matches += 1
                    
                    for pattern in english_patterns:
                        if re.search(pattern, text.lower()):
                            english_matches += 1
                    
                    detected_lang = "nl" if dutch_matches > english_matches else "en"
                    print(f"Simple detection result: {detected_lang} (Dutch: {dutch_matches}, English: {english_matches})")
            except Exception as e:
                print(f"Language detection error: {e}")
                detected_lang = "en"  # Default to English if detection fails
        
        # Use detected language or provided source language
        source_lang = detected_lang if auto_detect else source_lang
        
        # Skip translation if source and target are the same
        if source_lang == target_lang:
            return jsonify({
                "translatedText": text,
                "detectedLanguage": detected_lang,
                "detectedLanguageName": language_names.get(detected_lang, "Unknown"),
                "targetLanguageName": language_names.get(target_lang, target_lang)
            })
        
        # Load LLM settings
        settings = load_settings().get("llm", {})
        if not isinstance(settings, dict):
            raise ValueError(f"Settings is not a dictionary! Got: {type(settings)} with value: {settings}")
        provider = settings.get("provider", "openai")
        
        # Construct the translation prompt based on languages
        if source_lang and not auto_detect:
            # When source language is explicitly specified
            system_prompt = f"You are a translator. Translate the following text from {language_names.get(source_lang, source_lang)} to {language_names.get(target_lang, target_lang)}. Provide only the translated text without explanations or additional information."
        elif auto_detect:
            # When auto-detecting source language
            system_prompt = f"You are a translator. Translate the following text to {language_names.get(target_lang, target_lang)}. The source language should be automatically detected. Provide only the translated text without explanations or additional information."
        else:
            # Fallback
            system_prompt = f"You are a translator. Translate the following text to {language_names.get(target_lang, target_lang)}. Provide only the translated text without explanations or additional information."
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        # Get translation based on selected provider
        if provider == "openai":
            translation = get_openai_translation(messages, settings)
        elif provider == "groq":
            translation = get_groq_translation(messages, settings)
        else:
            print(f"Unknown provider: {provider}, falling back to OpenAI")
            translation = get_openai_translation(messages, settings)
        
        # Get language names for response
        detected_lang_name = language_names.get(detected_lang, "Unknown")
        target_lang_name = language_names.get(target_lang, target_lang)
        
        return jsonify({
            "translatedText": translation,
            "detectedLanguage": detected_lang,
            "detectedLanguageName": detected_lang_name,
            "targetLanguageName": target_lang_name
        })
        
    except Exception as e:
        print(f"Error in translation: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def get_openai_translation(messages, settings):
    """Get translation from OpenAI"""
    # Get OpenAI settings
    openai_settings = settings.get("openai", {})
    model = openai_settings.get("model", "gpt-4.1-mini")
    api_key = openai_settings.get("apiKey", openai.api_key)
    
    # Call OpenAI API
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.3,  # Lower temperature for more accurate translations
    }
    
    print(f"Sending translation request to OpenAI with model {model}")
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()
    
    # Extract translation
    translation = result["choices"][0]["message"]["content"].strip()
    print(f"Received translation from OpenAI: {translation[:50]}...")
    
    return translation

def get_groq_translation(messages, settings):
    """Get translation from Groq"""
    # Get Groq settings
    groq_settings = settings.get("groq", {})
    model = groq_settings.get("model", "llama-3.3-70b-specdec")
    api_key = groq_settings.get("apiKey", GROQ_API_KEY)
    
    # Set up Groq API details
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.3,  # Lower temperature for more accurate translations
    }
    
    print(f"Sending translation request to Groq with model {model}")
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()
    
    # Extract translation
    translation = result["choices"][0]["message"]["content"].strip()
    print(f"Received translation from Groq: {translation[:50]}...")
    
    return translation

##############################################################################
# FLASK ROUTES FOR PARALLEL PIPELINE
##############################################################################

# Initialize the parallel pipeline
pipeline = ParallelPipeline(socketio)

@app.route("/process_audio_parallel", methods=["POST"])
def route_process_audio_parallel():
    """
    Process audio with the optimized parallel pipeline.
    This route initiates the pipeline and returns immediately,
    with updates sent via Socket.IO.
    """
    try:
        # Get session ID
        session_id = request.form.get("session_id", str(uuid.uuid4()))
        auto_mode = request.form.get("auto_mode", "false").lower() == "true"
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided", "success": False}), 400
        audio_file = request.files['audio']
        audio_file.seek(0, 2)
        file_size = audio_file.tell()
        audio_file.seek(0)
        if file_size > 30 * 1024 * 1024:
            return jsonify({"error": "Recording too large (max 30 MB). Please try again.", "success": False}), 413

        # Save the audio file temporarily
        temp_filename = os.path.join(AUDIO_FOLDER, f"parallel_recording_{int(time.time())}")
        # Detect file type by mimetype or filename
        ext = ''
        if hasattr(audio_file, 'mimetype') and audio_file.mimetype:
            if 'webm' in audio_file.mimetype:
                ext = '.webm'
            elif 'ogg' in audio_file.mimetype:
                ext = '.ogg'
            elif 'wav' in audio_file.mimetype:
                ext = '.wav'
        if not ext and audio_file.filename:
            if audio_file.filename.endswith('.webm'):
                ext = '.webm'
            elif audio_file.filename.endswith('.ogg'):
                ext = '.ogg'
            elif audio_file.filename.endswith('.wav'):
                ext = '.wav'
        temp_filename += ext or '.webm'
        audio_file.save(temp_filename)

        # If not WAV, convert to WAV using pydub
        if not temp_filename.endswith('.wav'):
            from pydub import AudioSegment
            audio = AudioSegment.from_file(temp_filename)
            wav_filename = temp_filename.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_filename, format='wav')
            with open(wav_filename, 'rb') as f:
                audio_data = f.read()
            os.remove(temp_filename)
            os.remove(wav_filename)
        else:
            with open(temp_filename, 'rb') as f:
                audio_data = f.read()
            os.remove(temp_filename)

        # Start the pipeline in the background using socketio's background task
        socketio.start_background_task(
            pipeline.process_audio, 
            audio_data, 
            session_id,
            auto_mode
        )

        # Return immediately with the session ID
        return jsonify({
            "success": True,
            "session_id": session_id,
            "message": "Processing started in background"
        })
    except Exception as e:
        print(f"Error in process_audio_parallel: {e}")
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route("/select_completion", methods=["POST"])
def route_select_completion():
    """
    Handle selection of a completion for TTS conversion.
    """
    try:
        data = request.json
        session_id = data.get("session_id")
        completion_index = data.get("completion_index", 0)
        completion_text = data.get("completion_text", "")
        
        # Check that we have either a valid index or text
        if not completion_text and completion_index < 0:
            return jsonify({
                "error": "No completion selected", 
                "success": False
            }), 400
            
        # Find the pipeline for this session
        pipeline_id = None
        for pid in pipeline.active_pipelines:
            if pid.startswith(session_id):
                pipeline_id = pid
                break
                
        if not pipeline_id:
            return jsonify({
                "error": "No active pipeline found for this session",
                "success": False
            }), 404
            
        # Get completions from the pipeline
        pipe_data = pipeline.active_pipelines[pipeline_id]
        completions = pipe_data.get("llm_responses", [])
        transcript = pipe_data.get("transcript", "") 
        # PATCH: Ensure completions is always a list
        if isinstance(completions, str):
            completions = [completions]
        elif not isinstance(completions, list):
            completions = [f"Error: Unexpected completions type: {type(completions)}"]
        # Get selected completion (either by index or text)
        selected_completion = None
        if completion_text:
            selected_completion = completion_text
        elif completion_index >= 0 and completion_index < len(completions):
            selected_completion = completions[completion_index]
        else:
            return jsonify({
                "error": "Invalid completion index",
                "success": False
            }), 400
            
        # Start TTS generation for the selected completion in the background
        socketio.start_background_task(
            process_selected_completion,
            pipeline_id,
            session_id, 
            selected_completion,
            transcript
        )
        
        # Return success to client
        return jsonify({
            "success": True,
            "message": "Processing selected completion"
        })
        
    except Exception as e:
        print(f"Error in select_completion: {e}")
        print(traceback.format_exc())
        
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

##############################################################################
# SOCKETIO EVENT HANDLERS
##############################################################################

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    # Add the client to a room with their session ID
    join_room(request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    leave_room(request.sid)

@socketio.on('cancel_pipeline')
def handle_cancel_pipeline(data):
    """Handle cancellation of an active pipeline"""
    session_id = data.get('session_id')
    if session_id in pipeline.active_pipelines:
        print(f"Cancelling pipeline for session {session_id}")
        # TODO: Implement cancellation logic
        # This would require adding cancellation signals to the pipeline tasks

##############################################################################
# RUN THE APP
##############################################################################

if __name__ == "__main__":
    print("Starting Bernard Web UI server...")
    print(f"Serving static files from: {STATIC_FOLDER}")
    print(f"Audio files will be saved to: {AUDIO_FOLDER}")
    print(f"Knowledge base directory: {KB_DIR}")
    print(f"Vector store directory: {VECTOR_STORE_DIR}")
    print(f"Chat history JSON: {CHAT_HISTORY_JSON}")
    
    # Make sure all necessary directories exist
    os.makedirs(KB_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    
    # Initialize chat history
    initialize_chat_history()
    
    
    # Print the server URL BEFORE starting the server
    print(f"Server will be running on http://0.0.0.0:5000")
    print("Press Ctrl+C to stop the server.")
    
    # Clean up audio files at startup
    print("[CLEANUP] Cleaning up audio files at startup...")
    try:
        if os.path.exists(AUDIO_FOLDER):
            for filename in os.listdir(AUDIO_FOLDER):
                file_path = os.path.join(AUDIO_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"[CLEANUP] Cleaned up all audio files in {AUDIO_FOLDER} at startup")
        else:
            print(f"[CLEANUP] AUDIO_FOLDER does not exist at startup: {AUDIO_FOLDER}")
    except Exception as e:
        print(f"[CLEANUP] Error cleaning up audio files at startup: {e}")
    
    try:
        # Run the Flask app with SocketIO
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Prevents double initialization in debug mode
        )
    except KeyboardInterrupt:
        print("\nShutting down server...")
        # Cleanup code can go here if needed
    except Exception as e:
        print(f"\nError running server: {e}")
        print(traceback.format_exc())

@app.route('/favicon.ico')
def favicon():
    return '', 204

def cleanup_audio_folder():
    print("[CLEANUP] Attempting to clean up audio files...")
    try:
        if os.path.exists(AUDIO_FOLDER):
            for filename in os.listdir(AUDIO_FOLDER):
                file_path = os.path.join(AUDIO_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"[CLEANUP] Cleaned up all audio files in {AUDIO_FOLDER}")
        else:
            print(f"[CLEANUP] AUDIO_FOLDER does not exist: {AUDIO_FOLDER}")
    except Exception as e:
        print(f"[CLEANUP] Error cleaning up audio files: {e}")

def handle_exit_signal(signum, frame):
    print(f"[CLEANUP] Received signal {signum}, running cleanup...")
    cleanup_audio_folder()
    import sys
    sys.exit(0)

atexit.register(cleanup_audio_folder)
signal.signal(signal.SIGINT, handle_exit_signal)
signal.signal(signal.SIGTERM, handle_exit_signal)

SETTINGS_FILE = 'settings.json'

def load_settings_from_file():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
            print(f"Loaded settings from {SETTINGS_FILE}: {settings}")
            return settings
        else:
            print(f"Settings file {SETTINGS_FILE} does not exist.")
            return None
    except Exception as e:
        print(f"Error loading settings from file: {e}")
        return None

def save_settings_to_file(settings):
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"Settings saved to {SETTINGS_FILE}: {settings}")
        return True
    except Exception as e:
        print(f"Error saving settings to file: {e}")
        return False

def create_default_settings():
    settings = {
        "voice": {
            "apiKey": "",
            "voiceId": "CwhRBWXzGAHq8TQ4Fs17",
            "model": "eleven_multilingual_v2",
            "outputFormat": "mp3_44100_192",
            "seed": 8675309,
            "voiceSettings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": 0.9,
                "style": 0.3
            }
        },
        "llm": {
            "openai": {
                "apiKey": "",
                "model": "gpt-4.1-mini"
            },
            "groq": {
                "apiKey": "",
                "model": "llama-3.3-70b-specdec"
            },
            "perplexity": {
                "apiKey": "",
                "model": "sonar"
            },
            "provider": "openai",
            "temperature": 0.9,
            "maxTokens": 150,
            "completionsCount": 4
        },
        "recorder": {
            "silenceThreshold": 500,
            "silenceDuration": 5.0,
            "minRecordingDuration": 1.0,
            "maxRecordingDuration": 60.0,
            "useNoiseReduction": True
        },
        "system": {
            "systemPrompt": "You are the persona Bernard Muller...",
            "fileLocations": {
                "kbDir": "data/kb",
                "vectorStoreDir": "data/vector_store",
                "chatVectorStoreDir": "data/chat_vector_store",
                "chatHistoryFile": "data/chat_history.json",
                "phrasesFile": "data/phrases.json"
            },
            "language": {
                "defaultLanguage": "auto",
                "useLanguageDetection": True
            },
            "debugMode": False
        }
    }
    print(f"Created default settings: {settings}")
    return settings

# --- SETTINGS JSON MIGRATION START ---
# Remove os.getenv for user-facing settings, use settings.json for all settings except secrets
# Only API keys (OPENAI_API_KEY, ELEVENLABS_API_KEY, etc.) may remain in .env

def ensure_settings_json():
    if not os.path.exists(SETTINGS_FILE):
        # Use the default structure from create_default_settings
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(create_default_settings(), f, indent=2)
        print(f"Created default settings.json at {SETTINGS_FILE}")

# Call this at startup
ensure_settings_json()

# Refactor all settings access to use settings.json
# Example usage: settings = load_settings_from_file(); settings['recorder']['silenceThreshold']

# Remove init_from_settings and all os.getenv for user-facing settings
# Refactor all code that uses os.getenv for settings to use load_settings_from_file()

# Example for recorder:
# settings = load_settings_from_file()
# recorder.silence_threshold = settings['recorder']['silenceThreshold']
# ...

# Example for voice:
# settings = load_settings_from_file()
# voice_id = settings['voice']['voiceId']
# ...

# Example for LLM:
# settings = load_settings_from_file()
# model = settings['llm']['openai']['model']
# ...

# Remove all os.getenv usage for settings below
# --- SETTINGS JSON MIGRATION END ---


# In[ ]:

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    print("RequestEntityTooLarge triggered!")
    print("Request content length:", getattr(request, 'content_length', 'unknown'))
    return (
        jsonify({
            "error": "Recording too large (max 32 MB, server limit). Please try again.",
            "success": False
        }),
        413,
    )

