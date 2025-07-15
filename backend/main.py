import logging
import sys
import os
import subprocess
import time
import threading
import asyncio
import wave
import base64
import io
import uuid
import shutil
import signal
import tempfile
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import psutil
from PIL import Image
import docx
import PyPDF2
import webbrowser
from gtts import gTTS
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel

# Optionally imported modules
try:
    import pyautogui
except ImportError:
    pyautogui = None
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# Configure root logger to output to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure logs go to stdout
    ]
)
logger = logging.getLogger(__name__)

# Add a file handler for persistent logs
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Set debug logging for specific modules
logging.getLogger('whisper').setLevel(logging.WARNING)  # Reduce whisper logs
logging.getLogger('torch').setLevel(logging.WARNING)    # Reduce torch logs

# For offline speech recognition and processing
import speech_recognition as sr
from faster_whisper import WhisperModel
import torch
import torchaudio
# import cv2
import numpy as np
from PIL import Image
import sounddevice as sd
import soundfile as sf
import warnings
# import pyttsx3  # Commented out - replaced with Piper TTS
import pyautogui
import webbrowser
import docx
import PyPDF2
import shutil
import psutil
import signal
import tempfile

# Piper TTS imports
try:
    import sounddevice as sd
    import soundfile as sf
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    print("Warning: sounddevice/soundfile not available for audio playback")

# Suppress warnings
warnings.filterwarnings("ignore")

# Global variables
app = FastAPI()
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# Piper TTS configuration
PIPER_MODEL_PATH = "C:/Users/ankit/piper/en_US-lessac-medium.onnx"
PIPER_CONFIG_PATH = "C:/Users/ankit/piper/en_en_US_lessac_medium_en_US-lessac-medium.onnx.json"

def check_piper_availability():
    """Check if Piper TTS is available and properly configured."""
    try:
        # Check if piper command is available
        result = subprocess.run(["piper", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("Piper TTS is available")
            return True
        else:
            logger.error("Piper TTS command failed")
            return False
    except FileNotFoundError:
        logger.error("Piper TTS not found. Please install Piper TTS first.")
        return False
    except Exception as e:
        logger.error(f"Error checking Piper TTS availability: {e}")
        return False

def initialize_piper():
    """Initialize Piper TTS with the Lessac voice model."""
    if not check_piper_availability():
        raise RuntimeError("Piper TTS is not available")
    
    # Check if the model file exists
    model_path = Path(PIPER_MODEL_PATH)
    config_path = Path(PIPER_CONFIG_PATH)
    
    if not model_path.exists():
        logger.warning(f"Piper model file not found: {model_path}")
        logger.info("You may need to download the en_US-lessac-medium.onnx model")
    
    if not config_path.exists():
        logger.warning(f"Piper config file not found: {config_path}")
    
    logger.info("Piper TTS initialized with en_US-lessac-medium model")
    return True

# Initialize Piper TTS on startup
try:
    initialize_piper()
    PIPER_READY = True
except Exception as e:
    logger.error(f"Failed to initialize Piper TTS: {e}")
    PIPER_READY = False

# Initialize fast-whisper model
whisper_model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "float32")

class VideoAssistant:
    def __init__(self):
        self.is_listening = False
        self.audio_queue = asyncio.Queue()
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.recording = []
        self.stream = None
        self.use_gpu = True  # Default to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        # Initialize FastWhisper model
        self.whisper_model = None
        self.load_whisper_model()
    
    def load_whisper_model(self):
        """Load or reload the FastWhisper model with current device settings"""
        try:
            logger.info(f"Starting model load. Requested GPU: {self.use_gpu}")
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            if cuda_available:
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            target_device = "cuda" if (self.use_gpu and cuda_available) else "cpu"
            compute_type = "float16" if target_device == "cuda" else "float32"
            logger.info(f"Target device: {target_device.upper()} | Compute type: {compute_type}")
            if hasattr(self, 'device') and self.device == target_device and self.whisper_model is not None:
                logger.info(f"Model already loaded on {target_device.upper()}, skipping reload")
                return
            self.device = target_device
            self.compute_type = compute_type
            if self.whisper_model is not None:
                del self.whisper_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            self.whisper_model = WhisperModel("base", device=self.device, compute_type=self.compute_type)
            print("\n" + "="*50)
            print(f"FAST-WHISPER MODEL SUCCESSFULLY LOADED ON: {self.device.upper()}")
            if self.device == "cuda":
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            else:
                print("Using CPU for processing")
            print(f"Model: base")
            print(f"Precision: {'FP16' if (self.device == 'cuda') else 'FP32'}")
            print("="*50 + "\n")
        except Exception as e:
            print(f"\n!!! ERROR LOADING MODEL ON {self.device.upper()}: {str(e)}")
            if self.device == "cuda":
                print("Falling back to CPU...")
                self.use_gpu = False
                self.device = "cpu"
                self.compute_type = "float32"
                self.load_whisper_model()
            else:
                raise
    
    def set_processing_mode(self, use_gpu):
        """Switch between GPU and CPU processing"""
        if self.use_gpu != use_gpu:
            print("\n" + "-"*50)
            print(f"REQUESTED PROCESSING MODE: {'GPU' if use_gpu else 'CPU'}")
            
            # Check if requested device is available
            if use_gpu and not torch.cuda.is_available():
                print("Warning: CUDA is not available. Using CPU instead.")
                use_gpu = False
            
            self.use_gpu = use_gpu
            
            # Clear CUDA cache if switching from GPU to free up memory
            if self.device == 'cuda' and not use_gpu:
                torch.cuda.empty_cache()
                
            try:
                # Log before switching
                print("\n" + "="*50)
                print(f"BEFORE DEVICE SWITCH - Current: {self.device.upper()}, Target: {'cuda' if use_gpu else 'cpu'}")
                if self.device == "cuda":
                    print(f"GPU Memory Before Clear: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                
                # Load the model with new settings
                self.load_whisper_model()
                
                # Log after switching
                print(f"AFTER DEVICE SWITCH - Current: {self.device.upper()}")
                if self.device == "cuda":
                    print(f"GPU Memory After Clear: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print("="*50 + "\n")
                
                print(f"Successfully switched to {self.device.upper()} processing")
                print("-"*50 + "\n")
                
                # Log full device status
                self._log_device_status("after_device_switch")
                
                return True
                
            except Exception as e:
                print(f"Error switching to {'GPU' if use_gpu else 'CPU'}: {str(e)}")
                # Revert to CPU if GPU switching fails
                if use_gpu:
                    print("Falling back to CPU")
                    self.use_gpu = False
                    self.load_whisper_model()
                    
                    # Log fallback status
                    self._log_device_status("after_fallback_to_cpu")
                    
                print("-"*50 + "\n")
                return False
        else:
            print(f"Already using {'GPU' if use_gpu else 'CPU'} processing")
            self._log_device_status("device_check")
            
        return False
        
    async def start_listening(self):
        """Start listening to the microphone"""
        if self.is_listening:
            return
            
        self.is_listening = True
        self.recording = []
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            if self.is_listening:
                self.recording.append(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=audio_callback,
            blocksize=self.chunk_size
        )
        self.stream.start()
        
    def _log_device_status(self, operation: str):
        """Log the current device status"""
        try:
            # Create log message
            log_lines = [
                "\n" + "="*60,
                f"DEVICE STATUS - {operation.upper()}",
                "-"*60,
                f"Selected Device: {'GPU' if self.use_gpu else 'CPU'}",
                f"Actual Device: {self.device.upper()}"
            ]
            
            # Add GPU-specific info if available
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                mem_alloc = torch.cuda.memory_allocated() / 1024**2
                mem_cached = torch.cuda.memory_reserved() / 1024**2
                log_lines.extend([
                    f"GPU Name: {gpu_name}",
                    f"GPU Memory Allocated: {mem_alloc:.2f} MB",
                    f"GPU Memory Cached: {mem_cached:.2f} MB"
                ])
            
            log_lines.append("="*60 + "\n")
            
            # Log using both print and logger
            log_message = "\n".join(log_lines)
            print(log_message, flush=True)  # Force flush
            logger.info(f"Device Status - {operation}: Using {self.device.upper()}")
            
        except Exception as e:
            error_msg = f"Error in _log_device_status: {str(e)}"
            print(error_msg, flush=True)
            logger.error(error_msg, exc_info=True)

    async def stop_listening(self) -> Optional[str]:
        """Stop listening and return the transcribed text"""
        if not self.is_listening:
            return None
            
        self.is_listening = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if not self.recording:
            return None
            
        # Save recording to a temporary WAV file
        audio_data = np.concatenate(self.recording, axis=0)
        temp_wav = UPLOADS_DIR / f"temp_recording_{uuid.uuid4()}.wav"
        sf.write(temp_wav, audio_data, self.sample_rate)
        
        try:
            # Log device status before processing
            self._log_device_status("before_transcription")
            
            # Log transcription info
            print("\n" + "*"*50)
            print(f"STARTING TRANSCRIPTION ON: {self.device.upper()}")
            if self.device == "cuda":
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"Audio file: {temp_wav}")
            print("*"*50 + "\n")
            
            start_time = time.time()
            
            # Ensure we're using the correct device
            if self.device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.use_gpu = False
                self.device = "cpu"
                self.compute_type = "float32"
                self.load_whisper_model()
            
            # Transcribe using FastWhisper
            segments, info = self.whisper_model.transcribe(
                str(temp_wav),
                beam_size=5,
                language="en"  # Force English for better performance
            )
            
            # Log transcription completion
            elapsed = time.time() - start_time
            print("\n" + "*"*50)
            print(f"TRANSCRIPTION COMPLETED IN {elapsed:.2f} SECONDS")
            print(f"Device: {self.device.upper()}")
            if self.device == "cuda":
                print(f"GPU Memory After: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Text: {segments[0].text}")
            print("*"*50 + "\n")
            
            # Log device status after processing
            self._log_device_status("after_transcription")
            
            return segments[0].text.strip()
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None
        finally:
            # Clean up
            if temp_wav.exists():
                temp_wav.unlink()
    
    async def process_image(self, image_data: bytes, prompt: str) -> str:
        """Process image using LLaVA model"""
        try:
            # Save image to a temporary file
            img_path = UPLOADS_DIR / f"temp_img_{uuid.uuid4()}.jpg"
            with open(img_path, "wb") as f:
                f.write(image_data)
            
            # Prepare the prompt for LLaVA
            full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            
            # Call LLaVA model using subprocess
            cmd = [
                "ollama", "run", "llava:latest",
                "--prompt", full_prompt,
                "--images", str(img_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 seconds timeout
            )
            
            # Clean up
            if img_path.exists():
                img_path.unlink()
                
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"LLaVA error: {result.stderr}")
                return "I'm sorry, I couldn't process that image right now."
                
        except Exception as e:
            print(f"Error in image processing: {e}")
            return "I encountered an error while processing the image."
    
    async def text_to_speech(self, text: str):
        """Convert text to speech using Piper TTS with female voice"""
        try:
            # Generate a unique filename
            output_path = UPLOADS_DIR / f"tts_{uuid.uuid4()}.wav"
            
            # Use female voice model (en_US-lessac-medium is a female voice)
            success = await asyncio.to_thread(
                generate_speech_with_piper,
                text=text,
                output_path=output_path,
                voice_model="en_US-lessac-medium"
            )
            
            if success and output_path.exists():
                return output_path
            else:
                logger.error(f"Failed to generate TTS audio: {output_path}")
                
        except Exception as e:
            logger.error(f"Error in TTS: {e}")
        
        return None

# Initialize FastWhisper model with device selection
def load_whisper_model(use_gpu=True):
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    print(f"Loading FastWhisper model with device: {device}")
    model = WhisperModel("base", device=device, compute_type=compute_type)
    print(f"FastWhisper model loaded successfully on {device.upper()}")
    return model, device

# Initialize FastWhisper model with default device (GPU if available)
whisper_model, current_device = load_whisper_model(use_gpu=True)

# Initialize the video assistant
video_assistant = VideoAssistant()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket for real-time communication
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

manager = ConnectionManager()

# API Endpoints
@app.post("/api/start_listening")
async def start_listening():
    """Start listening to the microphone"""
    await video_assistant.start_listening()
    return {"status": "listening"}

@app.post("/api/stop_listening")
async def stop_listening():
    """Stop listening and return the transcribed text"""
    text = await video_assistant.stop_listening()
    return {"text": text if text else ""}

@app.post("/api/process_image")
async def process_image(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    """Process an image with the given prompt"""
    image_data = await image.read()
    response = await video_assistant.process_image(image_data, prompt)
    return {"response": response}

@app.get("/api/tts/{text}")
async def text_to_speech(text: str):
    """Convert text to speech and return the audio file"""
    audio_path = await video_assistant.text_to_speech(text)
    if audio_path and audio_path.exists():
        return FileResponse(audio_path, media_type="audio/wav")
    raise HTTPException(status_code=500, detail="Failed to generate speech")

# WebSocket endpoint
@app.websocket("/ws/video_assistant/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle processing mode change
            if message.get("type") == "set_processing_mode":
                use_gpu = message.get("use_gpu", True)
                changed = video_assistant.set_processing_mode(use_gpu)
                if changed:
                    await websocket.send_text(json.dumps({
                        "type": "processing_mode_changed",
                        "use_gpu": use_gpu,
                        "message": f"Switched to {'GPU' if use_gpu else 'CPU'} processing"
                    }))
                continue
                
            if message.get("type") == "start_listening":
                await video_assistant.start_listening()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "status": "listening"
                }))
            elif message.get("type") == "stop_listening":
                text = await video_assistant.stop_listening()
                if text:
                    await manager.send_message(
                        json.dumps({"type": "transcription", "text": text}),
                        client_id
                    )
                
            elif message["type"] == "process_image":
                image_data = base64.b64decode(message["image"].split(",")[1])
                prompt = message.get("prompt", "What's in this image?")
                response = await video_assistant.process_image(image_data, prompt)
                await manager.send_message(
                    json.dumps({"type": "response", "text": response}),
                    client_id
                )
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(client_id)

from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import requests
from pathlib import Path
import logging
import subprocess
import signal
import psutil
import time
import json
import threading
import shutil
import PyPDF2
import uuid
import docx
import base64
from docx import Document
import asyncio
import io
import pyttsx3

# Configure logging to show INFO level messages in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models for API requests ---
class DownloadRequest(BaseModel):
    model_id: str
    file_name: str

class ChatRequest(BaseModel):
    model_file: str
    prompt: str

class ChatMessage(BaseModel):
    message: str
    model: str

# --- Constants ---
MODELS_DIR = Path("models")
UPLOADS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "uploads"

# --- FastAPI App Initialization ---
app = FastAPI()

# Ensure uploads directory exists on startup
@app.on_event("startup")
async def startup_event():
    """Initialize application services."""
    try:
        # Create uploads directory if it doesn't exist
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured uploads directory exists at: {UPLOADS_DIR}")
    except Exception as e:
        logger.error(f"Failed to create uploads directory: {e}")
        raise

# Directory for downloaded models
MODELS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# --- CORS Middleware (Good practice, even when serving static files) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

ollama_process = None
ollama_lock = threading.Lock()

# In-memory storage for extracted file texts
extracted_texts = {}

# --- Whisper and Voice Assistant Setup ---
try:
    # Using FastWhisper for all transcription tasks. For higher accuracy, use larger FastWhisper models if needed.
    whisper_model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "float32")
    logger.info("FastWhisper 'base' model loaded successfully.")
except Exception as e:
    whisper_model = None
    logger.error(f"Could not load whisper model: {e}")

tasks = [] # In-memory task list for the assistant

def get_tts_engine(engine_type='piper'):
    """
    Initializes and returns a TTS engine based on the specified type.
    
    Args:
        engine_type (str): The type of TTS engine to use ('piper' or 'gtts')
        
    Returns:
        dict: Dictionary containing the engine and its type
    """
    logger.info(f"TTS Engine: {engine_type} selected")
    if engine_type == 'piper':
        if not PIPER_READY:
            raise RuntimeError("Piper TTS is not ready. Please check installation.")
        
        logger.info("Piper TTS engine initialized with en_US-lessac-medium model")
        return {'type': 'piper', 'engine': 'piper'}
    
    elif engine_type == 'gtts':
        return {'type': 'gtts', 'engine': None}  # gTTS doesn't need engine initialization
    
    else:
        raise ValueError(f"Unsupported TTS engine type: {engine_type}")

def text_to_speech_google(text: str) -> str:
    """Convert text to speech using gTTS and return the path to the audio file."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            # Generate speech
            tts = gTTS(text=text, lang='en')
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        logger.error(f"Error in gTTS: {str(e)}")
        raise

def handle_known_commands(command: str):
    """Processes known voice commands and returns a text response if a command is matched."""
    command = command.lower()
    if "add a task" in command:
        # This part is tricky in a streaming setup. For now, we'll just acknowledge.
        # A more robust solution would require a state machine.
        # task = listen_for_command() # This needs to be handled in the client
        tasks.append(command.replace("add a task", "").strip())
        return f"Added task: {tasks[-1]}. You now have {len(tasks)} tasks."

    elif "list tasks" in command:
        if tasks:
            response = "Here are your tasks: " + ", ".join(tasks)
            return response
        else:
            return "You have no tasks."

    elif "take a screenshot" in command:
        try:
            screenshot_path = os.path.join(UPLOADS_DIR, "screenshot.png")
            pyautogui.screenshot(screenshot_path)
            return f"Screenshot taken and saved to {screenshot_path}"
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return "I was unable to take a screenshot."

    elif "open chrome" in command or "open youtube" in command:
        try:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube."
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")
            return "I was unable to open the web browser."

    return None  # No known command matched

def is_ollama_running():
    # Check if any process is listening on port 11434
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'ollama' in proc.info['name'].lower():
                for arg in proc.info['cmdline']:
                    if 'serve' in arg:
                        return True
        except Exception:
            continue
    return False

# --- API Endpoints ---

from fastapi import Body
from fastapi.responses import JSONResponse, StreamingResponse
import re

@app.post('/tts')
async def tts_endpoint(payload: dict = Body(...)):
    """
    Convert text to speech using Piper TTS and return the audio file.
    Input: {'text': 'your text here', 'voice': 'model_id'}
    Output: Audio file in WAV format
    """
    text = payload.get('text', '').strip()
    voice_model = payload.get('voice', 'en_US-lessac-medium')  # Default to Lessac voice
    
    if not text:
        return JSONResponse(
            {'error': 'No text provided.'}, 
            status_code=400
        )
    
    # Create a temporary file in the system's temp directory
    temp_dir = Path(tempfile.gettempdir()) / 'somasai_tts'
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate a unique filename
    temp_file = temp_dir / f'tts_{uuid.uuid4()}.wav'
    
    try:
        # Use Piper TTS with the specified voice model
        await asyncio.to_thread(save_tts_audio, 'piper', text, temp_file, voice_model)
        
        async def iterfile():
            with open(temp_file, 'rb') as f:
                while chunk := f.read(8192):
                    yield chunk
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")
                
        return StreamingResponse(
            iterfile(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=tts_output.wav",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
            
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return JSONResponse(
            {'error': f'Failed to generate speech: {str(e)}'},
            status_code=500
        )
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/analyze_image")
def analyze_image(payload: dict = Body(...)):
    """
    Accepts {"prompt": str, "image": base64 string (dataURL)}
    Uses llava:latest to analyze image and prompt, returns response text.
    """
    prompt = payload.get("prompt", "")
    image_data_url = payload.get("image", "")
    logger.info(f"[analyze_image] Received image data, length: {len(image_data_url)}")
    logger.info(f"[analyze_image] image_data_url (first 100 chars): {image_data_url[:100]}")
    if not prompt or not image_data_url:
        return JSONResponse({"response": "Missing prompt or image."}, status_code=400)

    # Extract base64 from data URL
    match = re.match(r"data:image/[^;]+;base64,(.*)", image_data_url)
    if not match:
        return JSONResponse({"response": "Invalid image data."}, status_code=400)
    img_base64 = match.group(1)

    # Save image to temp file
    img_bytes = base64.b64decode(img_base64)
    temp_img_path = os.path.join(UPLOADS_DIR, f"video_assistant_{uuid.uuid4().hex}.jpg")
    with open(temp_img_path, "wb") as f:
        f.write(img_bytes)

    # Prepare payload for llava with concise response instructions
    model = "llava:latest"
    ollama_url = "http://localhost:11434/api/generate"
    
    # Add instructions for concise response
    concise_prompt = (
        f"{prompt} "
        "Please provide a very brief response of exactly 2-3 lines. "
        "Be concise and to the point, focusing on the most important details. "
        "Do not exceed 3 lines under any circumstances."
    )
    
    payload = {
        "model": model,
        "prompt": concise_prompt,
        "images": [img_base64],
        "stream": False,
        "options": {
            "num_ctx": 1024,  # Reduced context for more focused responses
            "temperature": 0.1,  # Lower temperature for more focused responses
            "top_k": 20,
            "top_p": 0.7,
            "num_predict": 150,  # Limit response length
        }
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        # Get the raw response and ensure it's concise
        raw_response = data.get("response", "No response from model.").strip()
        
        # Post-process to ensure 2-3 lines
        lines = [line.strip() for line in raw_response.split('\n') if line.strip()]
        if len(lines) > 3:
            # If more than 3 lines, take first 3 and add "..."
            result = '\n'.join(lines[:3])
        else:
            result = '\n'.join(lines)
            
    except Exception as e:
        result = f"Vision model error: {e}"
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

    # TTS (optional, but for consistency)
    try:
        tts_engine = get_tts_engine()
        output_path = os.path.join(UPLOADS_DIR, f"tts_{uuid.uuid4().hex}.wav")
        save_tts_audio(tts_engine, result, output_path)
        # Optionally, return audio file URL or base64 audio if you want to play it on frontend
        # For now, just return text
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception as e:
        pass  # TTS failure should not block response

    return {"response": result}


@app.post("/download_model")
def download_model(req: DownloadRequest):
    url = f"https://huggingface.co/{req.model_id}/resolve/main/{req.file_name}"
    local_path = MODELS_DIR / req.file_name
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")
    return {"status": "success", "file": req.file_name}

@app.get("/list_models")
def list_models():
    """Lists the available models from the local Ollama server."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        # Ollama returns {"models": [{"name": "mistral"}, ...]}
        models = [m["name"] for m in data.get("models", [])]
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}

@app.post("/chat")
def chat_handler(data: dict):
    message = data.get("message", "")
    logger.info(f"Received chat message: {message}")
    model = data.get("model", "")
    if not message or not model:
        return {"response": "No message or model provided."}

    image_path = None
    prompt = message

    # Default token_count, will be overridden if multiple documents are found
    token_count = int(data.get("token_count", 2048))

    if extracted_texts:
        all_doc_texts = []
        # Collect text from all documents
        for filename, text in extracted_texts.items():
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in {'.pdf', '.txt', '.docx', '.doc'}:
                if text and text not in ("[DOCUMENT_NO_TEXT]", "[Error: The legacy .doc format is not supported. Please save the file as .docx and re-upload.]"):
                    all_doc_texts.append(text)

        if all_doc_texts:
            # Dynamically increase context size for multiple documents to avoid truncation
            if len(all_doc_texts) > 1:
                token_count = 16384  # A much larger context window

            # Add numbered separators to help the model distinguish between documents
            doc_context = ""
            for i, text in enumerate(all_doc_texts):
                doc_context += f"--- START OF DOCUMENT {i+1} ---\n{text}\n--- END OF DOCUMENT {i+1} ---\n\n"

            # Restructure the prompt to place the large context first, followed by the specific instruction.
            prompt = f"""Review the following document(s) and provide a specific answer to the question below. Do not summarize the content unless explicitly asked.\n\n{doc_context}Question: {message}"""

            # Log diagnostic information to the console
            logger.info(f"--- CHAT CONTEXT DIAGNOSTICS ---")
            logger.info(f"Number of documents in context: {len(all_doc_texts)}")
            logger.info(f"Total prompt length (characters): {len(prompt)}")
            logger.info(f"Context window (num_ctx) set to: {token_count}")
            logger.info(f"Prompt preview (first 200 chars): {prompt[:200]}")
            logger.info(f"Prompt preview (last 200 chars): {prompt[-200:]}")
            logger.info(f"---------------------------------")
        else:
            # No documents with text, fall back to checking for an image (last file uploaded)
            last_file = list(extracted_texts.keys())[-1]
            file_ext = os.path.splitext(last_file)[1].lower()
            if file_ext in {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}:
                image_path = str(UPLOADS_DIR / last_file)
                prompt = f"""Analyze the image and answer the following question: {message}"""

    # Advanced options from frontend
    device = data.get("device", "cpu")
    threads = int(data.get("threads", 4))
    gpu_cores = int(data.get("gpu_cores", 1))
    cpu_cores = int(data.get("cpu_cores", 4))

    # Add instruction to keep the response within 150 characters
    concise_instruction = "Respond in no more than 300 characters. Do not just truncate; formulate a concise answer within 150 characters."
    if "Question:" in prompt:
        prompt = prompt + "\n" + concise_instruction
    else:
        prompt = f"{prompt}\n{concise_instruction}"

    options = {
        "num_ctx": token_count,
        "num_thread": threads,
        "num_batch": 512,
        "use_mlock": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "num_gpu": 999,  # Offload all possible layers to the GPU
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }

    if image_path:
        try:
            with open(image_path, "rb") as img_file:
                img_bytes = img_file.read()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                payload["images"] = [img_base64]
        except Exception as e:
            return {"response": f"Error reading image file: {e}"}

    try:
        start_time = time.time()
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=400)
        response.raise_for_status()
        ollama_response = response.json()
        end_time = time.time()

        response_text = ollama_response.get("response", "").strip()
        MAX_CHARACTERS = 300
        if len(response_text) > MAX_CHARACTERS:
            # If Ollama did not follow the instruction, truncate as a fallback
            response_text = response_text[:MAX_CHARACTERS] + "..."

        time_taken = end_time - start_time
        tokens_per_second = ollama_response.get("eval_count", 0) / time_taken if time_taken > 0 else 0
        total_tokens = ollama_response.get("eval_count", 0)

        return {
            "response": response_text,
            "time_taken": time_taken,
            "tokens_per_second": tokens_per_second,
            "total_tokens": total_tokens,
            "characters": len(response_text),
            "model_name": model
        }
    except requests.exceptions.RequestException as e:
        return {"response": f"Could not connect to Ollama: {e}"}
    except Exception as e:
        return {"response": f"An error occurred: {e}"}

@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile = File(...)):
    if whisper_model is None:
        return JSONResponse({"error": "Whisper model not loaded"}, status_code=500)

    try:
        # Save uploaded audio file
        temp_input_path = os.path.join(UPLOADS_DIR, f"temp_{uuid.uuid4().hex}.wav")
        with open(temp_input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Transcribe using FastWhisper
        segments, info = whisper_model.transcribe(temp_input_path, beam_size=5, language="en")
        os.remove(temp_input_path)

        text = " ".join([segment.text for segment in segments]).strip()
        return {"text": text}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/get_model")
async def get_model(model_name: str):
    """Returns the path to a model file."""
    model_path = MODELS_DIR / model_name
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_name}")
    return {"path": str(model_path)}

@app.get("/api/local_models")
async def list_local_models():
    """List locally downloaded models from the MODELS_DIR folder"""
    logger.info(f"Listing local models from {MODELS_DIR}")
    try:
        local_models = []
        for file in MODELS_DIR.glob("*.gguf"):
            model_id = file.stem.replace("_", "/")
            print(model_id)
            local_models.append({
                "id": model_id,
                "name": model_id.split("/")[-1],
                "local_path": str(file),
                "format": "gguf",
                "size": file.stat().st_size / (1024 * 1024 * 1024)  # Convert to GB
            })
        return local_models
    except Exception as e:
        logger.error(f"Error listing local models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_ollama")
def start_ollama():
    global ollama_process
    with ollama_lock:
        if is_ollama_running():
            logger.info("Ollama server is already running")
            return {"status": "success", "message": "Ollama already running"}
        try:
            # Log system information before starting
            logger.info("\n" + "="*50)
            logger.info("STARTING OLLAMA SERVER")
            logger.info("="*50)
            
            # Log CPU and GPU information
            logger.info(f"Available CPU Cores: {os.cpu_count()}")
            logger.info(f"CPU Usage: {psutil.cpu_percent()}%")
            logger.info(f"Memory Usage: {psutil.virtual_memory().percent}%")
            
            if torch.cuda.is_available():
                logger.info("CUDA is available!")
                logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA Version: {torch.version.cuda}")
                logger.info(f"PyTorch Version: {torch.__version__}")
            else:
                logger.warning("CUDA is NOT available. Using CPU only.")
            
            # Start Ollama server
            logger.info("Starting Ollama server...")
            ollama_process = subprocess.Popen(["ollama", "serve"])
            time.sleep(2)  # Give it a moment to start
            
            if is_ollama_running():
                logger.info("Ollama server started successfully")
                logger.info("="*50 + "\n")
                return {"status": "success"}
            else:
                error_msg = "Failed to start Ollama server"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
                
        except Exception as e:
            error_msg = f"Error starting Ollama: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

@app.post("/stop_ollama")
def stop_ollama():
    global ollama_process
    with ollama_lock:
        if ollama_process is not None and ollama_process.poll() is None:
            logger.info("\n" + "="*50)
            logger.info("STOPPING OLLAMA SERVER")
            logger.info("="*50)
            
            # Log resource usage before stopping
            logger.info(f"CPU Usage before stop: {psutil.cpu_percent()}%")
            logger.info(f"Memory Usage before stop: {psutil.virtual_memory().percent}%")
            
            # Stop the process
            ollama_process.terminate()
            try:
                ollama_process.wait(timeout=5)  # Wait up to 5 seconds for clean shutdown
            except subprocess.TimeoutExpired:
                logger.warning("Ollama process did not terminate gracefully, forcing...")
                ollama_process.kill()
                
            ollama_process = None
            logger.info("Ollama server stopped successfully")
            logger.info("="*50 + "\n")
            return {"status": "success"}
            
        # If Ollama was not started by this backend, do not try to kill it
        logger.warning("Ollama is not running (or not managed by backend)")
        return {"status": "error", "message": "Ollama is not running (or not managed by backend)"}

@app.get("/ollama_status")
def ollama_status():
    global ollama_process
    with ollama_lock:
        if ollama_process is not None and ollama_process.poll() is None:
            return {"status": "running"}
        return {"status": "stopped"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs(UPLOADS_DIR, exist_ok=True)

        file_ext = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(UPLOADS_DIR, unique_filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        text = ""
        file_type = 'document'

        if file_ext in {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}:
            file_type = 'image'
            text = "[IMAGE]"  # Placeholder for images
        elif file_ext == '.pdf':
            try:
                with open(file_path, "rb") as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = "\n".join(page.extract_text() or '' for page in reader.pages)
            except Exception as e:
                text = f"[Error extracting PDF text: {e}]"
        elif file_ext == '.txt':
            with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                text = f.read()
        elif file_ext == '.docx':
            try:
                doc = docx.Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
            except Exception as e:
                logger.error(f"Error extracting .docx file {file.filename}: {e}")
                text = f"[Error extracting DOCX text: {e}]"
        elif file_ext == '.doc':
            text = "[Error: The legacy .doc format is not supported. Please save the file as .docx and re-upload.]"

        # Store the extracted text and metadata
        if not text:
            extracted_texts[unique_filename] = {
                'text': "[DOCUMENT_NO_TEXT]",
                'original_name': file.filename,
                'pages': []
            }
        else:
            # Split text into pages if it's a PDF
            pages = []
            if file_ext == '.pdf':
                with open(file_path, "rb") as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    pages = [page.extract_text() or '' for page in reader.pages]
            
            extracted_texts[unique_filename] = {
                'text': text,
                'original_name': file.filename,
                'pages': pages
            }

        return {"filename": unique_filename, "filetype": file_type}
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    try:
        file_path = UPLOADS_DIR / filename
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        os.remove(file_path)
        extracted_texts.pop(filename, None)  # Remove from our in-memory store

        return {"status": "success", "message": f"File {filename} deleted."}
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def get_tts_engine():
    """Initialize and return a Piper TTS engine with the Lessac voice."""
    try:
        print("\n=== INITIALIZING PIPER TTS ENGINE ===")
        print("Attempting to initialize Piper TTS engine...")
        
        if not PIPER_READY:
            raise RuntimeError("Piper TTS is not ready. Please check installation.")
        
        # Log Piper TTS information
        logger.info("--- Piper TTS Configuration ---")
        logger.info(f"Model: {PIPER_MODEL_PATH}")
        logger.info(f"Config: {PIPER_CONFIG_PATH}")
        logger.info("Voice: en_US-lessac-medium (Female)")
        
        print(f"\n=== TTS VOICE SELECTED ===")
        print(f"Piper TTS voice: en_US-lessac-medium")
        print(f"Voice type: Female (Lessac)")
        print(f"Model file: {PIPER_MODEL_PATH}")
        print("==========================\n")
        
        print("Piper TTS engine initialized successfully!")
        print("==============================\n")
        return 'piper'  # Return a simple identifier for Piper
        
    except Exception as e:
        logger.error(f"Error initializing Piper TTS engine: {e}")
        print(f"\n!!! ERROR initializing Piper TTS engine: {e} !!!\n")
        raise

def save_tts_audio(engine, text, output_path, voice_model=None):
    """Synchronous function to save TTS audio to a file using Piper TTS."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # New: pass voice_model to generate_speech_with_piper
        success = generate_speech_with_piper(text, output_path, voice_model)
        if not success:
            raise IOError("Failed to generate TTS audio file with Piper")
    except Exception as e:
        logger.error(f"Error in save_tts_audio: {str(e)}")
        raise

def generate_speech_with_piper(text, output_path, voice_model=None):
    """
    Generate speech using Piper TTS with the selected voice model.
    Args:
        text (str): The text to synthesize.
        output_path (Path or str): The path to save the output WAV file.
        voice_model (str): The model_id of the selected voice (e.g., 'en_US-lessac-medium').
    Returns:
        bool: True if successful, False otherwise.
    """
    import subprocess
    import tempfile
    from pathlib import Path
    import os
    # Use the specified piper voices directory
    voices_dir = r"C:/Users/ankit/piper/voices"
    # Default to Lessac if not provided
    model_id = voice_model or "en_US-lessac-medium"
    model_path = os.path.join(voices_dir, f"{model_id}.onnx")
    config_path = os.path.join(voices_dir, f"en_en_US_{model_id}_en_US-{model_id}.onnx.json")
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    if not os.path.exists(config_path):
        # Try fallback: find any .json config for this model
        configs = [f for f in os.listdir(voices_dir) if f.endswith('.json') and model_id in f]
        if configs:
            config_path = os.path.join(voices_dir, configs[0])
        else:
            logger.error(f"Config file not found for model: {model_id}")
            return False
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as tf:
        tf.write(text)
        temp_text_file = tf.name
    try:
        ps_cmd = f'Get-Content "{temp_text_file}" | piper --model "{model_path}" --config "{config_path}" --output_file "{output_path}"'
        result = subprocess.run([
            "powershell", "-Command", ps_cmd
        ], capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            return True
        else:
            logger.error(f"Piper TTS failed: {result.stderr}\n{result.stdout}")
            return False
    except Exception as e:
        logger.error(f"Error running Piper TTS: {e}")
        return False
    finally:
        try:
            Path(temp_text_file).unlink()
        except Exception:
            pass

import json #changed from pyjson
import re
from gtts import gTTS
import tempfile
import os

def clean_text(text):
    """Remove special characters, emojis, and extra whitespace from text."""
    if not text:
        return ""
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s.,!?\-]', ' ', text)
    # Remove asterisks and other special characters
    text = re.sub(r'[\*_~`]', '', text)
    return text

async def chat_handler_logic(model_file, prompt):
    """Logic for handling chat, extracted to be reusable by WebSocket."""
    if not is_ollama_running():
        logger.error("Ollama is not running. Please start it before using chat.")
        yield "Error: Ollama is not running. Please start it from the main page."
        return

    doc_context = ""
    if extracted_texts:
        doc_context = ""
        # Collect text from all documents with proper references
        for file_id, doc_data in extracted_texts.items():
            text = doc_data['text']
            original_name = doc_data['original_name']
            doc_name = os.path.splitext(original_name)[0]  # Remove file extension
            pages = doc_data.get('pages', [])
            
            if text and text not in ("[DOCUMENT_NO_TEXT]", "[Error: The legacy .doc format is not supported. Please save the file as .docx and re-upload.]"):
                # Add document reference with original name
                doc_header = f"--- {doc_name} ---\n"
                
                # If we have page information from PDF, include it in the context
                if pages:
                    for page_num, page_text in enumerate(pages, 1):
                        if page_text.strip():
                            doc_context += f"{doc_header}Page {page_num}:\n{page_text}\n\n"
                else:
                    doc_context += f"{doc_header}{text}\n\n"

            # Restructure the prompt to place the large context first, followed by the specific instruction.
            # Add a system prompt to guide the AI's persona
    # System message to ensure complete responses
    system_msg = """You are a helpful AI assistant. Please provide complete, concise responses that naturally conclude within the character limit. 
If the answer is complex, provide a clear summary first, then key points. 
End your response naturally when complete. Do not cut off mid-sentence."""
    
    # Update prompt with system message and context
    if doc_context:
        prompt = f"""{system_msg}

Based on these documents:
{doc_context}

User's question: {prompt}

Please provide a complete response that naturally concludes."""
    else:
        prompt = f"""{system_msg}

User's question: {prompt}

Please provide a complete response that naturally concludes."""

    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_file,
            "prompt": prompt,
            "stream": True
        }
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, json=payload, stream=True, timeout=120)
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line) #changed from pyjson.loads
                    if not json_line.get("done"):
                        # For /api/generate, the streaming field is usually 'response'
                        content = json_line.get("response", "")
                        # Clean the content before yielding
                        clean_content = clean_text(content)
                        yield clean_content
                except json.JSONDecodeError: #changed from pyjson.JSONDecodeError
                    continue
    except Exception as e:
        logger.error(f"Error in chat_handler_logic: {e}")
        yield f"Error: {e}"


async def process_final_text(websocket, text, model_name, tts_engine):
    """Helper function to process final recognized text and send response."""
    if not text:
        return
    logger.info(f"User said: {text}")
    await websocket.send_json({"type": "user_speech", "data": text})
    try:
        response_generator = chat_handler_logic(model_name, text)

        MAX_CHARACTERS = 300
        full_response = ""
        exceeded = False
        async for chunk in response_generator:
            full_response += chunk
            await websocket.send_json({"type": "bot_response_chunk", "data": chunk})
            if not exceeded and len(full_response) >= MAX_CHARACTERS:
                exceeded = True
            # After exceeding, wait for sentence-ending punctuation
            if exceeded and full_response[-1:] in ".!?":
                break

        if full_response:
            logger.info(f"Bot response: {full_response}")
            loop = asyncio.get_running_loop()
            output_filename = f"temp_{uuid.uuid4().hex}.wav"
            output_path = os.path.join(UPLOADS_DIR, output_filename)

            # For Piper TTS, we pass the engine type as a string
            await loop.run_in_executor(
                None, save_tts_audio, 'piper', full_response, output_path
            )

            with open(output_path, "rb") as f:
                await websocket.send_bytes(f.read())

            os.remove(output_path)
    except Exception as e:
        logger.error(f"Error during chat/TTS processing: {e}")
        await websocket.send_json({"type": "error", "data": str(e)})

@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    if not whisper_model:
        await websocket.close(code=1008, reason="Whisper model not loaded")
        return

    if not is_ollama_running():
        err_msg = "Ollama server is not running."
        logger.warning(err_msg)
        await websocket.send_json({"type": "error", "data": err_msg})
        await websocket.close(code=1008, reason=err_msg)
        return

    try:
        tts_engine = get_tts_engine()
    except Exception as e:
        logger.error(f"Failed to initialize TTS engine: {e}")
        await websocket.close(code=1008, reason="TTS engine failed")
        return

    try:
        # 1. Send initial ready message
        await websocket.send_json({"type": "status", "data": "Assistant is ready. How can I help?"})

        while True:
            # Buffer to collect all audio bytes for this utterance
            audio_buffer = b''
            while True:
                msg = await websocket.receive()
                if isinstance(msg, dict) and msg.get('type') == 'websocket.receive':
                    # Binary or text frame
                    if 'bytes' in msg and msg['bytes'] is not None:
                        audio_buffer += msg['bytes']
                    elif 'text' in msg and msg['text'] is not None:
                        try:
                            data = json.loads(msg['text'])
                            if data.get('type') == 'end_audio':
                                break  # End of utterance
                        except Exception:
                            continue
                elif isinstance(msg, bytes):
                    audio_buffer += msg
                elif isinstance(msg, str):
                    try:
                        data = json.loads(msg)
                        if data.get('type') == 'end_audio':
                            break
                    except Exception:
                        continue

            # 3. Save audio to a temporary file with a proper WAV header
            temp_audio_path = os.path.join(UPLOADS_DIR, f"temp_audio_{uuid.uuid4().hex}.wav")
            with wave.open(temp_audio_path, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000) # 16kHz
                wf.writeframes(audio_buffer)

            try:
                # Transcribe using FastWhisper
                segments, info = whisper_model.transcribe(temp_audio_path, beam_size=5, language="en")
                user_text = " ".join([segment.text for segment in segments]).strip()
                logger.info(f"Transcribed text: {user_text}")
                if not user_text:
                    user_text = '[No speech detected]'
                await websocket.send_json({"type": "user_speech", "data": user_text})
            finally:
                os.remove(temp_audio_path)

            # Always process the user_text, even if empty/placeholder


            # 4. Process the command
            await websocket.send_json({"type": "status", "data": "Thinking..."})
            response_text = handle_known_commands(user_text)

            # 5. If not a known command, ask Ollama
            if response_text is None:
                if user_text.lower() in ["exit", "goodbye"]:
                    response_text = "Goodbye!"
                else:
                    # Stream the response directly to the client
                    response_text = ""
                    async for chunk in chat_handler_logic("gemma3:1b", user_text):
                        response_text += chunk
                        await websocket.send_json({"type": "bot_response_chunk", "data": chunk})

            # The full response text is now assembled and ready for TTS, but we won't send it as a separate message,
            # as the content has already been streamed to the client.

            # 7. Generate TTS audio and send back to client
            output_path = os.path.join(UPLOADS_DIR, f"response_{uuid.uuid4().hex}.wav")
            try:
                # For Piper TTS, we pass the engine type as a string
                await asyncio.to_thread(save_tts_audio, 'piper', response_text, output_path)
                with open(output_path, "rb") as f:
                    await websocket.send_bytes(f.read())
            finally:
                if os.path.exists(output_path):
                    os.remove(output_path)

            if response_text == "Goodbye!":
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected from voice chat")
    except Exception as e:
        logger.error(f"An error occurred in the voice WebSocket: {e}")
    finally:
        try:
            await websocket.close(code=1000)
        except RuntimeError:
            pass # Socket already closed


@app.get("/api/voices")
async def list_voice_models():
    """
    List all available voice models in the C:/Users/ankit/piper directory.
    Returns a list of dictionaries with model information.
    """
    voices_dir = r"C:/Users/ankit/piper/voices"
    models = []
    try:
        # Get all .onnx files in the voices directory
        for file in os.listdir(voices_dir):
            if file.endswith(".onnx"):
                model_id = file.replace(".onnx", "")
                model_name = model_id.replace("-", " ").title()  # Convert filename to title case
                # Check if there's a corresponding JSON file
                json_file = f"en_en_US_{model_id}_{model_id}.onnx.json"
                json_path = os.path.join(voices_dir, json_file)
                model_info = {
                    "id": model_id,
                    "name": model_name,
                    "file": file,
                    "has_config": os.path.exists(json_path)
                }
                models.append(model_info)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing voice models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Static Files Mounting (The fix for CORS) ---
# This must be placed at the end of the file.
# It tells FastAPI to serve all files from the 'static' directory.
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/", StaticFiles(directory="static", html=True), name="static")