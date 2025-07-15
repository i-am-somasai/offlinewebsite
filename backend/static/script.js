import { marked } from "./marked/lib/marked.esm.js";

// Logging helper function
function logEvent(eventName, data = {}) {
    const timestamp = new Date().toISOString();
    const logData = {
        timestamp,
        event: eventName,
        ...data
    };
    console.log('[VOICE_DEBUG]', JSON.stringify(logData, null, 2));
}

// Toast Notification Function
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    if (!toast) return;
    
    // Set message and type
    toast.textContent = message;
    toast.className = 'toast';
    
    // Add type class (success, error, etc.)
    if (type) {
        toast.classList.add(type);
    }
    
    // Add show class to make it visible
    setTimeout(() => toast.classList.add('show'), 10);
    
    // Hide after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show');
        // Remove type class after animation completes
        setTimeout(() => toast.className = 'toast', 300);
    }, 3000);
}

// Offline Voice Recognition using MediaRecorder + Backend
// --- Voice Assistant --- //
const voiceModal = document.getElementById('voice-chat-modal');
const voiceStatus = document.getElementById('voice-chat-status');
const voiceMessages = document.getElementById('voice-chat-messages');
const closeVoiceBtn = document.getElementById('close-voice-chat');

let voiceSocket = null;
let audioContext = null;
let scriptProcessor = null;
let audioStream = null;

let isRecording = false;
let isAssistantSpeaking = false;
let isVoiceModalOpen = false;
let isSpaceHeld = false;
let spaceHoldTimer = null;
const SPACE_HOLD_DURATION = 1000; // 1 second hold to start recording

// Device configuration
let useGPU = true; // Default to GPU if available

// Check for GPU availability
async function checkGPUAvailability() {
    try {
        if (!navigator.gpu) {
            console.log('WebGPU not supported on this browser.');
            return false;
        }
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.log('No appropriate GPU adapter found.');
            return false;
        }
        console.log('GPU is available and will be used by default');
        return true;
    } catch (error) {
        console.log('Error checking for GPU:', error);
        return false;
    }
}

// Initialize GPU check on page load
document.addEventListener('DOMContentLoaded', async () => {
    const gpuAvailable = await checkGPUAvailability();
    const gpuToggle = document.getElementById('gpuToggle');
    
    if (!gpuAvailable) {
        useGPU = false;
        gpuToggle.checked = false;
        gpuToggle.disabled = true;
        console.log('GPU not available, falling back to CPU');
    }
    
    // Add event listener for GPU/CPU toggle
    gpuToggle.addEventListener('change', (e) => {
        useGPU = e.target.checked;
        showToast(`Switched to ${useGPU ? 'GPU' : 'CPU'} processing`, 'info');
        
        // Send the new processing mode to the backend
        if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
            voiceSocket.send(JSON.stringify({
                type: 'set_processing_mode',
                use_gpu: useGPU
            }));
        }
    });
});

async function openVoiceChat() {
    if (isVoiceModalOpen) return;
    voiceModal.style.display = 'flex';
    isVoiceModalOpen = true;
    isSpaceHeld = false;
    
    // Initialize audio context when opening the modal
    try {
        console.time('micInit');
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        
        // Time the microphone access
        const micStartTime = performance.now();
        try {
            audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: { 
                    sampleRate: 16000, 
                    channelCount: 1,
                    noiseSuppression: true,
                    echoCancellation: true
                } 
            });
            
            const micInitTime = performance.now() - micStartTime;
            console.log(`Microphone initialized in ${micInitTime.toFixed(0)}ms`);
            console.timeEnd('micInit');
            
            // Show success notification
            showToast(`Microphone ready in ${micInitTime.toFixed(0)}ms`, 'success');
            
            // Add visual feedback for microphone activation with timing info
            updateVoiceStatus(`<strong>Microphone ready in ${micInitTime.toFixed(0)}ms. Hold SPACE to talk</strong>`);
        } catch (error) {
            console.error('Error accessing microphone:', error);
            showToast('Could not access microphone. Please check permissions.', 'error');
            updateVoiceStatus('<strong>Error accessing microphone. Please check permissions.</strong>');
            closeVoiceChat();
            throw error; // Re-throw to be caught by the outer try-catch
        }
        
        document.addEventListener('keydown', handleKeyDown);
        document.addEventListener('keyup', handleKeyUp);
    } catch (err) {
        console.error('Error accessing microphone:', err);
        updateVoiceStatus('<strong>Error accessing microphone. Please check permissions.</strong>');
        closeVoiceChat();
    }
}

function closeVoiceChat() {
    if (!isVoiceModalOpen) return;
    stopRecording();
    if (voiceSocket) {
        voiceSocket.close();
        voiceSocket = null;
    }
    voiceModal.style.display = 'none';
    isVoiceModalOpen = false;
    isSpaceHeld = false;
    clearTimeout(spaceHoldTimer);
    spaceHoldTimer = null;
    document.removeEventListener('keydown', handleKeyDown);
    document.removeEventListener('keyup', handleKeyUp);
    
    // Clean up audio context when closing
    if (audioContext) {
        audioContext.close().catch(console.error);
        audioContext = null;
    }
    
    // Stop all audio tracks
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
}

function updateVoiceStatus(html, listening = false, holdProgress = 0) {
    voiceStatus.innerHTML = html;
    
    // Update progress indicator if provided
    const progressBar = document.getElementById('hold-progress-bar');
    if (progressBar) {
        progressBar.style.width = `${holdProgress}%`;
        progressBar.style.display = holdProgress > 0 ? 'block' : 'none';
    }
    
    // Show/hide hold indicator
    const holdIndicator = document.getElementById('hold-indicator');
    if (holdIndicator) {
        holdIndicator.style.display = isVoiceModalOpen && !isRecording ? 'block' : 'none';
    }
    
    // Update visual feedback based on state
    if (listening) {
        voiceStatus.classList.add('listening');
    } else {
        voiceStatus.classList.remove('listening');
    }
}

function addChatMessage(message, sender) {
    const messageElement = document.createElement('div');
    messageElement.classList.add('voice-chat-message', sender);
    // Use marked for bot, plain for user (can adjust for markdown if desired)
    messageElement.innerHTML = sender === 'user' ? message : marked.parse(message);
    // Always append, never remove previous
    voiceMessages.appendChild(messageElement);
    // Fade-in is handled by CSS animation on .voice-chat-message
    // Always scroll to bottom after adding
    setTimeout(() => {
        voiceMessages.scrollTop = voiceMessages.scrollHeight;
    }, 10);
}

// If ever resetting chat, clear explicitly with voiceMessages.innerHTML = '' elsewhere.


function handleKeyDown(e) {
    // Only handle space key when modal is open and not already recording/speaking
    if (e.code !== 'Space' || isRecording || isAssistantSpeaking || !isVoiceModalOpen) return;
    e.preventDefault();
    
    // If already holding, ignore
    if (isSpaceHeld) return;
    
    // Start hold timer
    const startTime = performance.now();
    isSpaceHeld = true;
    logEvent('HOLD_STARTED', { startTime });
    
    // Show notification that microphone is activating
    showToast('Microphone activating...', 'info');
    
    // Update progress during hold
    spaceHoldTimer = setInterval(() => {
        if (!isSpaceHeld) {
            clearInterval(spaceHoldTimer);
            return;
        }
        
        const elapsed = Date.now() - startTime;
        const progress = Math.min(100, (elapsed / SPACE_HOLD_DURATION) * 100);
        updateVoiceStatus(`<strong>Keep holding to speak (${Math.round(progress)}%)</strong>`, false, progress);
        
        // If held long enough, start recording
        if (elapsed >= SPACE_HOLD_DURATION) {
            clearInterval(spaceHoldTimer);
            logEvent('HOLD_COMPLETE', { holdDuration: performance.now() - startTime });
            startRecording();
        }
    }, 50);
}

async function startRecording() {
    if (isRecording) return;
    
    const recordingStartTime = performance.now();
    isRecording = true;
    updateVoiceStatus('Listening...', true, 100);
    logEvent('RECORDING_START', { recordingStartTime });
    
    try {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsStartTime = performance.now();
        voiceSocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/voice`);
        logEvent('WEBSOCKET_CREATED', { wsStartTime });

        voiceSocket.onopen = () => {
            // Send the current processing mode when connection opens
            voiceSocket.send(JSON.stringify({
                type: 'set_processing_mode',
                use_gpu: useGPU
            }));
            const wsOpenTime = performance.now();
            logEvent('WEBSOCKET_OPEN', { 
                timeToOpen: wsOpenTime - wsStartTime,
                timestamp: wsOpenTime
            });
            if (!audioStream || !audioContext) {
                throw new Error('Audio stream not available');
            }
            
            // Show notification only when the WebSocket is actually ready
            const readyTime = performance.now();
            logEvent('MICROPHONE_READY', { 
                timeToReady: readyTime - recordingStartTime,
                timestamp: readyTime
            });
            showToast('Microphone is active - Speak now', 'success');
            
            const source = audioContext.createMediaStreamSource(audioStream);
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

            scriptProcessor.onaudioprocess = (event) => {
                if (!isRecording) return;
                const inputData = event.inputBuffer.getChannelData(0);
                const int16Array = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    int16Array[i] = Math.max(-1, Math.min(1, inputData[i])) * 32767;
                }
                if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
                    voiceSocket.send(int16Array.buffer);
                }
            };
            
            source.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);
        };
        
        let botMessageElement = null;

        voiceSocket.onmessage = async (event) => {
            const messageTime = performance.now();
            logEvent('WEBSOCKET_MESSAGE', { 
                messageType: event.data instanceof Blob ? 'audio' : 'text',
                timestamp: messageTime
            });
            if (event.data instanceof Blob) {
                isAssistantSpeaking = true;
                updateVoiceStatus('Speaking...');
                const audio = new Audio(URL.createObjectURL(event.data));
                audio.play();
                audio.onended = () => {
                    isAssistantSpeaking = false;
                    updateVoiceStatus('<strong>Hold SPACE to talk3</strong>');
                };
            } else {
                try {
                    const message = JSON.parse(event.data);
                    console.log('WebSocket message received:', message);
                    
                    switch (message.type) {
                        case 'user_speech':
                            // Show user message
                            addChatMessage(message.data, 'user');
                            updateVoiceStatus('Thinking...');
                            // Create a new message element for bot's response
                            botMessageElement = document.createElement('div');
                            botMessageElement.classList.add('voice-chat-message', 'bot');
                            voiceMessages.appendChild(botMessageElement);
                            break;
                            
                        case 'bot_response':
                            // Handle complete bot response
                            if (botMessageElement) {
                                botMessageElement.innerHTML = marked.parse(message.data);
                                voiceMessages.scrollTop = voiceMessages.scrollHeight;
                            }
                            break;
                            
                        case 'bot_response_chunk':
                            // Handle streaming response chunks
                            if (botMessageElement) {
                                botMessageElement.innerHTML += message.data;
                                voiceMessages.scrollTop = voiceMessages.scrollHeight;
                            }
                            break;
                            
                        case 'error':
                            addChatMessage(`Error: ${message.data}`, 'bot');
                            console.error('WebSocket error:', message.data);
                            break;
                            
                        default:
                            console.log('Unknown message type:', message.type);
                    }
                } catch (error) {
                    console.error('Error processing WebSocket message:', error);
                    addChatMessage('Error processing response', 'bot');
                }
            }
        };


        voiceSocket.onclose = () => {
            stopRecording();
            if (!isAssistantSpeaking) {
                 updateVoiceStatus('Connection closed. <strong>Hold SPACE to talk4</strong>');
            }
        };

        voiceSocket.onerror = () => stopRecording();

    } catch (err) {
        updateVoiceStatus(`Error: ${err.message}`);
        isRecording = false;
    }
}

function handleKeyUp(e) {
    if (e.code !== 'Space' || !isVoiceModalOpen) return;
    e.preventDefault();
    
    // If recording, stop it
    if (isRecording) {
        stopRecording();
        showToast('Processing your speech...', 'info');
        updateVoiceStatus('Processing...');
        // Send an explicit end-of-audio signal to the backend
        if (voiceSocket && voiceSocket.readyState === WebSocket.OPEN) {
            voiceSocket.send(JSON.stringify({ type: 'end_audio' }));
        }
    } 
    // If still in hold phase, cancel it
    else if (isSpaceHeld) {
        isSpaceHeld = false;
        clearInterval(spaceHoldTimer);
        showToast('Microphone ready', 'info');
        updateVoiceStatus('<strong>Hold SPACE to talk</strong>', false, 0);
    }
}

function stopRecording() {
    if (!isRecording) return;
    
    const stopTime = performance.now();
    logEvent('RECORDING_STOP', { timestamp: stopTime });
    
    isRecording = false;
    isSpaceHeld = false;
    clearInterval(spaceHoldTimer);
    
    // Disconnect audio processing
    if (scriptProcessor) {
        try {
            scriptProcessor.disconnect();
        } catch (e) {
            console.warn('Error disconnecting script processor:', e);
        }
        scriptProcessor = null;
    }
    
    // Don't close audioContext or audioStream here as we want to keep them active
    // They will be cleaned up when closing the modal
    
    // Reset UI if modal is still open
    if (isVoiceModalOpen) {
        showToast('Microphone ready', 'info');
        updateVoiceStatus('<strong>Hold SPACE to talk again</strong>', false, 0);
    }
}

// Camera button logic
let cameraStream = null;
const cameraButton = document.getElementById('cameraButton');
const cameraModal = document.getElementById('camera-modal');
const cameraVideo = document.getElementById('camera-video');
const closeCameraModal = document.getElementById('close-camera-modal');
const videoCaptureCanvas = document.getElementById('video-capture-canvas');

// Vision Assistant Elements
const visionEntry = document.getElementById('vision-assistant-entry');
const enableVisionBtn = document.getElementById('enable-vision-assistant-btn');
const takePictureBtn = document.getElementById('take-picture-btn');
const visionResultDiv = document.getElementById('vision-assistant-result');
const visionAudio = document.getElementById('vision-assistant-audio');

// Vision Assistant Workflow
if (enableVisionBtn) {
    enableVisionBtn.addEventListener('click', async () => {
        visionEntry.style.display = 'none';
        cameraModal.style.display = 'flex';
        visionResultDiv.textContent = '';
        visionAudio.style.display = 'none';
        // Disable Take Picture button until video is ready
        if (takePictureBtn) takePictureBtn.disabled = true;
        // Start webcam
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
            cameraVideo.srcObject = cameraStream;
            cameraVideo.play();
            // Enable Take Picture when video is ready
            cameraVideo.onloadedmetadata = () => {
                if (takePictureBtn) takePictureBtn.disabled = false;
            };
        } catch (e) {
            visionResultDiv.textContent = 'Unable to access camera.';
            if (takePictureBtn) takePictureBtn.disabled = true;
        }
    });
}

if (closeCameraModal) {
    closeCameraModal.addEventListener('click', () => {
        cameraModal.style.display = 'none';
        visionEntry.style.display = 'flex';
        visionResultDiv.textContent = '';
        visionAudio.style.display = 'none';
        // Stop webcam
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
    });
}

if (takePictureBtn) {
    takePictureBtn.addEventListener('click', async () => {
        if (!cameraStream) {
            visionResultDiv.textContent = 'Camera is not active.';
            return;
        }
        // Check if video is ready
        if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
            visionResultDiv.textContent = 'Video not ready yet. Please wait.';
            return;
        }
        // Capture frame
        const ctx = videoCaptureCanvas.getContext('2d');
        videoCaptureCanvas.width = cameraVideo.videoWidth;
        videoCaptureCanvas.height = cameraVideo.videoHeight;
        ctx.drawImage(cameraVideo, 0, 0, cameraVideo.videoWidth, cameraVideo.videoHeight);
        const imageDataUrl = videoCaptureCanvas.toDataURL('image/jpeg');
        // Debug: Log the Data URL length and start
        console.log('Captured imageDataUrl:', imageDataUrl.substring(0, 100), '... length:', imageDataUrl.length);
        if (!imageDataUrl.startsWith('data:image/jpeg;base64,')) {
            visionResultDiv.textContent = 'Failed to capture image.';
            return;
        }
        // Show loading overlay
        showVideoAssistantOverlay('Analyzing image...', true);
        visionResultDiv.textContent = '';
        visionAudio.style.display = 'none';
        // Send to backend
        try {
            const response = await fetch('/analyze_image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: 'Describe the objects and elements in this image.', image: imageDataUrl })
            });
            hideVideoAssistantOverlay();
            if (response.ok) {
                const data = await response.json();
                visionResultDiv.textContent = data.response || 'No description available.';
                // Request TTS audio for the description with selected voice
                const ttsResp = await fetch('/tts', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        text: data.response,
                        voice: selectedVoice || undefined,
                        engine: useGoogleTTS ? 'gtts' : 'piper'
                    })
                });
                if (ttsResp.ok) {
                    const blob = await ttsResp.blob();
                    visionAudio.src = URL.createObjectURL(blob);
                    visionAudio.style.display = 'block';
                    visionAudio.play();
                }
            } else {
                visionResultDiv.textContent = 'Analysis failed.';
            }
        } catch (e) {
            hideVideoAssistantOverlay();
            visionResultDiv.textContent = 'Error analyzing image.';
        }
    });
}

// Overlay DOM elements for Vision Assistant
const videoAssistantOverlay = document.getElementById('video-assistant-overlay');
const videoAssistantMessage = document.getElementById('video-assistant-message');
const videoAssistantSpinner = document.getElementById('video-assistant-spinner');

// Helper: Show overlay message
function showVideoAssistantOverlay(message, showSpinner = false) {
    if (!videoAssistantMessage || !videoAssistantOverlay || !videoAssistantSpinner) {
        console.error('Vision overlay elements not found');
        return;
    }
    videoAssistantMessage.textContent = message;
    videoAssistantOverlay.style.display = 'flex';
    videoAssistantSpinner.style.display = showSpinner ? 'block' : 'none';
}
function hideVideoAssistantOverlay() {
    if (!videoAssistantMessage || !videoAssistantOverlay || !videoAssistantSpinner) return;
    videoAssistantOverlay.style.display = 'none';
    videoAssistantSpinner.style.display = 'none';
    videoAssistantMessage.textContent = '';
}

// Capture frame from video
function captureFrameFromVideo() {
    if (!cameraVideo || !videoCaptureCanvas) return null;
    const ctx = videoCaptureCanvas.getContext('2d');
    ctx.drawImage(cameraVideo, 0, 0, videoCaptureCanvas.width, videoCaptureCanvas.height);
    return videoCaptureCanvas.toDataURL('image/jpeg');
}

// Send image and prompt to backend
async function analyzeVideoFrame(prompt, imageDataUrl) {
    showVideoAssistantOverlay('Analyzing...', true);
    try {
        const response = await fetch('/analyze_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, image: imageDataUrl, voice: selectedVoice || undefined })
        });
        if (response.ok) {
            const data = await response.json();
            hideVideoAssistantOverlay();
            // Show response in chat and speak
            addMessage(data.response, false);
            speakText(data.response);
        } else {
            hideVideoAssistantOverlay();
            addMessage('Sorry, analysis failed.', false);
        }
    } catch (err) {
        hideVideoAssistantOverlay();
        addMessage('Error analyzing image: ' + err.message, false);
    }
}

// Orchestrate video assistant flow
/* async function handleVideoAssistantPrompt(prompt) {
    if (!cameraModal || cameraModal.style.display !== 'flex') return;
    isVideoAssistantActive = true;
    showVideoAssistantOverlay('Please hold the view still for 5 seconds...');
    await new Promise(res => setTimeout(res, 5000));
    const imageDataUrl = captureFrameFromVideo();
    if (imageDataUrl) {
        await analyzeVideoFrame(prompt, imageDataUrl);
    } else {
        hideVideoAssistantOverlay();
        addMessage('Could not capture image from video.', false);
    }
    isVideoAssistantActive = false;
} */

// No longer needed - Removed voice prompt listener for camera modal

// Initialize camera modal without voice prompt listener

function openCameraModal() {
    if (cameraModal) {
        cameraModal.style.display = 'flex';
        cameraModal.classList.add('active');
        // Hide instructions overlay (not needed for direct picture taking)
        const instructions = document.getElementById('video-assistant-instructions');
        if (instructions) instructions.style.display = 'none';
        // Hide other overlays
        hideVideoAssistantOverlay();
        // Start camera
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    cameraStream = stream;
                    cameraVideo.srcObject = stream;
                    cameraVideo.play();
                    setTimeout(() => {
                        // speakText("Please hold the camera still and ask a question about what you're pointing at.");
                        listenForVisualPrompt();
                    }, 1000);                    
                })
                .catch(err => {
                    // alert('Unable to access camera: ' + err.message);
                    closeCameraModalFunc();
                });
        } else {
            alert('Camera not supported on this browser.');
            closeCameraModalFunc();
        }
    }
}

function closeCameraModalFunc() {
    if (cameraModal) {
        cameraModal.style.display = 'none';
        cameraModal.classList.remove('active');
    }
    // Stop camera stream
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        cameraVideo.srcObject = null;
    }
}

if (cameraButton) {
    cameraButton.addEventListener('click', openCameraModal);
}
if (closeCameraModal) {
    closeCameraModal.addEventListener('click', closeCameraModalFunc);
}

// Also close camera modal if user clicks outside the modal content
if (cameraModal) {
    cameraModal.addEventListener('click', (e) => {
        if (e.target === cameraModal) {
            closeCameraModalFunc();
        }
    });
}

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const voiceButton = document.getElementById('voiceButton');
    if (voiceButton) {
        voiceButton.addEventListener('click', openVoiceChat);
    }
    if (closeVoiceBtn) {
        closeVoiceBtn.addEventListener('click', closeVoiceChat);
    }
});

// Stop recording when clicking outside the input area
document.addEventListener('click', (e) => {
    if (isRecording && !e.target.closest('.input-wrapper') && !e.target.closest('.voice-status')) {
        stopRecording();
    }
});

// Stop recording when pressing Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && isRecording) {
        stopRecording();
        updateVoiceStatus('Voice input cancelled');
        setTimeout(() => updateVoiceStatus('Click the mic to try again'), 2000);
    }
});

async function listenForVisualPrompt() {
    try {
        // Start listening
        await fetch('/api/start_listening', { method: 'POST' });

        // Wait 6 seconds, then stop and get text
        setTimeout(async () => {
            const response = await fetch('/api/stop_listening', { method: 'POST' });
            const result = await response.json();
            const userPrompt = result.text?.trim();

            if (userPrompt && (
                userPrompt.toLowerCase().includes("object") ||
                userPrompt.toLowerCase().includes("room") ||
                userPrompt.toLowerCase().includes("holding") ||
                userPrompt.toLowerCase().includes("describe")
            )) {
                // Speak feedback
                showVideoAssistantOverlay("Analyzing your view...", true);

                // Capture image
                const imageDataUrl = captureFrameFromVideo();
                if (!imageDataUrl) return;

                const response = await fetch('/analyze_image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: userPrompt, image: imageDataUrl, voice: selectedVoice || undefined })
                });
                const data = await response.json();
                hideVideoAssistantOverlay();

                // Speak result
                addMessage(data.response, false);
                speakText(data.response);
            } else {
                hideVideoAssistantOverlay();
                // speakText("I didn't catch a visual question. Please try again.");
            }
        }, 6000);
    } catch (e) {
        console.error("Error in voice visual prompt:", e);
    }
}

// Theme toggle functionality
const themeToggle = document.getElementById('themeToggle');
const moonIcon = themeToggle.querySelector('.moon-icon');

const refreshBtn = document.getElementById('refresh-btn');
if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
        window.location.reload(true);
    });
}

themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-theme');
    moonIcon.style.transform = document.body.classList.contains('dark-theme') ? 'rotate(360deg)' : 'rotate(0deg)';
});

// File upload handling
const fileUpload = document.getElementById('file-upload');
const fileList = document.getElementById('fileList');
const allowedTypes = ['.pdf', '.doc', '.docx', '.txt', '.png', '.jpg', '.jpeg', '.bmp', '.gif'];

// Available models list update
async function updateAvailableModelList() {
    try {
        const response = await fetch('/list_models');
        const data = await response.json();
        const modelsList = document.getElementById('availableModelList');
        if (!modelsList) {
            return;
        }
        modelsList.innerHTML = '';
        (data.models || []).forEach(model => {
            const li = document.createElement('li');
            li.textContent = model;
            modelsList.appendChild(li);
        });
    } catch (e) {
        console.error('[updateAvailableModelList] Error fetching models:', e);
    }
}


document.addEventListener('DOMContentLoaded', updateAvailableModelList);

fileUpload.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    
    for (const file of files) {
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowedTypes.includes(fileExt)) {
            alert(`File type ${fileExt} is not allowed. Please upload only PDF, DOC, DOCX, TXT, PNG, JPG, JPEG, BMP, or GIF files.`);
            continue;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                addFileToList(file.name, data.filename, data.filetype);
            } else {
                alert(`Failed to upload ${file.name}`);
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert(`Error uploading ${file.name}`);
        }
    }
});

function addFileToList(originalName, savedName, filetype) {
    const li = document.createElement('li');
    
    // Create view button with eye icon
    const viewButton = `
        <button class="view-file" data-filename="${savedName}" title="View File">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                <circle cx="12" cy="12" r="3"></circle>
            </svg>
        </button>`;

    const removeButton = `
        <button class="remove-file" data-filename="${savedName}" title="Remove File">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
        </button>`;

    if (filetype === 'image') {
        li.innerHTML = `
            <span><img src="/uploads/${savedName}" alt="${originalName}" style="max-width:32px;max-height:32px;vertical-align:middle;margin-right:8px;">${originalName}</span>
            <div class="file-actions">
                ${viewButton}
                ${removeButton}
            </div>
        `;
    } else {
        li.innerHTML = `
            <span>${originalName}</span>
            <div class="file-actions">
                ${viewButton}
                ${removeButton}
            </div>
        `;
    }
    fileList.appendChild(li);
}

// File handling (removal and preview)
fileList.addEventListener('click', async (e) => {
    // Handle file removal
    if (e.target.closest('.remove-file')) {
        const button = e.target.closest('.remove-file');
        const filename = button.dataset.filename;
        const listItem = button.closest('li');

        try {
            const response = await fetch(`/delete/${filename}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                listItem.remove();
            } else {
                alert('Failed to remove file');
            }
        } catch (error) {
            console.error('Remove error:', error);
            alert('Error removing file');
        }
    }
    
    // Handle file preview
    if (e.target.closest('.view-file')) {
        const button = e.target.closest('.view-file');
        const filename = button.dataset.filename;
        const modal = document.getElementById('file-preview-modal');
        const iframe = document.getElementById('file-preview-iframe');
        
        // Set the iframe source to the file URL
        iframe.src = `/uploads/${filename}`;
        modal.style.display = 'block';
        
        // Close button functionality
        const closeButton = modal.querySelector('.close-button');
        closeButton.onclick = () => {
            modal.style.display = 'none';
            iframe.src = '';
        };
        
        // Close when clicking outside the modal
        window.onclick = (event) => {
            if (event.target === modal) {
                modal.style.display = 'none';
                iframe.src = '';
            }
        };
    }
});

// Model selection handling
const modelSelect = document.getElementById('modelSelect');
const selectedModelDisplay = document.getElementById('selectedModel');

// Function to update the model dropdown
async function updateModelDropdown() {
    try {
        const response = await fetch('/list_models');
        if (response.ok) {
            const data = await response.json();
            modelSelect.innerHTML = '<option value="" disabled selected>Select a model</option>';
            if (data.models && data.models.length > 0) {
                data.models.forEach(model => {
                    modelSelect.innerHTML += `<option value="${model}">${model}</option>`;
                });
            } else {
                modelSelect.innerHTML = '<option value="" disabled selected>No models available</option>';
            }
        }
    } catch (error) {
        modelSelect.innerHTML = '<option value="" disabled selected>Error loading models</option>';
    }
}

// Call on page load
updateModelDropdown();

modelSelect.addEventListener('change', (e) => {
    const selectedModel = e.target.value;
    if (selectedModel) {
        selectedModelDisplay.textContent = selectedModel;
    } else {
        selectedModelDisplay.textContent = 'No model available';
    }
});

// Chat functionality
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.querySelector('.send-btn');
const muteToggle = document.getElementById('muteToggle');
let isMuted = false;
let currentAudio = null; // Track the current audio instance

// Function to stop all audio
function stopAllAudio() {
    if (currentAudio) {
        // Only clean up the audio if we're not just muting
        if (!isMuted) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
            currentAudio = null;
        } else {
            // If we're muted, just pause the audio but keep the reference
            currentAudio.pause();
        }
    }
    
    // Also stop any speech synthesis that might be running
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }
}

// Voice Selection
const voiceSelect = document.getElementById('voiceSelect');
let voices = [];
let selectedVoice = localStorage.getItem('selectedVoice') || '';

// Helper function to construct API URLs
function getApiUrl(path) {
    // Use the current window location to construct the URL
    return `${window.location.origin}${path}`;
}

// Load available voices from local API
async function loadVoices() {
    try {
        console.log('Fetching voices from /api/voices...');
        const response = await fetch('/api/voices');
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Failed to load voices:', response.status, errorText);
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        const data = await response.json();
        console.log('Received voices data:', data);
        
        // Handle the response format where voices are in data.models
        const voices = data.models || data.voices || [];
        console.log('Parsed voices:', voices);
        
        // If we have an array of voice objects, map them to the expected format
        if (Array.isArray(voices)) {
            return voices.map(voice => ({
                id: voice.voice_id || voice.id || voice.name,
                name: voice.name || voice.voice_id || `Voice ${voice.id}`
            }));
        }
        
        return [];
    } catch (error) {
        console.error('Error in loadVoices:', error);
        return [];
    }
}
async function populateVoiceDropdown() {
    if (!voiceSelect) {
        console.warn('Voice select element not found');
        return;
    }
    
    // Show loading state
    voiceSelect.innerHTML = '<option value="" disabled>Loading voices...</option>';
    voiceSelect.disabled = true;
    voiceSelect.classList.add('loading');
    
    try {
        console.log('Fetching voices...');
        const startTime = performance.now();
        
        // Load voices from API
        voices = await loadVoices();
        
        const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
        console.log(`Loaded ${voices.length} voices in ${loadTime}s`);
        
        // Clear existing options
        voiceSelect.innerHTML = '';
        
        if (voices.length === 0) {
            // No voices available
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No voices available';
            option.disabled = true;
            voiceSelect.appendChild(option);
            voiceSelect.disabled = true;
            
            // Show warning notification
            showNotification('No TTS voices found. Please check your setup.', 4000, 'warning');
            return;
        }
        
        // Add default option with a nice icon
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = '🎙️ Select a voice...';
        defaultOption.disabled = true;
        voiceSelect.appendChild(defaultOption);
        
        // Add voice options with better formatting
        voices.forEach((voice, index) => {
            const option = document.createElement('option');
            const voiceId = voice.voice_id || voice.id || `voice-${index}`;
            const voiceName = voice.name || `Voice ${index + 1}`;
            
            option.value = voiceId;
            
            // Format the display text with emoji based on voice properties
            let displayText = voiceName;
            
            // Add language info if available
            if (voice.language) {
                displayText += ` (${voice.language})`;
            }
            
            option.textContent = displayText;
            
            // Add data attributes for filtering/sorting
            option.dataset.gender = voice.gender || '';
            option.dataset.language = voice.language || '';
            
            // Select the previously selected voice if it exists
            if (voiceId === selectedVoice) {
                option.selected = true;
                console.log('Restored selected voice:', voiceName);
            }
            
            voiceSelect.appendChild(option);
        });
        
        // Enable the select if we have voices
        voiceSelect.disabled = false;
        
        // If we have a selected voice from localStorage, make sure it's selected
        if (selectedVoice) {
            const voiceExists = Array.from(voiceSelect.options).some(opt => opt.value === selectedVoice);
            if (!voiceExists) {
                // If the saved voice doesn't exist in the new list, clear the selection
                console.warn('Previously selected voice not found, clearing selection');
                localStorage.removeItem('selectedVoice');
                selectedVoice = '';
                voiceSelect.selectedIndex = 0; // Select the default option
            }
        } else if (voices.length > 0) {
            // Auto-select the first voice if none is selected
            voiceSelect.selectedIndex = 1; // Skip the default option
            selectedVoice = voiceSelect.value;
            localStorage.setItem('selectedVoice', selectedVoice);
            console.log('Auto-selected first voice:', voiceSelect.options[voiceSelect.selectedIndex].text);
        }
        
        console.log(`Voice dropdown populated with ${voices.length} voices`);
        
        // Trigger a change event to ensure UI updates
        voiceSelect.dispatchEvent(new Event('change'));
        
    } catch (error) {
        console.error('Error populating voice dropdown:', error);
        
        // Show error state with retry button
        voiceSelect.innerHTML = `
            <option value="" disabled>
                ❌ Error loading voices
            </option>
            <option value="retry">
                🔄 Click to retry
            </option>
        `;
        voiceSelect.disabled = false;
        
        // Add retry handler
        voiceSelect.onchange = function() {
            if (this.value === 'retry') {
                this.onchange = null; // Remove this handler
                populateVoiceDropdown(); // Retry loading
            }
        };
        
        // Show error notification
        showNotification('Failed to load voices. Click the dropdown to retry.', 5000, 'error');
    } finally {
        // Remove loading state
        voiceSelect.classList.remove('loading');
    }
}

// Handle voice selection change with feedback
function handleVoiceSelect(event) {
    const newVoice = event.target.value;
    
    // Don't do anything if the selection didn't change
    if (newVoice === selectedVoice) return;
    
    selectedVoice = newVoice;
    localStorage.setItem('selectedVoice', selectedVoice);
    
    // Provide feedback when voice is selected
    if (selectedVoice) {
        const selectedOption = event.target.options[event.target.selectedIndex];
        const voiceName = selectedOption.text.split(' (')[0]; // Remove language code for display
        console.log('Selected voice:', voiceName);
        
        // Show a subtle notification with the selected voice
        showNotification(`Voice set to: ${voiceName}`, 2000, 'success');
        
        // Test the voice with a short sample if not muted
        if (!isMuted) {
            // Use a short, neutral test phrase
            const testPhrases = [
                'Hello, this is your selected voice.',
                'Voice test, one two three.',
                'This is how I will sound.'
            ];
            const testPhrase = testPhrases[Math.floor(Math.random() * testPhrases.length)];
            
            // Small delay to let the notification be visible
            setTimeout(() => {
                speakText(testPhrase).catch(error => {
                    console.error('Voice test failed:', error);
                });
            }, 300);
        }
    }
}

// Initialize voice selection
console.log('Initializing voice selection...');
if (voiceSelect) {
    console.log('voiceSelect element found, adding event listeners');
    voiceSelect.addEventListener('change', handleVoiceSelect);
    
    // Load voices when the page loads
    document.addEventListener('DOMContentLoaded', () => {
        console.log('DOMContentLoaded event fired');
        populateVoiceDropdown().catch(err => {
            console.error('Error in populateVoiceDropdown:', err);
        });
    });
    
    // Also try loading voices immediately in case DOM is already loaded
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        console.log('Document already loaded, populating voices now');
        populateVoiceDropdown().catch(console.error);
    }
} else {
    console.error('voiceSelect element not found in the document');
}

// Initialize voice selection
function initVoiceSelection() {
    const voiceSelect = document.getElementById('voiceSelect');
    if (!voiceSelect) return;

    // Load saved voice if exists
    const savedVoice = localStorage.getItem('selectedVoice');
    
    // Function to populate voices
    function populateVoices() {
        voices = window.speechSynthesis.getVoices();
        
        // Filter and sort voices
        const englishVoices = voices
            .filter(voice => voice.lang.startsWith('en-'))
            .sort((a, b) => a.name.localeCompare(b.name));
        
        // Clear existing options
        voiceSelect.innerHTML = '';
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.textContent = 'Select a voice...';
        defaultOption.value = '';
        voiceSelect.appendChild(defaultOption);
        
        // Add voice options
        englishVoices.forEach((voice, index) => {
            const option = document.createElement('option');
            const voiceName = `${voice.name} (${voice.lang})`;
            option.textContent = voiceName;
            option.value = index;
            option.dataset.voiceName = voiceName;
            voiceSelect.appendChild(option);
            
            // Select saved voice if it matches
            if (savedVoice === voiceName) {
                option.selected = true;
                selectedVoice = voice;
            }
        });
        
        // If no saved voice, select first one by default
        if (!savedVoice && englishVoices.length > 0) {
            voiceSelect.selectedIndex = 1; // Skip the default option
            selectedVoice = englishVoices[0];
            localStorage.setItem('selectedVoice', englishVoices[0].name + ' (' + englishVoices[0].lang + ')');
        }
    }
    
    // Handle voice selection change
    voiceSelect.addEventListener('change', (e) => {
        const selectedIndex = parseInt(e.target.value);
        if (isNaN(selectedIndex)) {
            selectedVoice = null;
            return;
        }
        
        // Get all English voices
        const englishVoices = window.speechSynthesis.getVoices()
            .filter(voice => voice.lang.startsWith('en-'));
            
        selectedVoice = englishVoices[selectedIndex];
        localStorage.setItem('selectedVoice', selectedVoice.name + ' (' + selectedVoice.lang + ')');
    });
    
    // Initial population
    if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = populateVoices;
    }
    
    // Try to populate immediately in case voices are already loaded
    if (window.speechSynthesis.getVoices().length > 0) {
        populateVoices();
    }
}

// TTS Engine Toggle with immediate feedback
const ttsEngineToggle = document.getElementById('ttsEngineToggle');
let useGoogleTTS = localStorage.getItem('useGoogleTTS') === 'true';

// Listen Duration Toggle
const listenDurationToggle = document.getElementById('listenDurationToggle');
let listenDuration = parseInt(localStorage.getItem('listenDuration') || '3000'); // Default to 3 seconds

// Initialize listen duration toggle state
function updateListenDurationToggleState() {
    listenDurationToggle.checked = listenDuration === 5000; // Check if 5 seconds is selected
}

// Set initial state
updateListenDurationToggleState();

// Handle toggle changes
listenDurationToggle.addEventListener('change', (e) => {
    listenDuration = e.target.checked ? 5000 : 3000; // 5s or 3s
    localStorage.setItem('listenDuration', listenDuration.toString());
    
    // Provide feedback
    const status = document.createElement('div');
    status.className = 'status-message';
    status.textContent = `Listening duration set to ${listenDuration/1000} seconds`;
    status.style.position = 'fixed';
    status.style.bottom = '20px';
    status.style.left = '50%';
    status.style.transform = 'translateX(-50%)';
    status.style.background = 'rgba(0,0,0,0.8)';
    status.style.color = 'white';
    status.style.padding = '10px 20px';
    status.style.borderRadius = '20px';
    status.style.zIndex = '1000';
    status.style.transition = 'opacity 0.3s';
    document.body.appendChild(status);
    
    // Remove status message after 2 seconds
    setTimeout(() => {
        status.style.opacity = '0';
        setTimeout(() => status.remove(), 300);
    }, 2000);
});

// Initialize the toggle state
function updateTTSToggleState() {
    ttsEngineToggle.checked = useGoogleTTS;
    // Visual feedback is now handled by CSS
}

// Set initial state
updateTTSToggleState();

// Handle toggle changes
ttsEngineToggle.addEventListener('change', async (e) => {
    useGoogleTTS = e.target.checked;
    localStorage.setItem('useGoogleTTS', useGoogleTTS);
    
    // Update UI immediately
    updateTTSToggleState();
    
    // Show a subtle notification
    const status = useGoogleTTS ? 'Google TTS' : 'Local TTS';
    showNotification(`TTS: ${status}`, 1500);
    
    // Test the TTS with a short message
    try {
        await speakText('TTS engine changed');
    } catch (error) {
        console.error('TTS test failed:', error);
    }
});

// Initialize mute toggle
muteToggle.addEventListener('click', () => {
    isMuted = !isMuted;
    updateMuteUI();
    
    // Save the mute state to localStorage
    localStorage.setItem('isMuted', isMuted);
    
    // If we're muting while audio is playing, pause it but keep the reference
    if (isMuted && currentAudio) {
        currentAudio.pause();
    } 
    // If we're unmuting and there's a current audio, resume playback
    else if (!isMuted && currentAudio) {
        currentAudio.play().catch(e => console.error('Error resuming audio:', e));
    }
});

// Function to update mute button UI
function updateMuteUI() {
    muteToggle.classList.toggle('muted', isMuted);
    const icon = muteToggle.querySelector('.speaker-icon');
    if (!icon) return;
    
    // Update the icon based on mute state
    if (isMuted) {
        // Change to muted icon
        icon.innerHTML = `
            <path d="M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51A8.796 8.796 0 0021 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 4L2.99 5.27 7.29 9.6 3 9.6v4.8h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06a8.99 8.99 0 003.69-1.81L19.73 21l1.27-1.27L4.27 4zM12 4L9.91 6.09 12 8.18V4z"></path>
            <path d="M0 0h24v24H0z" fill="none"></path>
        `;
    } else {
        // Change to speaker icon
        icon.innerHTML = `
            <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0014 7.97v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"></path>
            <path d="M0 0h24v24H0z" fill="none"></path>
        `;
    }
}

// Load states from localStorage and initialize UI
const savedMuteState = localStorage.getItem('isMuted');
if (savedMuteState !== null) {
    isMuted = savedMuteState === 'true';
    updateMuteUI();
}

// Initialize the mute button state on page load
document.addEventListener('DOMContentLoaded', () => {
    updateMuteUI();
});

function addMessage(content, isUser = false, stats = null) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', isUser ? 'user-message' : 'model-message');

    // Remove emojis from content before displaying
    const contentWithoutEmojis = content.replace(/[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '');
    const renderedContent = marked.parse(contentWithoutEmojis);

    if (isUser) {
        messageDiv.innerHTML = `
            <div class="message-content user">
                ${renderedContent}
            </div>
        `;
    } else {
        const modelName = stats ? stats.model_name || 'Unknown Model' : 'Unknown Model';
        const timeTaken = stats ? (stats.time_taken || 0).toFixed(2) : 'N/A';
        const tokensPerSecond = stats ? (stats.tokens_per_second || 0).toFixed(2) : 'N/A';
        const totalTokens = stats ? stats.total_tokens || 'N/A' : 'N/A';

        messageDiv.innerHTML = `
            <div class="message-header">
                <strong>${modelName}</strong>
            </div>
            <div class="message-content model">
                ${renderedContent}
            </div>
            <div class="message-footer">
                <span>Time: ${timeTaken}s</span>
                <span>Tokens: ${totalTokens}</span>
                <span>Speed: ${tokensPerSecond} tok/s</span>
            </div>
        `;
        
        // Auto-read the message if not muted
        if (!isMuted) {
            speakText(contentWithoutEmojis);
        }
    }

    const images = messageDiv.querySelectorAll('img');
    images.forEach(img => {
        if (img.src.startsWith('data:image')) {
            img.addEventListener('click', () => openImageModal(img.src));
        }
    });
    
    chatMessages.appendChild(messageDiv);
    // Always scroll to the bottom after adding a message
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 10);
}

async function speakText(text) {
    // Don't generate speech if muted or no text provided
    if (isMuted || !text || typeof text !== 'string' || text.trim() === '') {
        console.log('TTS skipped: muted or no text provided');
        return;
    }
    
    console.log('Speaking text with voice:', selectedVoice);
    
    try {
        // Stop any currently playing audio
        stopAllAudio();

        // Call backend TTS endpoint with the selected engine and voice
        const ttsUrl = getApiUrl('/tts');
        const requestBody = { 
            text: text,
            engine: useGoogleTTS ? 'gtts' : 'piper'
        };
        
        // Only add voice parameter if a voice is selected
        if (selectedVoice) {
            requestBody.voice = selectedVoice;
        }
        
        console.log('Calling TTS endpoint:', ttsUrl);
        console.log('Request payload:', { 
            ...requestBody,
            text: text.substring(0, 100) + (text.length > 100 ? '...' : '')
        });
        
        const response = await fetch(ttsUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('TTS API error:', response.status, errorText);
            throw new Error(`TTS failed: ${response.status} ${errorText}`);
        }

        // Get the audio data
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Create the audio element
        const audio = new Audio(audioUrl);
        currentAudio = audio; // Store reference to current audio
        
        // Clean up when audio finishes or errors
        audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
            if (currentAudio === audio) {
                currentAudio = null;
            }
        };
        
        audio.onerror = (error) => {
            console.error('Error playing audio:', error);
            URL.revokeObjectURL(audioUrl);
            if (currentAudio === audio) {
                currentAudio = null;
            }
        };
        
        // Play the audio if not muted
        if (!isMuted) {
            await audio.play().catch(error => {
                console.error('Error playing audio:', error);
                if (currentAudio === audio) {
                    currentAudio = null;
                }
            });
        }
        
    } catch (error) {
        console.error('Error with backend TTS, falling back to browser TTS:', error);
        // Fallback to browser's speech synthesis if available
        if ('speechSynthesis' in window) {
            const voices = window.speechSynthesis.getVoices();
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Log available browser voices for debugging
            console.log('=== Browser TTS Voices ===');
            voices.forEach((voice, i) => {
                console.log(`${i}. ${voice.name} (${voice.lang})`);
            });
            
            // Use the selected voice if available, otherwise fall back to default
            if (selectedVoice) {
                // Find the voice by ID if it's a string, or use it directly if it's a SpeechSynthesisVoice
                const voiceToUse = typeof selectedVoice === 'string' 
                    ? voices.find(v => v.voiceURI === selectedVoice || v.name === selectedVoice)
                    : selectedVoice;
                
                if (voiceToUse) {
                    utterance.voice = voiceToUse;
                    console.log(`Using selected voice: ${voiceToUse.name} (${voiceToUse.lang})`);
                }
            } else {
                // Try to find an English voice as fallback
                const englishVoices = voices.filter(v => v.lang.startsWith('en'));
                if (englishVoices.length > 0) {
                    utterance.voice = englishVoices[0];
                    console.log(`Using fallback English voice: ${englishVoices[0].name}`);
                } else if (voices.length > 0) {
                    utterance.voice = voices[0];
                    console.log(`Using first available voice: ${voices[0].name}`);
                }
            }
            
            speechSynthesis.speak(utterance);
            console.log('Using browser fallback TTS');
        } else {
            console.error('No TTS available in browser');
            alert('Error with text-to-speech. Please check console for details.');
        }
    }
}

// Add image modal functionality
function openImageModal(src) {
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.innerHTML = `
        <div class="image-modal-content">
            <span class="image-modal-close">&times;</span>
            <img src="${src}" alt="Full size image">
        </div>
    `;
    document.body.appendChild(modal);
    
    modal.querySelector('.image-modal-close').onclick = () => {
        modal.remove();
    };
    
    modal.onclick = (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    };
}

const startOllamaBtn = document.getElementById('startOllamaBtn');
let ollamaProcess = null;

// Function to check Ollama server status
async function checkOllamaStatus() {
    try {
        const response = await fetch('/ollama_status');
        const data = await response.json();
        return data.status === 'running';
    } catch (error) {
        return false;
    }
}

// Function to update button state
function updateOllamaButton(isRunning) {
    if (isRunning) {
        startOllamaBtn.classList.add('running');
        startOllamaBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                <path d="M6 6h12v12H6z"/>
            </svg>
            Stop Ollama
        `;
    } else {
        startOllamaBtn.classList.remove('running');
        startOllamaBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                <path d="M8 5v14l11-7z"/>
            </svg>
            Start Ollama
        `;
    }
}

// Check status periodically
async function startStatusCheck() {
    const isRunning = await checkOllamaStatus();
    updateOllamaButton(isRunning);
    ollamaProcess = isRunning;
}

// Check status every 5 seconds
setInterval(startStatusCheck, 5000);

// Initial status check
startStatusCheck();

startOllamaBtn.addEventListener('click', async () => {
    if (ollamaProcess) {
        // If Ollama is running, stop it
        try {
            const response = await fetch('/stop_ollama', { method: 'POST' });
            const data = await response.json();
            if (data.status === 'success') {
                ollamaProcess = false;
                updateOllamaButton(false);
                showSuccess('Ollama server stopped successfully');
            } else {
                showError(data.message || 'Failed to stop Ollama server');
            }
        } catch (error) {
            showError('Failed to stop Ollama server');
        }
    } else {
        // Start Ollama
        try {
            startOllamaBtn.disabled = true;
            startOllamaBtn.innerHTML = `
                <svg class="spinner" viewBox="0 0 50 50">
                    <circle class="path" cx="25" cy="25" r="20" fill="none" stroke-width="5"></circle>
                </svg>
                Starting...
            `;
            
            const response = await fetch('/start_ollama', { method: 'POST' });
            const data = await response.json();
            
            if (data.status === 'success') {
                ollamaProcess = true;
                updateOllamaButton(true);
                showSuccess('Ollama server started successfully');
            } else {
                showError(data.message || 'Failed to start Ollama server');
                updateOllamaButton(false);
            }
        } catch (error) {
            showError('Failed to start Ollama server');
            updateOllamaButton(false);
        } finally {
            startOllamaBtn.disabled = false;
        }
    }
});

// Audio Recording and Transcription
const micButton = document.getElementById('micButton');
const recordingStatus = document.getElementById('recordingStatus');

if (micButton) {
    micButton.addEventListener('click', async () => {
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            const audioChunks = [];

            // Show recording status
            micButton.classList.add('recording');
            recordingStatus.style.display = 'flex';
            userInput.placeholder = 'Listening...';

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                try {
                    // Hide recording status
                    micButton.classList.remove('recording');
                    recordingStatus.style.display = 'none';
                    userInput.placeholder = 'Ask your queries here...';

                    // Create audio blob
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const formData = new FormData();
                    formData.append('file', audioBlob, 'recording.wav');

                    // Show processing state
                    userInput.placeholder = 'Processing...';
                    userInput.disabled = true;

                    // Step 1: Transcribe audio
                    const response = await fetch('/transcribe_audio', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error('Transcription failed');
                    }

                    const { text } = await response.json();
                    
                    if (text && text.trim()) {
                        // Set transcribed text to input
                        userInput.value = text;
                        
                        // Auto-send the message
                        handleSendMessage();
                    }
                } catch (error) {
                    console.error('Transcription error:', error);
                    userInput.placeholder = 'Error transcribing audio. Try again.';
                } finally {
                    userInput.disabled = false;
                    userInput.placeholder = 'Ask your queries here...';
                    
                    // Stop all tracks in the stream
                    stream.getTracks().forEach(track => track.stop());
                }
            };

            // Use the selected listening duration
            const silenceDuration = listenDuration;

            // Start recording
            mediaRecorder.start();
            console.log(`Recording started with ${silenceDuration/1000}s silence detection`);
            
            let silenceTimer;
            let isStopped = false;
            
            const stopRecordingWithDelay = () => {
                if (isStopped) return;
                isStopped = true;
                
                if (mediaRecorder?.state === 'recording') {
                    console.log(`Stopping recording after ${silenceDuration/1000}s of silence`);
                    mediaRecorder.stop();
                }
            };
            
            const resetSilenceTimer = () => {
                if (silenceTimer) clearTimeout(silenceTimer);
                silenceTimer = setTimeout(stopRecordingWithDelay, silenceDuration);
            };

            // Set up audio context for silence detection
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const analyser = audioContext.createAnalyser();
            const microphone = audioContext.createMediaStreamSource(stream);
            microphone.connect(analyser);
            analyser.fftSize = 256;
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);

            const checkVolume = () => {
                if (isStopped) return;
                
                analyser.getByteFrequencyData(dataArray);
                const volume = Math.max(...dataArray) / 255;
                
                if (volume > 0.1) { // Threshold for sound detection
                    resetSilenceTimer();
                }
                
                if (mediaRecorder.state === 'recording') {
                    requestAnimationFrame(checkVolume);
                }
            };
            
            // Start the timers
            resetSilenceTimer();
            checkVolume();
            
            // Stop recording after the selected duration
            setTimeout(() => {
                if (mediaRecorder?.state === 'recording') {
                    stopRecordingWithDelay();
                }
            }, listenDuration);

        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Could not access microphone. Please ensure you have granted microphone permissions.');
            micButton.classList.remove('recording');
            recordingStatus.style.display = 'none';
            userInput.placeholder = 'Ask your queries here...';
        }
    });
}

// Chat logic should use the selected model
async function handleSendMessage() {
    const message = userInput.value.trim();
    const selectedModel = modelSelect.value;
    if (!message) return;
    if (!selectedModel) {
        showError('Please select a model first');
        return;
    }
    if (!ollamaProcess) {
        showError('Please start the Ollama server first');
        return;
    }

    // Add user message
    addMessage(message, true);
    userInput.value = '';

    // Add loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message bot-message';
    loadingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                model: selectedModel,
                device: document.getElementById('device').value,
                threads: parseInt(document.getElementById('threads').value, 10),
                token_count: parseInt(document.getElementById('token-count').value, 10),
                gpu_cores: parseInt(document.getElementById('gpu-cores').value, 10),
                cpu_cores: parseInt(document.getElementById('cpu-cores').value, 10),
                voice: selectedVoice || undefined,
                tts_engine: useGoogleTTS ? 'gtts' : 'piper'
            })
        });

        if (response.ok) {
            const data = await response.json();
            loadingDiv.remove();
            
            // Pass the entire response data to addMessage
            addMessage(data.response, false, {
                time_taken: data.time_taken,
                tokens_per_second: data.tokens_per_second,
                total_tokens: data.total_tokens,
                characters: data.characters,
                model_name: data.model_name
            });
        } else {
            throw new Error('Failed to get response');
        }
    } catch (error) {
        console.error('Chat error:', error);
        loadingDiv.remove();
        addMessage('Sorry, I encountered an error. Please try again.');
    }
}

sendButton.addEventListener('click', handleSendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleSendMessage();
    }
});

// Settings panel functionality
const settingsBtn = document.getElementById('settingsBtn');
const settingsPanel = document.getElementById('settingsPanel');
const settingsClose = document.getElementById('settingsClose');

settingsBtn.addEventListener('click', () => {
    settingsPanel.classList.add('active');
});

settingsClose.addEventListener('click', () => {
    settingsPanel.classList.remove('active');
});

// Settings sliders
const tokenCount = document.getElementById('token-count');
const tokenCountValue = document.getElementById('token-count-value');
const gpuCores = document.getElementById('gpu-cores');
const gpuCoresValue = document.getElementById('gpu-cores-value');
const cpuCores = document.getElementById('cpu-cores');
const cpuCoresValue = document.getElementById('cpu-cores-value');

tokenCount.addEventListener('input', () => {
    tokenCountValue.textContent = tokenCount.value;
});

gpuCores.addEventListener('input', () => {
    gpuCoresValue.textContent = gpuCores.value;
});

cpuCores.addEventListener('input', () => {
    cpuCoresValue.textContent = cpuCores.value;
});

// Unify file upload triggers
const attachmentBtn = document.getElementById('attachmentBtn');
attachmentBtn.addEventListener('click', () => fileUpload.click());
const uploadButton = document.querySelector('.upload-button');
uploadButton.addEventListener('click', () => fileUpload.click());

// Display available models in sidebar

document.addEventListener('DOMContentLoaded', () => {
    const availableModelList = document.getElementById('availableModelList');
    async function updateAvailableModelList() {
        if (!availableModelList) return;
        try {
            const response = await fetch('/list_models');
            if (response.ok) {
                const data = await response.json();
                availableModelList.innerHTML = '';
                if (data.models && data.models.length > 0) {
                    data.models.forEach(model => {
                        const li = document.createElement('li');
                        li.textContent = model;
                        availableModelList.appendChild(li);
                    });
                } else {
                    availableModelList.innerHTML = '<li>No models available</li>';
                }
            }
        } catch (error) {
            availableModelList.innerHTML = '<li>Error loading models</li>';
        }
    }
    updateAvailableModelList();
    setInterval(updateAvailableModelList, 10000);
});

// Status dot for Ollama and settings button state
const ollamaStatusDot = document.getElementById('ollamaStatusDot');

async function updateOllamaStatusDot() {
    const isRunning = await checkOllamaStatus();
    const settingsBtn = document.getElementById('settingsBtn');
    
    if (isRunning) {
        ollamaStatusDot.classList.add('online');
        ollamaStatusDot.classList.remove('offline');
        if (settingsBtn) {
            settingsBtn.querySelector('svg').style.fill = 'currentColor';
            settingsBtn.title = 'Settings';
        }
    } else {
        ollamaStatusDot.classList.remove('online');
        ollamaStatusDot.classList.add('offline');
        if (settingsBtn) {
            settingsBtn.querySelector('svg').style.fill = '#6b7280'; // gray-500
            settingsBtn.title = 'Settings (Offline)';
        }
    }
}
// Initialize voice selection when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initVoiceSelection();
    updateOllamaStatusDot();
    populateVoiceDropdown();
    
    // Initialize listen duration from localStorage if available
    const savedDuration = localStorage.getItem('listenDuration');
    if (savedDuration) {
        listenDuration = parseInt(savedDuration);
        updateListenDurationToggleState();
    }
}); 