import os
import zipfile
import subprocess
import gradio as gr
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice
import torchaudio
import soundfile as sf
import librosa

# --- Path Configurations ---
ZIP_FILE = "Trump-LipSync.zip"
EXTRACTED_FOLDER = "Trump-LipSync"
TRUMP_VIDEO_PATH = os.path.join(EXTRACTED_FOLDER, "videos", "trump.mp4")
CHECKPOINT_PATH = os.path.join(EXTRACTED_FOLDER, "Wav2Lip", "checkpoints", "wav2lip_gan.pth")
VOICE_DIR = "trump_voices"
OUTPUT_WAV = "generated_speech.wav"
OUTPUT_VIDEO = "lip_synced_output.mp4"

# --- Helper Functions ---
def get_audio_duration(wav_path):
    """Get duration of audio file in seconds"""
    audio, sr = sf.read(wav_path)
    return len(audio) / sr

def get_video_duration(video_path):
    """Get duration of video file using FFprobe"""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout)

def validate_voice_samples():
    """Check if voice samples exist and are valid"""
    if not os.path.exists(VOICE_DIR):
        return False, f"Directory '{VOICE_DIR}' not found"
    
    samples = [f for f in os.listdir(VOICE_DIR) if f.startswith("trump_sample_") and f.endswith(".wav")]
    if len(samples) < 3:
        return False, f"Need at least 3 voice samples (found {len(samples)})"
    
    # Verify first sample's properties
    try:
        audio, sr = sf.read(os.path.join(VOICE_DIR, samples[0]))
        if sr != 22050:
            return False, f"Sample rate should be 22050Hz (found {sr}Hz)"
        if len(audio) != 22050 * 10:  # 10 seconds
            return False, "Samples should be 10 seconds long"
    except Exception as e:
        return False, f"Error checking samples: {str(e)}"
    
    return True, "Voice samples validated"

# --- Core Functions ---
def extract_zip():
    """Extract Wav2Lip assets from zip"""
    try:
        if not os.path.exists(ZIP_FILE):
            return False, f"{ZIP_FILE} not found"
        if os.path.exists(EXTRACTED_FOLDER):
            return True, f"{EXTRACTED_FOLDER} already exists"
        
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_FOLDER)
        return True, "Assets extracted successfully"
    except Exception as e:
        return False, f"Extraction failed: {str(e)}"

def generate_speech(text):
    """Generate Trump-like speech from text"""
    try:
        # Initialize TTS with performance optimizations
        tts = TextToSpeech()  # No arguments needed in newer versions
        
        # Load voice samples
        voice_samples, conditioning_latents = load_voice(
            VOICE_DIR,
            extra_voice_dirs=[os.getcwd()],
        )
        
        # Generate with quality fallback
        try:
            gen = tts.tts_with_preset(
                text,
                voice_samples=voice_samples,
                conditioning_latents=conditioning_latents,
                preset="high_quality",
                num_autoregressive_samples=2,
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("Falling back to 'fast' preset due to GPU memory")
                gen = tts.tts_with_preset(
                    text,
                    voice_samples=voice_samples,
                    conditioning_latents=conditioning_latents,
                    preset="fast"
                )
            else:
                raise
        
        # Save in Wav2Lip-compatible format
        torchaudio.save(
            OUTPUT_WAV,
            gen.squeeze(0).cpu(),
            22050,
            format="wav",
            bits_per_sample=16,
        )
        return True, "Speech generated"
    except Exception as e:
        return False, f"TTS failed: {str(e)}"

def run_wav2lip():
    """Run Wav2Lip synchronization"""
    try:
        command = [
            "python",
            os.path.join(EXTRACTED_FOLDER, "Wav2Lip", "inference.py"),
            "--checkpoint_path", CHECKPOINT_PATH,
            "--face", TRUMP_VIDEO_PATH,
            "--audio", OUTPUT_WAV,
            "--outfile", OUTPUT_VIDEO,
            "--fps", "25",  # Match Trump video's FPS
            "--pads", "0", "10", "0", "0",  # Padding adjustments
        ]
        subprocess.run(command, check=True)
        return True, "Lip-sync complete"
    except subprocess.CalledProcessError as e:
        return False, f"Wav2Lip error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

# --- Main Processing ---
def process_lipsync(text):
    """Full pipeline from text to lip-synced video"""
    # Validate prerequisites
    valid, msg = validate_voice_samples()
    if not valid:
        return msg, None
    
    if not os.path.exists(TRUMP_VIDEO_PATH):
        return "Trump video not found", None
    
    # Generate speech
    success, msg = generate_speech(text)
    if not success:
        return msg, None
    
    # Run Wav2Lip
    success, msg = run_wav2lip()
    if not success:
        return msg, None
    
    return "Success!", OUTPUT_VIDEO

# --- Gradio Interface ---
def main():
    # Initial setup
    print("Starting setup...")
    extract_success, extract_msg = extract_zip()
    print(extract_msg)
    
    voice_valid, voice_msg = validate_voice_samples()
    print(voice_msg)
    
    if not (extract_success and voice_valid):
        print("❌ Setup failed - check errors above")
        return
    
    # Create interface
    iface = gr.Interface(
        fn=process_lipsync,
        inputs=gr.Textbox(
            label="Enter Text",
            placeholder="What would Trump say? (Keep it under 20 words for best results)",
            lines=3
        ),
        outputs=[
            gr.Textbox(label="Status"),
            gr.Video(label="Lip-Synced Video")
        ],
        title="Trump Voice Cloning & Lip Sync",
        description="Generates a lip-synced video of Trump saying your text",
        examples=[
            ["The wall just got 10 feet higher!"],
            ["Nobody knows more about technology than me"],
            ["We're gonna win so much, you'll get tired of winning"]
        ],
        allow_flagging="never"
    )
    
    # Launch with improved error handling
    try:
        iface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    except Exception as e:
        print(f"Failed to launch Gradio: {str(e)}")

if __name__ == "__main__":
    main()