import runpod
import time  
import torchaudio 
import yt_dlp
import os
import tempfile
import base64
import requests
import time
from chatterbox.tts import ChatterboxTTS
from pathlib import Path

model = None
output_filename = "output.wav"

def handler(event, responseFormat="base64"):
    input = event['input']    
    prompt = input.get('prompt')
    audio_url = input.get('audio_url')
    audio_base64_input = input.get('audio_base64')
    yt_url = input.get('yt_url')

    print(f"New request. Prompt: {prompt}")

    os.makedirs("./my_audio", exist_ok=True)
    wav_file = None

    try:
        # --- Prioridad: audio_url > audio_base64 > yt_url ---
        if audio_url:
            print(f"Descargando audio de referencia desde: {audio_url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            }
            resp = requests.get(audio_url, headers=headers, timeout=60)
            resp.raise_for_status()
            wav_file = "./my_audio/reference.wav"
            with open(wav_file, "wb") as f:
                f.write(resp.content)

        elif audio_base64_input:
            print("Usando audio de referencia enviado en base64")
            wav_file = "./my_audio/reference.wav"
            with open(wav_file, "wb") as f:
                f.write(base64.b64decode(audio_base64_input))

        elif yt_url:
            print(f"Descargando audio de referencia desde YouTube: {yt_url}")
            result = download_youtube_audio(yt_url, output_path="./my_audio", audio_format="wav")
            if result is None:
                return {"error": "No se pudo descargar el audio de YouTube (revisa el link o usa audio_url en su lugar)"}
            dl_info, wav_file = result

        else:
            return {"error": "Necesitas mandar audio_url, audio_base64 o yt_url"}

        # Prompt Chatterbox
        audio_tensor = model.generate(
            prompt,
            audio_prompt_path=wav_file
        )

        # Save as WAV
        torchaudio.save(output_filename, audio_tensor, model.sr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": str(e)}

    # Convert to base64 string
    audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)

    if responseFormat == "base64":
        # Return base64
        response = {
            "status": "success",
            "audio_base64": audio_base64,
            "metadata": {
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape)
            }
        }
    elif responseFormat == "binary":
        with open(output_filename, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')

        # Clean up the file
        os.remove(output_filename)

        response = audio_data  # Just return the base64 string

    # Clean up temporary files (solo si existe)
    if wav_file and os.path.exists(wav_file):
        os.remove(wav_file)

    return response 

def audio_tensor_to_base64(audio_tensor, sample_rate):
    """Convert audio tensor to base64 encoded WAV data."""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio_tensor, sample_rate)
            
            # Read back as binary data
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            # Encode as base64
            return base64.b64encode(audio_data).decode('utf-8')
            
    except Exception as e:
        print(f"Error converting audio to base64: {e}")
        raise


def initialize_model():
    global model
    
    if model is not None:
        print("Model already initialized")
        return model
    
    print("Initializing ChatterboxTTS model...")
    model = ChatterboxTTS.from_pretrained(device="cuda")
    print("Model initialized")

def download_youtube_audio(url, output_path="./downloads", audio_format="mp3", duration_limit=60):
    """
    Download audio from a YouTube video
    
    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save the audio file
        audio_format (str): Audio format (mp3, wav, m4a, etc.)
    
    Returns:
        str: Path to the downloaded audio file, or None if download failed
    """
    
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',  # Download best quality audio
        'outtmpl': f'{output_path}/output.%(ext)s',  # Output filename template
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': '192',  # Audio quality in kbps
        }],
        'postprocessor_args': [
            '-ar', '44100'  # Set sample rate
        ],
        'prefer_ffmpeg': True,
    }
    if duration_limit:
        ydl_opts['postprocessors'].append({
            'key': 'FFmpegVideoConvertor',
            'preferedformat': audio_format,
        })
        # Add FFmpeg arguments for trimming
        ydl_opts['postprocessor_args'].extend([
            '-t', str(duration_limit)  # Trim to specified duration
        ])
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            video_duration = info.get('duration', 0)
            print(f"Title: {info.get('title', 'Unknown')}")
            print(f"Duration: {info.get('duration', 'Unknown')} seconds")
            print(f"Uploader: {info.get('uploader', 'Unknown')}")
        
            if duration_limit:
                actual_duration = min(duration_limit, video_duration)
                print(f"Downloading first {actual_duration} seconds")
            
            # Download the audio
            print("Downloading audio...")
            ydl.download([url])
            print("Download completed successfully!")

            expected_filepath = os.path.join(output_path, f"output.{audio_format}")
            
            return info, expected_filepath
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == '__main__':
    initialize_model()
    runpod.serverless.start({'handler': handler })
