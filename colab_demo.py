"""
VieNeu-TTS Demo cho Google Colab
Không cần sounddevice - dùng IPython.display.Audio
"""

from vieneu import Vieneu
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

def gemini_stream(user_message: str, api_key: str, model: str = "gemini-2.0-flash"):
    """Stream text từ Gemini API"""
    from google import genai
    
    SENTENCE_DELIMITERS = '.!?;:'
    CLAUSE_DELIMITERS = ','
    
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content_stream(
        model=model,
        contents=f"Bạn là trợ lý ảo thông minh. Trả lời ngắn gọn, tự nhiên bằng tiếng Việt.\n\nUser: {user_message}"
    )
    
    buffer = ""
    for chunk in response:
        if chunk.text:
            buffer += chunk.text
            
            last_punct_idx = -1
            for delim in SENTENCE_DELIMITERS + CLAUSE_DELIMITERS:
                idx = buffer.rfind(delim)
                if idx > last_punct_idx:
                    last_punct_idx = idx
            
            if last_punct_idx >= 0:
                yield buffer[:last_punct_idx + 1].strip()
                buffer = buffer[last_punct_idx + 1:]
    
    if buffer.strip():
        yield buffer.strip()


def generate_speech(user_input: str, tts, voice, api_key: str):
    """Generate speech từ câu hỏi user"""
    print(f"\n🎤 User: {user_input}")
    print("-" * 50)
    
    all_audio = []
    all_text = []
    
    for text_chunk in gemini_stream(user_input, api_key):
        print(f"📝 LLM: {text_chunk}")
        all_text.append(text_chunk)
        
        for audio in tts.infer_stream(text=text_chunk, voice=voice):
            all_audio.append(audio)
            print(f"   🔊 Audio chunk: {len(audio)} samples")
    
    full_audio = np.concatenate(all_audio) if all_audio else np.array([])
    full_text = " ".join(all_text)
    
    print("-" * 50)
    print(f"✅ Full response: {full_text}")
    print(f"✅ Total audio: {len(full_audio)} samples ({len(full_audio)/24000:.2f}s)")
    
    return full_audio, full_text


def main():
    """Main function cho Colab"""
    from IPython.display import Audio, display
    
    # Lấy API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Nhập Gemini API Key: ").strip()
    
    print("🚀 Đang khởi tạo TTS engine...")
    tts = Vieneu(
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device="cpu",  # Đổi thành "cuda" nếu có GPU
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cpu"
    )
    print("✅ TTS sẵn sàng!")
    
    voice = tts.get_preset_voice()
    
    # Demo với câu hỏi mẫu
    user_input = input("\n💬 Nhập câu hỏi (hoặc Enter để dùng mẫu): ").strip()
    if not user_input:
        user_input = "Kể cho tôi một câu chuyện ngắn 3 câu"
    
    audio, text = generate_speech(user_input, tts, voice, api_key)
    
    # Phát audio trong Colab
    print("\n🔊 Đang phát audio...")
    display(Audio(audio, rate=24000, autoplay=True))
    
    tts.close()


if __name__ == "__main__":
    main()
