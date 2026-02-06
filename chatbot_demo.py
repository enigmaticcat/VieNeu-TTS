"""
🤖 VieNeu-TTS: Chatbot Demo với LLM + TTS Streaming
Mô phỏng hệ thống chatbot thời gian thực với latency thấp

Pipeline:
[User Input] → [LLM streaming] → [TTS streaming] → [Audio Output]
"""

from vieneu import Vieneu
import numpy as np
import soundfile as sf
import os
import time
from typing import Generator

# ============================================================================
# Simulated LLM Streaming (Giả lập LLM response)
# Thay thế bằng OpenAI, Ollama, hoặc LLM thực của bạn
# ============================================================================

def simulate_llm_stream(user_message: str) -> Generator[str, None, None]:
    """
    Giả lập LLM streaming response.
    Trong thực tế, thay thế bằng:
    - OpenAI: openai.ChatCompletion.create(stream=True)
    - Ollama: ollama.chat(stream=True)
    - Local LLM: llama-cpp-python với stream=True
    """
    # Simulated response based on input
    responses = {
        "xin chào": "Xin chào bạn! Tôi là trợ lý ảo VieNeu. Tôi có thể giúp gì cho bạn hôm nay?",
        "thời tiết": "Hôm nay thời tiết khá đẹp, trời nắng nhẹ và nhiệt độ khoảng 25 độ C. Rất thích hợp để ra ngoài dạo chơi.",
        "giới thiệu": "Tôi là VieNeu, một hệ thống chuyển văn bản thành giọng nói tiếng Việt. Tôi có thể đọc văn bản, clone giọng nói, và hỗ trợ chatbot thời gian thực.",
    }
    
    # Default response
    response = responses.get(
        user_message.lower().strip(),
        "Cảm ơn bạn đã nhắn tin. Đây là phản hồi mẫu từ chatbot. Trong thực tế, bạn có thể kết nối với các mô hình ngôn ngữ lớn như GPT hoặc Gemini."
    )
    
    # Simulate streaming: yield từng từ với delay nhỏ
    words = response.split()
    buffer = ""
    
    for i, word in enumerate(words):
        buffer += word + " "
        time.sleep(0.05)  # Simulate LLM token generation delay (50ms/token)
        
        # Yield khi đủ câu hoặc đủ dài
        if word.endswith(('.', '!', '?', ',')) or len(buffer) > 50:
            yield buffer.strip()
            buffer = ""
    
    # Yield remaining
    if buffer.strip():
        yield buffer.strip()


# ============================================================================
# Real LLM Integration Examples (Uncomment to use)
# ============================================================================

# def openai_llm_stream(user_message: str) -> Generator[str, None, None]:
#     """Stream từ OpenAI GPT"""
#     import openai
#     client = openai.OpenAI(api_key="your-api-key")
#     
#     stream = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": user_message}],
#         stream=True
#     )
#     
#     buffer = ""
#     for chunk in stream:
#         if chunk.choices[0].delta.content:
#             buffer += chunk.choices[0].delta.content
#             # Yield khi gặp dấu câu
#             if buffer.rstrip().endswith(('.', '!', '?', ',')):
#                 yield buffer.strip()
#                 buffer = ""
#     if buffer.strip():
#         yield buffer.strip()


# def ollama_llm_stream(user_message: str) -> Generator[str, None, None]:
#     """Stream từ Ollama local LLM"""
#     import ollama
#     
#     stream = ollama.chat(
#         model='llama3.2',
#         messages=[{"role": "user", "content": user_message}],
#         stream=True
#     )
#     
#     buffer = ""
#     for chunk in stream:
#         buffer += chunk['message']['content']
#         if buffer.rstrip().endswith(('.', '!', '?', ',')):
#             yield buffer.strip()
#             buffer = ""
#     if buffer.strip():
#         yield buffer.strip()


# ============================================================================
# Main Chatbot Pipeline
# ============================================================================

def chatbot_pipeline(user_message: str, tts: Vieneu, voice_data: dict):
    """
    Pipeline chatbot hoàn chỉnh:
    1. Nhận tin nhắn user
    2. Stream response từ LLM
    3. Stream TTS cho từng phần response
    4. Phát audio real-time
    """
    print(f"\n👤 User: {user_message}")
    print("=" * 60)
    
    pipeline_start = time.perf_counter()
    first_audio_time = None
    
    all_audio_chunks = []
    all_text_parts = []
    
    print("\n🤖 Assistant (streaming):")
    print("-" * 60)
    
    # Stream từ LLM và TTS song song
    for text_chunk in simulate_llm_stream(user_message):
        chunk_start = time.perf_counter()
        all_text_parts.append(text_chunk)
        
        print(f"  💬 LLM: \"{text_chunk}\"")
        
        # Stream TTS cho đoạn text này
        for audio_chunk in tts.infer_stream(
            text=text_chunk,
            voice=voice_data,
            max_chars=256,
            temperature=1.0,
            top_k=50
        ):
            if first_audio_time is None:
                first_audio_time = time.perf_counter() - pipeline_start
                print(f"\n  🚀 FIRST AUDIO LATENCY: {first_audio_time*1000:.0f}ms")
                print("-" * 60)
            
            all_audio_chunks.append(audio_chunk)
            audio_ms = len(audio_chunk) / 24000 * 1000
            print(f"  🔊 Audio chunk: {audio_ms:.0f}ms")
    
    # Tổng hợp kết quả
    pipeline_end = time.perf_counter()
    total_time = pipeline_end - pipeline_start
    
    if all_audio_chunks:
        final_audio = np.concatenate(all_audio_chunks)
        audio_duration = len(final_audio) / 24000
        rtf = total_time / audio_duration
    else:
        final_audio = np.array([])
        audio_duration = 0
        rtf = 0
    
    print("\n" + "=" * 60)
    print("📊 CHATBOT PIPELINE LATENCY REPORT")
    print("=" * 60)
    print(f"  📈 First Audio Latency:    {first_audio_time*1000:.0f}ms" if first_audio_time else "  📈 First Audio Latency:    N/A")
    print(f"  📈 Total Pipeline Time:    {total_time*1000:.0f}ms ({total_time:.2f}s)")
    print(f"  📈 Audio Duration:         {audio_duration*1000:.0f}ms ({audio_duration:.2f}s)")
    print(f"  📈 Real-Time Factor:       {rtf:.2f}x")
    print(f"  📈 LLM Chunks:             {len(all_text_parts)}")
    print(f"  📈 Audio Chunks:           {len(all_audio_chunks)}")
    print("=" * 60)
    
    return final_audio, " ".join(all_text_parts)


def main():
    print("🤖 VieNeu Chatbot Demo - LLM + TTS Streaming")
    print("=" * 60)
    
    os.makedirs("outputs", exist_ok=True)
    
    # Khởi tạo TTS
    print("\n⏳ Đang khởi tạo TTS engine...")
    init_start = time.perf_counter()
    
    tts = Vieneu(
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device="cpu",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cpu"
    )
    
    init_time = time.perf_counter() - init_start
    print(f"✅ TTS khởi tạo trong {init_time:.2f}s")
    
    voice_data = tts.get_preset_voice()
    
    # Demo các câu hỏi
    test_messages = [
        "xin chào",
        "giới thiệu",
        "thời tiết"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'#' * 60}")
        print(f"# TEST {i}/{len(test_messages)}")
        print(f"{'#' * 60}")
        
        audio, response_text = chatbot_pipeline(message, tts, voice_data)
        
        if len(audio) > 0:
            output_path = f"outputs/chatbot_response_{i}.wav"
            sf.write(output_path, audio, 24000)
            print(f"\n💾 Đã lưu: {output_path}")
    
    tts.close()
    print("\n🎉 Demo hoàn thành!")


if __name__ == "__main__":
    main()
