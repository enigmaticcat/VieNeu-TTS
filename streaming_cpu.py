"""
VieNeu-TTS: CPU Streaming Demo with Latency Measurement
Đo latency khi sinh âm thanh streaming trên CPU
"""

from vieneu import Vieneu
import numpy as np
import soundfile as sf
import os
import time

def main():
    print("Khởi tạo VieNeu-TTS cho CPU streaming")
    print("=" * 60)
    
    os.makedirs("outputs", exist_ok=True)
    
    init_start = time.perf_counter()
    
    tts = Vieneu(
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device="cpu",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cpu"
    )
    
    init_time = time.perf_counter() - init_start
    print(f"\n Thời gian khởi tạo model: {init_time:.2f}s")
    print("=" * 60)
    
    # Text cần đọc
    text = """
    Xin chào, đây là demo TTS streaming chạy hoàn toàn trên CPU.
    Chế độ streaming cho phép phát âm thanh ngay khi đang sinh.
    Không cần đợi toàn bộ văn bản được xử lý xong.
    """
    
    # Lấy voice mặc định
    voice_data = tts.get_preset_voice()
    print(f"🎤 Sử dụng voice mặc định")
    print(f"Text: {len(text.strip())} ký tự")
    print("=" * 60)
    
    print("\n Bắt đầu streaming inference...")
    print("-" * 60)
    
    # ===== Đo latency cho từng chunk =====
    audio_chunks = []
    chunk_latencies = []
    chunk_count = 0
    
    # Thời điểm bắt đầu inference
    inference_start = time.perf_counter()
    first_chunk_time = None
    
    for audio_chunk in tts.infer_stream(
        text=text,
        voice=voice_data,
        max_chars=256,
        temperature=1.0,
        top_k=50
    ):
        chunk_end = time.perf_counter()
        chunk_count += 1
        
        # Đo Time-to-First-Chunk (TTFC)
        if first_chunk_time is None:
            first_chunk_time = chunk_end - inference_start
            print(f"\n  TIME-TO-FIRST-CHUNK (TTFC): {first_chunk_time*1000:.0f}ms")
            print("-" * 60)
        
        audio_chunks.append(audio_chunk)
        
        # Tính latency và thông tin chunk
        chunk_duration_ms = len(audio_chunk) / 24000 * 1000  # Audio duration
        elapsed = chunk_end - inference_start
        chunk_latencies.append(elapsed)
        
        print(f"  Chunk {chunk_count}:")
        print(f"     • Samples: {len(audio_chunk):,}")
        print(f"     • Audio duration: {chunk_duration_ms:.0f}ms")
        print(f"     • Elapsed time: {elapsed*1000:.0f}ms")
    
    # ===== Tổng hợp kết quả latency =====
    inference_end = time.perf_counter()
    total_inference_time = inference_end - inference_start
    
    print("-" * 60)
    print("\n    LATENCY REPORT")
    print("=" * 60)
    
    # Ghép audio
    final_audio = np.concatenate(audio_chunks)
    total_audio_duration = len(final_audio) / 24000
    
    # Tính Real-Time Factor (RTF)
    rtf = total_inference_time / total_audio_duration
    
    print(f" Time-to-First-Chunk (TTFC): {first_chunk_time*1000:.0f}ms")
    print(f" Total inference time:       {total_inference_time*1000:.0f}ms ({total_inference_time:.2f}s)")
    print(f" Total audio duration:       {total_audio_duration*1000:.0f}ms ({total_audio_duration:.2f}s)")
    print(f" Real-Time Factor (RTF):     {rtf:.2f}x")
    print(f"     └─ RTF < 1.0 = Faster than real-time ")
    print(f"     └─ RTF > 1.0 = Slower than real-time ")
    print(f" Chunks generated:           {chunk_count}")
    print(f" Avg latency per chunk:      {(total_inference_time/chunk_count)*1000:.0f}ms")
    
    print("=" * 60)
    
    # Lưu file WAV
    output_path = "outputs/streaming_output.wav"
    sf.write(output_path, final_audio, 24000)
    print(f"\n Đã lưu: {output_path}")
    
    # Cleanup
    tts.close()
    print("Done!")

if __name__ == "__main__":
    main()
