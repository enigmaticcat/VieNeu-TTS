"""
VieNeu-TTS: Optimized Low-Latency Streaming
Thử nghiệm các cấu hình để giảm TTFC
"""

from vieneu import Vieneu
import numpy as np
import soundfile as sf
import os
import time

def test_streaming_config(tts, text, voice_data, config_name):
    """Test streaming với config hiện tại và đo latency"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {config_name}")
    print(f"   streaming_frames_per_chunk: {tts.streaming_frames_per_chunk}")
    print(f"   streaming_lookforward: {tts.streaming_lookforward}")
    print(f"{'='*60}")
    
    audio_chunks = []
    
    start_time = time.perf_counter()
    first_chunk_time = None
    
    for chunk in tts.infer_stream(text=text, voice=voice_data, max_chars=256):
        if first_chunk_time is None:
            first_chunk_time = time.perf_counter() - start_time
            print(f"   🚀 TTFC: {first_chunk_time*1000:.0f}ms")
        audio_chunks.append(chunk)
    
    total_time = time.perf_counter() - start_time
    
    if audio_chunks:
        final_audio = np.concatenate(audio_chunks)
        audio_duration = len(final_audio) / 24000
        rtf = total_time / audio_duration
        
        print(f"   📊 Total time: {total_time*1000:.0f}ms")
        print(f"   📊 Audio: {audio_duration:.2f}s")
        print(f"   📊 RTF: {rtf:.2f}x")
        print(f"   📊 Chunks: {len(audio_chunks)}")
        
        return {
            "config": config_name,
            "ttfc": first_chunk_time * 1000,
            "total_time": total_time * 1000,
            "rtf": rtf,
            "chunks": len(audio_chunks),
            "audio": final_audio
        }
    return None


def main():
    print("🔬 VieNeu-TTS Latency Optimization Test")
    print("=" * 60)
    
    os.makedirs("outputs", exist_ok=True)
    
    # Khởi tạo model
    print("\n⏳ Loading model...")
    tts = Vieneu(
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device="cpu",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cpu"
    )
    
    voice_data = tts.get_preset_voice()
    
    # Text ngắn để test latency thuần
    short_text = "Xin chào, đây là bài test latency."
    
    # ================================================================
    # TEST 1: Default config
    # ================================================================
    result_default = test_streaming_config(
        tts, short_text, voice_data, 
        "Default (frames=25, lookforward=5)"
    )
    
    # ================================================================
    # TEST 2: Giảm streaming_frames_per_chunk
    # ================================================================
    tts.streaming_frames_per_chunk = 15  # Giảm từ 25 xuống 15
    tts.streaming_stride_samples = tts.streaming_frames_per_chunk * tts.hop_length
    
    result_fast = test_streaming_config(
        tts, short_text, voice_data,
        "Fast (frames=15, lookforward=5)"
    )
    
    # ================================================================
    # TEST 3: Giảm cả lookforward
    # ================================================================
    tts.streaming_frames_per_chunk = 10  # Giảm mạnh
    tts.streaming_lookforward = 3
    tts.streaming_stride_samples = tts.streaming_frames_per_chunk * tts.hop_length
    
    result_ultra = test_streaming_config(
        tts, short_text, voice_data,
        "Ultra Fast (frames=10, lookforward=3)"
    )
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("📊 LATENCY COMPARISON SUMMARY")
    print("=" * 60)
    
    results = [r for r in [result_default, result_fast, result_ultra] if r]
    
    for r in results:
        ttfc_indicator = "🟢" if r["ttfc"] < 500 else ("🟡" if r["ttfc"] < 1000 else "🔴")
        print(f"\n{r['config']}:")
        print(f"   {ttfc_indicator} TTFC: {r['ttfc']:.0f}ms")
        print(f"   RTF: {r['rtf']:.2f}x | Chunks: {r['chunks']}")
    
    # Lưu audio từ config tốt nhất
    if results:
        best = min(results, key=lambda x: x["ttfc"])
        sf.write("outputs/optimized_output.wav", best["audio"], 24000)
        print(f"\n💾 Best config ({best['config']}) saved to outputs/optimized_output.wav")
    
    # Reset về default
    tts.streaming_frames_per_chunk = 25
    tts.streaming_lookforward = 5
    tts.streaming_stride_samples = tts.streaming_frames_per_chunk * tts.hop_length
    
    tts.close()
    print("\n🎉 Test completed!")


if __name__ == "__main__":
    main()
