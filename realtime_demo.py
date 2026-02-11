from vieneu import Vieneu
import numpy as np
import time
import threading
import queue
import sounddevice as sd

class RealtimePlayer:
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.stream = None
        
    def start(self):
        self.is_playing = True
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._audio_callback,
            blocksize=1024
        )
        self.stream.start()
        
    def _audio_callback(self, outdata, frames, time_info, status):
        try:
            data = self.audio_queue.get_nowait()
            if len(data) < frames:
                outdata[:len(data), 0] = data
                outdata[len(data):, 0] = 0
            else:
                outdata[:, 0] = data[:frames]
                if len(data) > frames:
                    self.audio_queue.put(data[frames:])
        except queue.Empty:
            outdata.fill(0)
            
    def add_chunk(self, audio_chunk):
        chunk_size = 1024
        for i in range(0, len(audio_chunk), chunk_size):
            self.audio_queue.put(audio_chunk[i:i+chunk_size].astype(np.float32))
            
    def stop(self):
        self.is_playing = False
        if self.stream:
            while not self.audio_queue.empty():
                time.sleep(0.1)
            time.sleep(0.5)
            self.stream.stop()
            self.stream.close()


def simple_realtime_demo():
    """Demo đơn giản: phát audio ngay khi sinh"""
    print("🔊 VieNeu-TTS: Real-time Streaming Demo")
    print("=" * 60)
    
    # Khởi tạo TTS
    print("\n⏳ Đang khởi tạo TTS engine...")
    tts = Vieneu(
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device="cpu",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cpu"
    )
    print("✅ TTS đã sẵn sàng!")
    
    voice = tts.get_preset_voice()
    
    # Text demo
    text = """
    Xin chào! Đây là demo phát âm thanh theo thời gian thực.
    Bạn sẽ nghe thấy giọng nói ngay khi mô hình đang sinh.
    Không cần đợi toàn bộ văn bản được xử lý xong.
    """
    
    print(f"\n📝 Text: {text.strip()}")
    print("\n" + "=" * 60)
    print("🎵 BẮT ĐẦU PHÁT REAL-TIME...")
    print("=" * 60)
    
    # Khởi tạo player
    player = RealtimePlayer(sample_rate=24000)
    player.start()
    
    start_time = time.perf_counter()
    first_audio = None
    chunk_count = 0
    
    # Stream inference và phát real-time
    for audio_chunk in tts.infer_stream(
        text=text,
        voice=voice,
        max_chars=256,
        temperature=1.0,
        top_k=50
    ):
        chunk_count += 1
        
        # Đo Time-to-First-Audio
        if first_audio is None:
            first_audio = time.perf_counter() - start_time
            print(f"\n🚀 FIRST AUDIO LATENCY: {first_audio*1000:.0f}ms")
            print("-" * 40)
        
        # Phát audio ngay lập tức
        player.add_chunk(audio_chunk)
        
        duration_ms = len(audio_chunk) / 24000 * 1000
        print(f"  🔊 Chunk {chunk_count}: {duration_ms:.0f}ms audio")
    
    # Đợi phát xong
    print("\n⏳ Đang phát nốt...")
    player.stop()
    
    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print("✅ HOÀN THÀNH!")
    print(f"   • First Audio Latency: {first_audio*1000:.0f}ms")
    print(f"   • Tổng thời gian: {total_time:.2f}s")
    print(f"   • Số chunks: {chunk_count}")
    print("=" * 60)
    
    tts.close()


def interactive_demo():
    """Demo tương tác - nhập text và nghe ngay"""
    print("🔊 VieNeu-TTS: Interactive Real-time Demo")
    print("=" * 60)
    print("Nhập văn bản và nghe ngay lập tức!")
    print("Gõ 'exit' để thoát.\n")
    
    # Khởi tạo TTS
    print("⏳ Đang khởi tạo TTS engine...")
    tts = Vieneu(
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device="cpu",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cpu"
    )
    print("✅ TTS đã sẵn sàng!\n")
    
    voice = tts.get_preset_voice()
    player = RealtimePlayer(sample_rate=24000)
    
    while True:
        try:
            text = input("📝 Nhập text: ").strip()
            
            if text.lower() == 'exit':
                break
                
            if not text:
                continue
            
            print("🎵 Đang phát...")
            player.start()
            
            for audio_chunk in tts.infer_stream(text=text, voice=voice):
                player.add_chunk(audio_chunk)
                
            player.stop()
            print("✅ Done!\n")
            
        except KeyboardInterrupt:
            break
    
    print("\n👋 Tạm biệt!")
    tts.close()


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("  VIENEU-TTS REAL-TIME STREAMING DEMO")
    print("=" * 60)
    print("\nChọn chế độ demo:")
    print("  1. simple      - Demo đơn giản với text mẫu")
    print("  2. interactive - Demo tương tác (nhập text)")
    print()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = input("Chọn (1/2) [mặc định: 1]: ").strip() or "1"
    
    if mode in ["1", "simple"]:
        simple_realtime_demo()
    elif mode in ["2", "interactive"]:
        interactive_demo()
    else:
        print("Chế độ không hợp lệ!")
