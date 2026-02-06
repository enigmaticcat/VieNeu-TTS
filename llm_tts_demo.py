from vieneu import Vieneu
import numpy as np
import time
import threading
import queue
import sounddevice as sd
import os
from dotenv import load_dotenv

load_dotenv()

# Các dấu câu để tách text
SENTENCE_DELIMITERS = '.!?;:'
CLAUSE_DELIMITERS = ','  

def ollama_stream(user_message: str, model: str = "llama3.2"):
    import ollama
    
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": "Bạn là trợ lý ảo thông minh. Trả lời ngắn gọn, tự nhiên bằng tiếng Việt."},
            {"role": "user", "content": user_message}
        ],
        stream=True
    )
    
    buffer = ""
    for chunk in stream:
        content = chunk['message']['content']
        buffer += content
        
        # Tìm dấu câu cuối cùng trong buffer
        last_punct_idx = -1
        for delim in SENTENCE_DELIMITERS + CLAUSE_DELIMITERS:
            idx = buffer.rfind(delim)
            if idx > last_punct_idx:
                last_punct_idx = idx
        
        # Nếu tìm thấy dấu câu, yield phần trước (bao gồm dấu câu)
        if last_punct_idx >= 0:
            yield buffer[:last_punct_idx + 1].strip()
            buffer = buffer[last_punct_idx + 1:]
    
    if buffer.strip():
        yield buffer.strip()


def gemini_stream(user_message: str, api_key: str, model: str = "gemini-2.0-flash"):
    from google import genai
    
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content_stream(
        model=model,
        contents=f"Bạn là trợ lý ảo thông minh. Trả lời ngắn gọn, tự nhiên bằng tiếng Việt.\n\nUser: {user_message}"
    )
    
    buffer = ""
    for chunk in response:
        if chunk.text:
            buffer += chunk.text
            
            # Tìm dấu câu cuối cùng trong buffer
            last_punct_idx = -1
            for delim in SENTENCE_DELIMITERS + CLAUSE_DELIMITERS:
                idx = buffer.rfind(delim)
                if idx > last_punct_idx:
                    last_punct_idx = idx
            
            # Nếu tìm thấy dấu câu, yield phần trước (bao gồm dấu câu)
            if last_punct_idx >= 0:
                yield buffer[:last_punct_idx + 1].strip()
                buffer = buffer[last_punct_idx + 1:]
    
    if buffer.strip():
        yield buffer.strip()

class RealtimePlayer:
    HOP_LENGTH = 480
    STREAMING_FRAMES_PER_CHUNK = 25
    STREAMING_STRIDE_SAMPLES = STREAMING_FRAMES_PER_CHUNK * HOP_LENGTH  
    
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        
        self.audio_chunks = []
        self.n_decoded_samples = 0
        
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        self.min_buffer = sample_rate // 2  
        self.started_output = False
        
        self.is_playing = False
        self.stream = None
        self.channels = 2
        
    def start(self):
        self.is_playing = True
        self.audio_chunks = []
        self.n_decoded_samples = 0
        self.buffer = np.array([], dtype=np.float32)
        self.started_output = False
        
        devices_to_try = [None, 4, 2]
        
        for device in devices_to_try:
            try:
                if device is not None:
                    device_info = sd.query_devices(device)
                    channels = min(device_info['max_output_channels'], 2)
                    if channels == 0:
                        continue
                else:
                    channels = 2
                
                self.channels = channels
                self.stream = sd.OutputStream(
                    device=device,
                    samplerate=self.sample_rate,
                    channels=channels,
                    dtype='float32',
                    callback=self._audio_callback,
                    blocksize=2048,
                    latency='high'
                )
                self.stream.start()
                device_name = sd.query_devices(device)['name'] if device is not None else "default"
                print(f"Audio: {device_name}")
                return
            except Exception:
                continue
        
        print("Audio error: No device")
        self.stream = None
        
    def _audio_callback(self, outdata, frames, time_info, status):
        with self.buffer_lock:
            if not self.started_output and len(self.buffer) < self.min_buffer:
                outdata.fill(0)
                return
            
            self.started_output = True
            
            if len(self.buffer) >= frames:
                for ch in range(self.channels):
                    outdata[:, ch] = self.buffer[:frames]
                self.buffer = self.buffer[frames:]
            elif len(self.buffer) > 0:
                for ch in range(self.channels):
                    outdata[:len(self.buffer), ch] = self.buffer
                outdata[len(self.buffer):, :] = 0
                self.buffer = np.array([], dtype=np.float32)
            else:
                outdata.fill(0)
            
    def add_chunk(self, audio_chunk):
        with self.buffer_lock:
            audio_chunk = audio_chunk.astype(np.float32)
            self.buffer = np.concatenate([self.buffer, audio_chunk])
            
    def stop(self):
        self.is_playing = False
        if self.stream:
            initial_buffer = len(self.buffer)
            print(f"  [PLAYER] Waiting for buffer to drain... ({initial_buffer} samples)")
            max_wait = 300  
            while len(self.buffer) > 0 and max_wait > 0:
                time.sleep(0.1)
                max_wait -= 1
            remaining = len(self.buffer)
            if remaining > 0:
                print(f"  [PLAYER] WARNING: Buffer not fully drained! {remaining} samples remaining")
            else:
                print(f"  [PLAYER] Buffer drained successfully")
            time.sleep(0.3)
            self.stream.stop()
            self.stream.close()


def parallel_llm_tts(user_input, tts, voice, llm_fn, player):
    
    text_queue = queue.Queue()
    results = {
        "ttfc": None,           
        "first_llm": None,      
        "first_audio": None,    
        "first_llm_time": None, 
        "texts": [], 
        "audio_chunks": 0,
        "debug_logs": []        
    }
    start_time = time.perf_counter()
    
    def log(msg):
        elapsed = (time.perf_counter() - start_time) * 1000
        log_entry = f"[{elapsed:7.0f}ms] {msg}"
        results["debug_logs"].append(log_entry)
        print(f"  {log_entry}")
    
    def llm_worker():
        first_text = True
        chunk_idx = 0
        for text_chunk in llm_fn(user_input):
            chunk_idx += 1
            if first_text:
                results["first_llm"] = time.perf_counter() - start_time
                results["first_llm_time"] = time.perf_counter()
                log(f"LLM FIRST TEXT: '{text_chunk[:40]}...' (len={len(text_chunk)})")
                first_text = False
            else:
                log(f"LLM chunk {chunk_idx}: '{text_chunk[:30]}...'")
            results["texts"].append(text_chunk)
            text_queue.put(text_chunk)
        text_queue.put(None)
        log("LLM DONE")
    
    llm_thread = threading.Thread(target=llm_worker)
    llm_thread.start()
    
    player.start()
    text_idx = 0
    
    while True:
        wait_start = time.perf_counter()
        text = text_queue.get()
        wait_time = (time.perf_counter() - wait_start) * 1000
        
        if text is None:
            break
        
        text_idx += 1
        log(f"TTS START chunk {text_idx}: '{text[:30]}...' (waited {wait_time:.0f}ms for queue)")
        
        tts_start = time.perf_counter()
        audio_idx = 0
        
        for audio in tts.infer_stream(text=text, voice=voice):
            
            audio_idx += 1
            
            if results["first_audio"] is None:
                results["first_audio"] = time.perf_counter() - start_time
                if results["first_llm_time"]:
                    results["ttfc"] = time.perf_counter() - results["first_llm_time"]
                log(f"FIRST AUDIO! chunk={text_idx}, audio_idx={audio_idx}, samples={len(audio)}")
            
            player.add_chunk(audio)
            results["audio_chunks"] += 1
        

        
        tts_time = (time.perf_counter() - tts_start) * 1000
        log(f"TTS DONE chunk {text_idx}: {audio_idx} audio chunks in {tts_time:.0f}ms")
    
    llm_thread.join()
    player.stop()
    
    results["total_time"] = time.perf_counter() - start_time
    return results

def main():
    print("Demo")
    print("=" * 60)
    
    print("\nChon LLM provider:")
    print("  1. Ollama")
    print("  2. Gemini API")
    
    choice = input("Chon (1/2) [mac dinh: 1]: ").strip() or "1"
    
    if choice == "2":
        api_key = os.getenv("GEMINI_API_KEY") or input("Nhap Gemini API Key: ").strip()
        model = input("Nhap model Gemini [mac dinh: gemini-2.5-flash]: ").strip() or "gemini-2.0-flash"
        llm_fn = lambda msg: gemini_stream(msg, api_key, model)
        provider_name = f"Gemini/{model}"
    else:
        model = input("Nhap ten model Ollama [mac dinh: qwen2.5:1.5b-instruct]: ").strip() or "qwen2.5:1.5b-instruct"
        llm_fn = lambda msg: ollama_stream(msg, model)
        provider_name = f"Ollama/{model}"
    
    print("\nDang khoi tao TTS engine...")
    tts = Vieneu(
        backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
        backbone_device="cpu",
        codec_repo="neuphonic/distill-neucodec",
        codec_device="cpu"
    )
    print(f"TTS san sang (LLM: {provider_name})")
    
    voice = tts.get_preset_voice()
    player = RealtimePlayer(sample_rate=24000)
    print("Player: ready (split by punctuation)")
    
    print("\n" + "=" * 60)
    print("Nhap cau hoi va nghe phan hoi real-time")
    print("Go 'exit' de thoat.")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nBan: ").strip()
            
            if user_input.lower() == 'exit':
                break
            if not user_input:
                continue
            
            print("\nAssistant:")
            print("-" * 40)
            
            results = parallel_llm_tts(user_input, tts, voice, llm_fn, player)
            
            print("\n" + "-" * 40)
            print("LATENCY REPORT:")
            print(f"   Time to First LLM Text: {results['first_llm']*1000:.0f}ms" if results['first_llm'] else "   TTFL: N/A")
            print(f"   TTFC (LLM text -> Audio): {results['ttfc']*1000:.0f}ms" if results['ttfc'] else "   TTFC: N/A")
            print(f"   Total Time: {results['total_time']*1000:.0f}ms")
            print(f"   Audio Chunks: {results['audio_chunks']}")
                 
            print("\nFULL RESPONSE TEXT:")
            full_text = " ".join(results['texts'])
            print(f"   {full_text}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Loi: {e}")
    
    print("\nKet thuc")
    tts.close()


if __name__ == "__main__":
    main()
