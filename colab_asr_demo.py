"""
VieNeu-TTS: ASR + LLM + TTS Demo cho Google Colab
Pipeline: [Mic] → [ASR] → [LLM streaming] → [TTS streaming] → [Audio Output]

Sử dụng lại các hàm từ llm_tts_demo.py mà không sửa đổi.

Cài đặt trên Colab:
    !pip install faster-whisper google-genai python-dotenv
"""

import time
import os
import numpy as np
import base64
import io
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Import từ llm_tts_demo.py (KHÔNG sửa file gốc)
# ============================================================================
from llm_tts_demo import gemini_stream, ColabPlayer

# ============================================================================
# JavaScript ghi âm từ Microphone trên Colab
# ============================================================================

RECORD_JS = """
async function recordAudio(durationMs) {
    // Xin quyền truy cập microphone
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true
        }
    });
    
    // Auto-detect mimeType phù hợp với trình duyệt
    const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/ogg',
        'audio/mp4',
        ''  // fallback: để trình duyệt tự chọn
    ];
    let selectedMime = '';
    for (const mime of mimeTypes) {
        if (mime === '' || MediaRecorder.isTypeSupported(mime)) {
            selectedMime = mime;
            break;
        }
    }
    console.log('Using mimeType:', selectedMime || 'browser default');
    
    const recorderOptions = selectedMime ? {mimeType: selectedMime} : {};
    const mediaRecorder = new MediaRecorder(stream, recorderOptions);
    const chunks = [];
    
    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
    };
    
    // Bắt đầu ghi
    mediaRecorder.start();
    
    // Hiển thị đếm ngược
    const startTime = Date.now();
    const updateInterval = setInterval(() => {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        document.getElementById('record-status').innerText = 
            'Đang ghi âm... ' + elapsed + 's';
    }, 100);
    
    // Đợi đủ thời gian hoặc user nhấn stop
    await new Promise(resolve => {
        // Auto-stop sau durationMs
        setTimeout(() => {
            resolve();
        }, durationMs);
        
        // Hoặc user nhấn nút stop
        window._stopRecording = resolve;
    });
    
    clearInterval(updateInterval);
    mediaRecorder.stop();
    stream.getTracks().forEach(track => track.stop());
    
    // Đợi MediaRecorder xử lý xong
    await new Promise(resolve => {
        mediaRecorder.onstop = resolve;
    });
    
    // Convert sang base64
    const actualType = selectedMime || mediaRecorder.mimeType || 'audio/webm';
    const blob = new Blob(chunks, {type: actualType});
    const reader = new FileReader();
    const base64Promise = new Promise(resolve => {
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
    });
    reader.readAsDataURL(blob);
    
    return await base64Promise;
}
"""

STOP_BUTTON_JS = """
(function() {
    if (window._stopRecording) {
        window._stopRecording();
    }
})();
"""


def record_audio_colab(duration_seconds=10):
    """
    Ghi âm từ microphone trên Google Colab.
    
    Args:
        duration_seconds: Thời gian ghi tối đa (giây)
    
    Returns:
        numpy array chứa audio (float32)
    """
    from google.colab import output
    from IPython.display import display, HTML, Javascript
    
    # Tạo UI
    display(HTML("""
        <div style="padding: 10px; background: #1a1a2e; border-radius: 8px; 
                    color: white; font-family: monospace;">
            <p id="record-status" style="font-size: 16px;">
                Chuẩn bị ghi âm...
            </p>
            <button onclick="window._stopRecording && window._stopRecording()" 
                    style="padding: 8px 20px; background: #e94560; color: white; 
                           border: none; border-radius: 4px; cursor: pointer;
                           font-size: 14px; margin-top: 5px;">
                Dừng ghi âm
            </button>
        </div>
    """))
    
    # Inject JS và ghi âm
    duration_ms = duration_seconds * 1000
    audio_base64 = output.eval_js(RECORD_JS + f"\nrecordAudio({duration_ms})")
    
    if not audio_base64:
        print("Không ghi được âm thanh!")
        return None
    
    # Decode base64 → audio bytes
    audio_bytes = base64.b64decode(audio_base64)
    
    # Lưu tạm rồi dùng ffmpeg convert sang WAV
    tmp_input = "/tmp/colab_recording.bin"
    tmp_wav = "/tmp/colab_recording.wav"
    
    with open(tmp_input, "wb") as f:
        f.write(audio_bytes)
    
    # Convert sang wav 16kHz mono bằng ffmpeg (tự detect format đầu vào)
    os.system(f"ffmpeg -y -i {tmp_input} -ar 16000 -ac 1 {tmp_wav} -loglevel quiet")
    
    # Đọc WAV file
    import soundfile as sf
    audio_array, sr = sf.read(tmp_wav, dtype='float32')
    
    duration = len(audio_array) / sr
    print(f"Ghi âm xong: {duration:.1f}s ({len(audio_array)} samples, {sr}Hz)")
    
    return audio_array


def record_audio_local(duration_seconds=5, sample_rate=16000):
    """
    Ghi âm từ microphone trên local (dùng sounddevice).
    Fallback khi không chạy trên Colab.
    """
    import sounddevice as sd
    
    print(f"Đang ghi âm {duration_seconds}s... (nói tiếng Việt)")
    audio = sd.rec(
        int(duration_seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("Ghi âm xong!")
    
    return audio.flatten()


# ============================================================================
# ASR - Nhận dạng giọng nói
# ============================================================================

_asr_model = None

def load_asr_model(model_size="base", device="cuda", compute_type="float16"):
    """Load Faster-Whisper model (chỉ load 1 lần)."""
    global _asr_model
    if _asr_model is None:
        from faster_whisper import WhisperModel
        print(f"Đang tải ASR model (faster-whisper {model_size})...")
        try:
            _asr_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception:
            # Fallback: CPU với int8
            print("GPU không khả dụng cho ASR, dùng CPU...")
            _asr_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("ASR model đã sẵn sàng!")
    return _asr_model


def transcribe(audio_array, model_size="base"):
    """
    Chuyển audio → text bằng Faster-Whisper.
    
    Args:
        audio_array: numpy array (float32, 16kHz)
        model_size: kích thước model Whisper
    
    Returns:
        text đã nhận dạng
    """
    model = load_asr_model(model_size)
    
    segments, info = model.transcribe(
        audio_array,
        language="vi",
        beam_size=5,
        vad_filter=True  # Lọc khoảng im lặng
    )
    
    text = " ".join([segment.text for segment in segments]).strip()
    return text


# ============================================================================
# Pipeline ASR → LLM → TTS với đo latency
# ============================================================================

def asr_llm_tts_pipeline(tts, voice, api_key, 
                          asr_model_size="base",
                          record_duration=10,
                          gemini_model="gemini-2.5-flash"):
    """
    Pipeline đầy đủ: Mic → ASR → LLM → TTS
    
    Đo các mốc latency:
    - ASR Latency: thời gian nhận dạng giọng nói
    - LLM First Token: thời gian chờ LLM token đầu tiên
    - TTFC: LLM text đầu tiên → Audio đầu tiên 
    - E2E Latency: từ nhấn Enter (kết thúc ghi âm) → Audio đầu tiên
    """
    from IPython.display import Audio, display
    
    # === BƯỚC 0: Ghi âm ===
    print("\n" + "=" * 60)
    print("BƯỚC 1: GHI ÂM")
    print("=" * 60)
    
    audio_input = record_audio_colab(duration_seconds=record_duration)
    if audio_input is None or len(audio_input) == 0:
        print("Không có âm thanh, thử lại!")
        return None
    
    # ====================================================
    # t_start: Mốc bắt đầu đo E2E (ngay sau khi ghi âm xong / nhấn Enter)
    # ====================================================
    t_start = time.perf_counter()
    
    # === BƯỚC 1: ASR ===
    print("\n" + "=" * 60)
    print("BƯỚC 2: NHẬN DẠNG GIỌNG NÓI (ASR)")
    print("=" * 60)
    
    recognized_text = transcribe(audio_input, model_size=asr_model_size)
    t_asr_done = time.perf_counter()
    asr_latency = t_asr_done - t_start
    
    print(f"ASR kết quả: \"{recognized_text}\"")
    print(f"ASR Latency: {asr_latency*1000:.0f}ms")
    
    if not recognized_text:
        print("Không nhận dạng được giọng nói!")
        return None
    
    # === BƯỚC 2: LLM + TTS Streaming ===
    print("\n" + "=" * 60)
    print("BƯỚC 3: LLM + TTS STREAMING")
    print("=" * 60)
    
    llm_fn = lambda msg: gemini_stream(msg, api_key, gemini_model)
    
    # Sử dụng ColabPlayer từ llm_tts_demo.py
    player = ColabPlayer(sample_rate=24000)
    player.start()
    
    all_audio = []
    all_texts = []
    t_first_llm = None
    t_first_audio = None
    chunk_count = 0
    
    for text_chunk in llm_fn(recognized_text):
        # Đo LLM first token
        if t_first_llm is None:
            t_first_llm = time.perf_counter()
            llm_first_token_latency = t_first_llm - t_asr_done
            print(f"\nLLM first token: {llm_first_token_latency*1000:.0f}ms (sau ASR)")
        
        all_texts.append(text_chunk)
        print(f"  LLM: \"{text_chunk}\"")
        
        # TTS streaming
        for audio_chunk in tts.infer_stream(
            text=text_chunk,
            voice=voice,
            max_chars=256,
            temperature=1.0,
            top_k=50
        ):
            chunk_count += 1
            
            # Đo first audio
            if t_first_audio is None:
                t_first_audio = time.perf_counter()
                ttfc = t_first_audio - t_first_llm
                e2e_latency = t_first_audio - t_start
                print(f"\n  FIRST AUDIO!")
                print(f"     TTFC (LLM→Audio): {ttfc*1000:.0f}ms")
                print(f"     E2E  (Enter→Audio): {e2e_latency*1000:.0f}ms")
            
            player.add_chunk(audio_chunk)
            all_audio.append(audio_chunk)
            
            audio_ms = len(audio_chunk) / 24000 * 1000
            print(f"  Audio chunk {chunk_count}: {audio_ms:.0f}ms")
    
    # Phát audio
    player.stop()
    
    t_end = time.perf_counter()
    total_time = t_end - t_start
    
    # === BÁO CÁO LATENCY ===
    print("\n" + "=" * 60)
    print("LATENCY REPORT: ASR → LLM → TTS")
    print("=" * 60)
    
    full_text = " ".join(all_texts)
    full_audio = np.concatenate(all_audio) if all_audio else np.array([])
    audio_duration = len(full_audio) / 24000 if len(full_audio) > 0 else 0
    
    print(f"  Input:               \"{recognized_text}\"")
    print(f"  Response:            \"{full_text[:100]}{'...' if len(full_text) > 100 else ''}\"")
    print(f"  ─────────────────────────────────────────")
    print(f"  ASR Latency:         {asr_latency*1000:.0f}ms")
    
    if t_first_llm:
        print(f"  LLM First Token:     {llm_first_token_latency*1000:.0f}ms")
    
    if t_first_audio:
        print(f"  TTFC (LLM→Audio):    {ttfc*1000:.0f}ms")
        print(f"  E2E  (Enter→Audio):  {e2e_latency*1000:.0f}ms  ← ĐỘ TRỄ TỔNG")
    
    print(f"  Total Pipeline:      {total_time*1000:.0f}ms")
    print(f"  Audio Duration:       {audio_duration:.2f}s")
    print(f"  Audio Chunks:         {chunk_count}")
    
    if audio_duration > 0:
        rtf = total_time / audio_duration
        print(f"  Real-Time Factor:     {rtf:.2f}x")
    
    print("=" * 60)
    
    return {
        "recognized_text": recognized_text,
        "response_text": full_text,
        "audio": full_audio,
        "asr_latency_ms": asr_latency * 1000,
        "llm_first_token_ms": llm_first_token_latency * 1000 if t_first_llm else None,
        "ttfc_ms": ttfc * 1000 if t_first_audio else None,
        "e2e_latency_ms": e2e_latency * 1000 if t_first_audio else None,
        "total_time_ms": total_time * 1000,
    }


# ============================================================================
# Main - chạy trên Colab
# ============================================================================

def main():
    """
    Main function - chạy trên Google Colab.
    
    Cách dùng trong Colab notebook:
        from colab_asr_demo import main
        main()
    
    Hoặc chạy từng bước:
        from colab_asr_demo import load_asr_model, asr_llm_tts_pipeline
        from vieneu import Vieneu
    """
    from IPython.display import display, HTML
    
    print("VieNeu: Voice-to-Voice Demo")
    print("Pipeline: Mic → ASR → LLM → TTS")
    print("=" * 60)
    
    # --- Cấu hình ---
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("GEMINI_API_KEY")
        except Exception:
            pass
    if not api_key:
        api_key = input("Nhập Gemini API Key: ").strip()
    
    # --- Load models ---
    print("\nĐang tải ASR model...")
    load_asr_model(model_size="base")
    
    print("\nĐang tải TTS engine...")
    from vieneu import Vieneu
    
    # Auto-detect GPU
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    
    if has_cuda:
        tts = Vieneu(
            backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
            backbone_device="gpu",
            codec_repo="neuphonic/distill-neucodec",
            codec_device="cuda"
        )
        print("GPU mode!")
    else:
        tts = Vieneu(
            backbone_repo="pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf",
            backbone_device="cpu",
            codec_repo="neuphonic/distill-neucodec",
            codec_device="cpu"
        )
        print("CPU mode")
    
    voice = tts.get_preset_voice()
    print("Tất cả models đã sẵn sàng!\n")
    
    # --- Vòng lặp hội thoại ---
    round_num = 0
    while True:
        round_num += 1
        print(f"\n{'#' * 60}")
        print(f"# LƯỢT {round_num}")
        print(f"{'#' * 60}")
        
        cont = input("\nNhấn Enter để bắt đầu ghi âm (hoặc gõ 'exit'): ").strip()
        if cont.lower() == 'exit':
            break
        
        result = asr_llm_tts_pipeline(
            tts=tts,
            voice=voice,
            api_key=api_key,
            asr_model_size="base",
            record_duration=10,
            gemini_model="gemini-2.5-flash"
        )
        
        if result:
            print(f"\nLượt {round_num} hoàn thành!")
    
    tts.close()
    print("\nKết thúc demo!")


if __name__ == "__main__":
    main()
