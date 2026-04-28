import librosa
from faster_whisper import WhisperModel

def analyze_call(audio_path):
    print(f"--- {audio_path} İşleniyor ---")
    
    # 1. Teknik Analiz (Süre ve Sessizlik)
    y, sr = librosa.load(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 2. Metne Dönüştürme (STT)
    # 'base' model hızlıdır, daha iyi sonuç için 'small' veya 'medium' seçebilirsin
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    
    full_text = ""
    print("\n[Konuşma Dökümü]:")
    for segment in segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
        full_text += segment.text + " "
    
    return {
        "duration": duration,
        "language": info.language,
        "text": full_text
    }

if __name__ == "__main__":
    # Test etmek için bir ses dosya yolu girin
    # analyze_call("test_cagri.wav")
    pass
