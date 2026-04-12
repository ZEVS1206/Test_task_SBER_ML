import json
from vosk import Model, KaldiRecognizer
import librosa
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps



def float32_to_int16(audio_float32):
    audio_int16 = (audio_float32 * 32767).astype(np.int16)
    return audio_int16.tobytes()

def transcribe_segment(segment, recognizer):
    recognizer.Reset()
    recognizer.AcceptWaveform(float32_to_int16(segment))
    result_json = recognizer.FinalResult()
    result = json.loads(result_json)
    return result.get("text", "")

def get_text_for_analyze_vosk_model(file_path = "../../vishing/samples/Fraud/out_a_1.wav"):
    vad_model = load_silero_vad(onnx=True)
    test_file, sr = librosa.load(file_path, sr=16000, mono=True)
    speech_timestamps = get_speech_timestamps(test_file, vad_model, sampling_rate=16000, threshold=0.75, window_size_samples=1536, speech_pad_ms=550)

    #model location
    model_location_path = "../../vosk-model-small-ru-0.22"
    model = Model(model_location_path)
    full_text = []

    for i, time_stamp in enumerate(speech_timestamps):
        segment = test_file[time_stamp['start']:time_stamp['end']]
        recognizer = KaldiRecognizer(model, sr)
        text = transcribe_segment(segment, recognizer)
        if (text):
            full_text.append(text)

    full_text_str = " ".join(full_text)
    output_test_file = "text_for_analyze.txt"
    with open(output_test_file, "w") as f:
        f.write(full_text_str)
    return full_text_str

if __name__ == "__main__":
    get_text_for_analyze_vosk_model()
