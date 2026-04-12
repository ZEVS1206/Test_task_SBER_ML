import onnx_asr


def get_text_for_analyze_gigaam_model(file_path = "../../vishing/samples/Fraud/out_a_1.wav"):

    vad = onnx_asr.load_vad("silero")
    model = onnx_asr.load_model("gigaam-v3-e2e-rnnt").with_vad(vad)

    try:
        transcription_segments = model.recognize(file_path)
        full_text = []
        for segment_result in transcription_segments:
            full_text.append(segment_result.text)
        full_text.append("\n")
        final_text = " ".join(full_text)
        output_test_file = "text_for_analyze.txt"
        with open(output_test_file, "w") as f:
            f.write(final_text)
        return final_text
    except Exception as e:
        print(f"Произошла ошибка при распознавании: {e}")

if __name__ == "__main__":
    text = get_text_for_analyze_gigaam_model()
    print(text)
