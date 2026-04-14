# import os
# os.environ["ORT_CUDA_CACHE_PATH"] = "/tmp/onnx_cache"
# os.environ["ORT_CUDA_MEMORY_LIMIT"] = "3072"
import onnx_asr
import onnxruntime as ort
ort.set_default_logger_severity(3)


def get_text_for_analyze_gigaam_model(mode = "cpu", file_path = "../../vishing/samples/Fraud/out_a_1.wav"):

    cuda_provider_options = {
        "device_id": 0,
        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2 ГБ на процесс
        "arena_extend_strategy": "kNextPowerOfTwo",
    }
    cuda_provider = [('CUDAExecutionProvider', cuda_provider_options)]
    cpu_provider  = ['CPUExecutionProvider']
    provider = []
    if (mode == "gpu" or mode == "both"):
        provider = cuda_provider
    else:
        provider = cpu_provider

    vad = onnx_asr.load_vad("silero", providers=provider)
    model = onnx_asr.load_model("gigaam-v3-e2e-rnnt", providers=provider).with_vad(vad)

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
