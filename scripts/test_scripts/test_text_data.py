import multiprocessing as mp
import time
from test_analyze_dictionary import call_analyze
import json

def process_one_sample(sample):
    sample_id = sample.get("id", "unknown")
    text = sample.get("text", "")
    start = time.perf_counter()
    verdict = call_analyze(case=1, text_for_analyze=text)   # ожидается 0 или 1
    elapsed = time.perf_counter() - start
    return (sample_id, verdict, elapsed)

def parallel_process_samples(json_samples, num_workers=None):
    with open(json_samples, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError("Неизвестный формат JSON: ожидается список объектов или объект с ключом 'samples'")

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 4)

    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_one_sample, samples)
    return (results, num_workers)
