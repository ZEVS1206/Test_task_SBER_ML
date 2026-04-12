import multiprocessing as mp
from pathlib import Path
import csv
from test_vosk_model import get_text_for_analyze_vosk_model
from test_gigaam_model import get_text_for_analyze_gigaam_model
from test_analyze_dictionary import call_analyze
import time

def analyze_verdict(verdict):
    if (verdict > 0.7):
        return 0
    return 1

def process_single_file(file_path, model):
    text_for_analyze = ""
    t0 = time.perf_counter()
    if (model == "vosk"):
        text_for_analyze = get_text_for_analyze_vosk_model(file_path = file_path)
        verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
        verdict = analyze_verdict(verdict)
    elif (model == "gigaam"):
        text_for_analyze = get_text_for_analyze_gigaam_model(file_path = file_path)
        verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
        verdict = analyze_verdict(verdict)
    elif (model == "both"):
        text_for_analyze = get_text_for_analyze_vosk_model(file_path = file_path)
        verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
        if (verdict > 0.2 and verdict <= 0.7):
            text_for_analyze = get_text_for_analyze_gigaam_model(file_path = file_path)
            verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
            verdict = analyze_verdict(verdict)
        elif (verdict <= 0.2):
            verdict = 1
        else:
            verdict = 0
    elapsed = time.perf_counter() - t0
    return (Path(file_path), verdict, elapsed)

def parallel_process_folder(model, folder_path = "../../vishing/samples/Fraud"):
    folder = Path(folder_path)

    audio_files = list(folder.glob("*.wav"))
    num_of_processes = mp.cpu_count() - 4
    print(f"Запускаем {num_of_processes} процессов...")

    with mp.Pool(processes=num_of_processes) as pool:
        results = pool.starmap(process_single_file, [(str(file), model) for file in audio_files])
    return (results, num_of_processes)

def main():
    model       = input("Введите модель для распознавания аудио(vosk, gigaam, both):")
    folder_path = input("Введите название папки с выборкой:")
    total_path = "../../vishing/" + folder_path
    results, num_of_processes = parallel_process_folder(model, folder_path = total_path)
    count_frauds = sum(1 for _, label, _ in results if label == 0)
    percentage_frauds = count_frauds / len(results) * 100
    print(f"Процент мошеннических разговоров в папке = {percentage_frauds}%")
    total_time = sum(elapsed for _, _, elapsed in results)
    avg_time   = total_time / len(results)
    max_file, _, max_time = max(results, key=lambda x: x[2])
    min_file, _, min_time = min(results, key=lambda x: x[2])
    print(f"Общее время анализа по всем процессам = {total_time:.2f}с")
    print(f"Реальное время анализа = {total_time / num_of_processes}с")
    print(f"Среднее время на один файл = {avg_time:.2f}с")
    print(f"Самый долгий файл {max_file} : {max_time:.2f}с")
    print(f"Самый быстрый файл {min_file} : {min_time:.2f}с")
    #results = []
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    folder_name = Path(total_path).name
    file_with_results = results_dir / f"{folder_name}_results.csv"
    with open(file_with_results, "w", newline= "", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Название файла", "label", "Время анализа"])
        writer.writerows(results)

if __name__ == "__main__":
    main()
