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

def process_single_file(file_path, model, mode):
    text_for_analyze = ""
    t0 = time.perf_counter()
    if (model == "vosk"):
        text_for_analyze = get_text_for_analyze_vosk_model(file_path = file_path)
        verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
        verdict = analyze_verdict(verdict)
    elif (model == "gigaam"):
        text_for_analyze = get_text_for_analyze_gigaam_model(mode, file_path = file_path)
        verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
        verdict = analyze_verdict(verdict)
    elif (model == "both"):
        text_for_analyze = get_text_for_analyze_vosk_model(file_path = file_path)
        verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
        if (verdict > 0.2 and verdict <= 0.7):
            text_for_analyze = get_text_for_analyze_gigaam_model(mode, file_path = file_path)
            verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
            verdict = analyze_verdict(verdict)
        elif (verdict <= 0.2):
            verdict = 1
        else:
            verdict = 0
    elapsed = time.perf_counter() - t0
    return (Path(file_path), verdict, elapsed)

def parallel_process_folder(mode, model, folder_path = "../../vishing/samples/Fraud"):
    folder = Path(folder_path)

    audio_files = list(folder.glob("*.wav"))
    num_of_processes = max(1, mp.cpu_count() - 4)
    print(f"Запускаем {num_of_processes} процессов...")

    with mp.Pool(processes=num_of_processes) as pool:
        results = pool.starmap(process_single_file, [(str(file), model, mode) for file in audio_files])
    return (results, num_of_processes)
def one_process_folder_by_gpu(mode, model, folder_path = "../../vishing/samples/Fraud"):
    folder = Path(folder_path)
    audio_files = list(folder.glob("*.wav"))
    results = []
    for file in audio_files:
        results.append(process_single_file(str(file), model, mode))
    return results


def gpu_worker(gpu_queue, result_queue):
    while True:
        item = gpu_queue.get()
        if item is None:
            break
        file_path, start_time = item
        if file_path is None:
            break
        text_for_analyze = get_text_for_analyze_gigaam_model(mode = "gpu", file_path = file_path)
        verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
        end_time = time.perf_counter() - start_time
        if (verdict > 0.7):
            result_queue.put((file_path, 0, end_time))
        else:
            result_queue.put((file_path, 1, end_time))

def cpu_worker(task_queue, gpu_queue, result_queue, model):
    while True:
        try:
            file_path = task_queue.get(timeout = 1)
        except:
            break
        if (file_path is None):
            break
        start_time = time.perf_counter()
        if (model == "both"):
            text_for_analyze = get_text_for_analyze_vosk_model(file_path = file_path)
            verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
            if (verdict <= 0.2):
                result_queue.put((file_path, 1, time.perf_counter() - start_time))
            elif (verdict > 0.7):
                result_queue.put((file_path, 0, time.perf_counter() - start_time))
            else:
                gpu_queue.put((file_path, start_time))
        elif (model == "vosk"):
            text_for_analyze = get_text_for_analyze_vosk_model(file_path = file_path)
            verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
            verdict = analyze_verdict(verdict)
            result_queue.put((file_path, verdict, time.perf_counter() - start_time))
        elif (model == "gigaam"):
            gpu_queue.put((file_path, start_time))

def gibrit_process_folder(model, folder_path = "../../vishing/samples/Fraud"):
    folder = Path(folder_path)
    audio_files = list(folder.glob("*.wav"))
    task_queue = mp.Queue()      # для CPU-воркеров
    gpu_queue = mp.Queue()       # для GPU-воркера
    result_queue = mp.Queue()    # для финальных результатов
    for file in audio_files:
        task_queue.put(str(file))
    num_of_processes = max(1, mp.cpu_count() - 2)
    print(f"Запускаем {num_of_processes} процессов...")

    cpu_processes = []
    for _ in range(num_of_processes):
        process = mp.Process(target = cpu_worker, args = (task_queue, gpu_queue, result_queue, model))
        process.start()
        cpu_processes.append(process)

    gpu_process = mp.Process(target = gpu_worker, args = (gpu_queue, result_queue))
    gpu_process.start()

    for process in cpu_processes:
        process.join()
    gpu_queue.put(None)
    gpu_process.join()
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    return (results, num_of_processes)



def main():
    mode_of_execution = input("Введите режим анализа(cpu(многопоточный), gpu(на одном потоке), both):")
    model             = input("Введите модель для распознавания аудио(vosk, gigaam, both):")
    folder_path       = input("Введите название папки с выборкой:")
    results = []
    total_path = "../../vishing/" + folder_path
    if (mode_of_execution == "cpu"):
        results, num_of_processes = parallel_process_folder(mode_of_execution, model, folder_path = total_path)
    elif (mode_of_execution == "gpu"):
        results = one_process_folder_by_gpu(mode_of_execution, model, folder_path = total_path)
    elif (mode_of_execution == "both"):
        results, num_of_processes = gibrit_process_folder(model, folder_path = total_path)
    else:
        print("Введен неизвестный режим обработки!")
        exit(1)

    count_frauds = sum(1 for _, label, _ in results if label == 0)
    percentage_frauds = count_frauds / len(results) * 100
    print(f"Процент мошеннических разговоров в папке = {percentage_frauds}%")
    total_time = sum(elapsed for _, _, elapsed in results)
    avg_time   = total_time / len(results)
    max_file, _, max_time = max(results, key=lambda x: x[2])
    min_file, _, min_time = min(results, key=lambda x: x[2])
    if (mode_of_execution == "cpu" or mode_of_execution == "both"):
        print(f"Общее время анализа по всем процессам = {total_time:.2f}с")
        print(f"Реальное время анализа = {total_time / num_of_processes:.2f}с")
    elif (mode_of_execution == "gpu"):
        print(f"Общее время анализа выборки = {total_time:.2f}с")
    print(f"Среднее время на один файл = {avg_time:.2f}с")
    print(f"Самый долгий файл {max_file} : {max_time:.2f}с")
    print(f"Самый быстрый файл {min_file} : {min_time:.2f}с")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    folder_name = Path(total_path).name
    file_with_results = results_dir / f"{folder_name}_results.csv"
    with open(file_with_results, "w", newline= "", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Название файла", "label", "Время анализа"])
        writer.writerows(results)

    file_with_execute_statistics = results_dir / "statistics.txt"
    with open(file_with_execute_statistics, "a", newline="", encoding="utf-8") as file:
        file.write(f"Анализируемая выборка: {folder_path}\n")
        file.write(f"Режим анализа: {mode_of_execution}\n")
        file.write(f"Анализ производился с помощью модели(моделей): {model}\n")
        file.write(f"Процент мошеннических разговоров в папке = {percentage_frauds}%\n")
        if (mode_of_execution == "cpu" or mode_of_execution == "both"):
            file.write(f"Общее время анализа по всем процессам = {total_time:.2f}с\n")
            file.write(f"Реальное время анализа = {total_time / num_of_processes:.2f}с\n")
        elif (mode_of_execution == "gpu"):
            file.write(f"Общее время анализа выборки = {total_time:.2f}с\n")
        file.write(f"Среднее время на один файл = {avg_time:.2f}с\n")
        file.write(f"Самый долгий файл {max_file} : {max_time:.2f}с\n")
        file.write(f"Самый быстрый файл {min_file} : {min_time:.2f}с\n")
        file.write("\n\n")

if __name__ == "__main__":
    main()
