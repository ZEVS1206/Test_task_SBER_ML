from test_vosk_model import get_text_for_analyze_vosk_model
from test_gigaam_model import get_text_for_analyze_gigaam_model
from test_analyze_dictionary import call_analyze
import time

def main():
    model       = input("Введите модель для распознавания аудио(vosk, gigaam):")
    folder_name = input("Введите название папки:")
    file_name   = input("Введите название файла:")
    path_to_file = "../../vishing/" + folder_name + "/" + file_name
    text_for_analyze = ""
    t0 = time.perf_counter()
    if (model == "vosk"):
        text_for_analyze = get_text_for_analyze_vosk_model(path_to_file)
    elif (model == "gigaam"):
        text_for_analyze = get_text_for_analyze_gigaam_model(path_to_file)
    print("\n\n")
    verdict = call_analyze(case = 1, text_for_analyze = text_for_analyze)
    if (verdict > 0.7):
        verdict = 0
    else:
        verdict = 1
    elapsed = time.perf_counter() - t0
    print(verdict)
    print(f"Файл анализировался {elapsed}с")

if __name__ == "__main__":
    main()
