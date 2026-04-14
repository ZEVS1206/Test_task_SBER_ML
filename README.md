# Fraud Call Detection System

A two‑layer system for automatic recognition and classification of fraudulent phone calls based on audio recordings (WAV).
The first layer is fast filtering using **Vosk** + a weighted dictionary (CPU). The second layer is precise transcription with **GigaAM v3** (CPU/GPU) followed by final analysis (also dictionary). The use of the hybrid structure was motivated by improved system performance.
Multiprocessing support for speed on multi‑core CPUs. GPU-based tests were also performed.

---

## 🚀 Features

- **Hybrid architecture**
  - Lightweight layer (Vosk + dictionary) rejects obviously clean and clearly fraudulent calls.
  - Heavy layer (GigaAM + dictionary) analyses only borderline cases (≈20%).
- **High performance**
  - Multiprocess audio file processing.
  - Optional GPU acceleration for GigaAM (CUDA, TensorRT).
  - Cascade filtering reduces GPU load.
- **Support for different input types**
  - Audio (WAV, mono, 16 kHz) – batch folder processing.
  - Text (JSON) – fast dictionary / classifier testing.
- **Flexible configuration**
  - Weighted dictionary of fraud phrases (JSON).
  - Opportunities for further development of the project to train your own or use an opensource model to improve recognition accuracy.

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/ZEVS1206/Test_task_SBER_ML
cd Test_task_SBER_ML
```

### 2. Create a virtual environment
To avoid **conflicts when working with libraries, a virtual environment is used.
```bash
python3 -m venv vosk_env
source vosk_env/bin/activate   # Linux/Mac
# or vosk_env\Scripts\activate for Windows
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Main libraries:
`vosk`, `onnx-asr`, `librosa`, `torch`, `transformers`, `numpy`, `pandas`, `multiprocessing`.

### 4. Download models
- **Vosk** (Russian model):
  ```bash
  wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip
  unzip vosk-model-small-ru-0.22.zip -d models/
  ```
- **GigaAM v3** – will be downloaded automatically on first run via `onnx-asr`.

### 5. (Optional) Set up GPU for GigaAM
```bash
pip install onnx-asr[gpu,hub]
```
Make sure NVIDIA drivers, CUDA Toolkit and cuDNN are installed.
For Optimus laptops, use the wrapper script `run_on_nvidia.sh` (see "Usage"), because I used it for testing model. **I note** that the system has shown good efficiency in multithreaded CPU mode, however, to improve performance, as well as possibly for other tests, it may be necessary

---

## 🧪 Usage
The program has a modular structure and supports several modes of analysis and resource usage.
### Basic example – process an audio folder
In order to process a folder with `.wav` audio files (be sure to check that the files have this extension), there are several ways. First, if you want to use a pure GPU or hybrid, then you need to run the program using a graphics card. And so, the system supports the analysis of two different types of incoming data: **audio** and **text**. Let's deal with the audio now.
You can run the main script using:
```bash
python3 test_main_parallel_execute.py
```
Or if you want to use GPU(if you have temprorary sheme):
```bash
your_script_for_gpu.sh python3 test_main_parallel_execute.py
```
Then, you need to input *type of data*: *audio* or *text*:
```bash
Введите тип анализируемой выборки(text, audio):audio
```

Then, enter *mode of execution*: *cpu*, *gpu* or *both*:
```bash
Введите режим анализа(cpu(многопоточный), gpu(на одном потоке), both):cpu
```

Then, input *model* for transcript:*vosk*, *gigaam* or *both*:
```bash
Введите модель для распознавания аудио(vosk, gigaam, both):vosk
```

Then, you need to enter path to folder with your selection.⚠️ You need to add folder at folder `vishing/`!
```bash
Введите название папки с выборкой:samples/Fraud
```
After that, you will in some time get results in `results/` as CSV with columns: `Filename; label` (0 – fraudulent, 1 – clean).


### Test on text JSON
Also add your test_file to `vishing/` and
```bash
Введите тип анализируемой выборки(text, audio):text
Введите режим анализа(cpu(многопоточный), gpu(на одном потоке), both):you can write anything, cpu will process programm.
Введите адрес файла с текстовыми данными относительно папки vishing:text_data/test_phone_conversations_v1.json
```
And you also will get results in the same folder.

---

## 🧠 Architecture

```
Audio files (WAV)
        │
        ▼
┌───────────────────┐
│  Vosk + VAD       │  (fast, CPU)
│  rough transcript │
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────────┐
│  Lightweight classifier       │
│  (weighted dictionary)        │
└─────────┬─────────────────────┘
          │
    ┌─────┴─────┐
    │           │
  score<0.2   score>0.7  0.2 ≤ score ≤ 0.7
    │           │              │
    ▼           ▼              ▼
 label=1     label=0    ┌───────────────────┐
 (clean)    (fraud)     │  GigaAM + VAD     │
                        │  precise          │
                        │  transcription    │
                        └─────────┬─────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │  Heavy classifier │
                        │  (also dictionary |
                        |   and model in    |
                        |   future)         │
                        └─────────┬─────────┘
                                  │
                                  ▼
                            final label
```

---

## ⚙️ Configuring the fraud phrase dictionary

File `config/fraud_keywords.json` structure:
```json
    {
      "text": "биометрия для подтверждения",
      "weight": 0.8,
      "category": "data_request"
    },
    {
      "text": "никому не говорите",
      "weight": 0.7,
      "category": "isolation"
    }
```
You can edit weights and add new phrases. The lightweight classifier sums the weights of found phrases and normalises the result to [0,1].

---

## 📈 Performance

- **Vosk + dictionary** (CPU, 1 thread): ≈ per 40 sec on file, but low accuracy
- **GigaAM + dictionary** (CPU): ≈ 1 - 2 min per file, accuracy higher
- **GigaAM + dictionary** (GPU, NVIDIA RTX 4050 Mobile): ≈ 5 seconds per file, very fast, but worse than a hybrid
- **Multiprocessing** (16 CPU workers(ONLY) + both models): 77 sec per file and 140 sec common time.
- **Hybrid**(18 CPU workers + 1 common GPU worker + both models): 49 sec per file and 82 sec common time. The best result.

I would like to note that tests were also conducted on the application of the hybrid strategy, in which all files were sent to the analyzer in turn in one stream, the results can be seen in the `results/` in the file `statistics.txt` . The final speed turned out to be **lower** than in the case of the improved hybrid, but **higher** than the usual CPU.


## ⚠️ Important details

### Further development

At the moment, I have managed to acquire significant datasets of telephone conversations for further training of the model. In the future, it is planned to add a well-trained model capable of recognizing operator and user phrases, emotional background, and context separately as a more accurate classifier.

### Where are datasets

Due to a **commercial license**, I have not made the main datasets publicly available.

---

## 📄 License

This project is distributed under the MIT License.
Third‑party models:
- Vosk – Apache 2.0
- GigaAM v3 – MIT
- Silero VAD – MIT

---

## 🤝 Contact

Author: [Zevs](https://github.com/ZEVS1206)

Email: egor_bud5@inbox.ru


