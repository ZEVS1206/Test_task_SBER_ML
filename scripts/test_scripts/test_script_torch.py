import torch

print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Имя устройства: {torch.cuda.get_device_name(0)}")
    print(f"Количество устройств: {torch.cuda.device_count()}")
