import torch

print("PyTorch Versiyonu:", torch.__version__)
print("CUDA Erişilebilir mi?:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Sayısı:", torch.cuda.device_count())
    print("Aktif GPU:", torch.cuda.get_device_name(0))
else:
    print("❌ Hala GPU görünmüyor!")