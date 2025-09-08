import torch, torchvision, platform
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print("gpu name:", torch.cuda.get_device_name(i))
    print("capability:", torch.cuda.get_device_capability(i))
