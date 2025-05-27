import torch
import sys
import subprocess
import os

def get_nvidia_smi_output():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        return result.stdout
    except:
        return "nvidia-smi command failed. NVIDIA driver might not be installed properly."

def check_pytorch_cuda():
    print("\n" + "="*50)
    print("PYTORCH AND CUDA INFORMATION")
    print("="*50)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} capability: {torch.cuda.get_device_capability(i)}")
        
        # Test GPU with a simple operation
        print("\nTesting GPU with a simple tensor operation...")
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        z = x @ y  # Matrix multiplication
        end.record()
        
        # Waits for everything to finish
        torch.cuda.synchronize()
        
        print(f"Operation completed in {start.elapsed_time(end):.2f} ms")
        print("GPU is working correctly!")
    else:
        print("\nCUDA is not available. Possible reasons:")
        print("1. PyTorch was installed without CUDA support")
        print("2. NVIDIA drivers are not installed or outdated")
        print("3. Your GPU is not CUDA-compatible")
        print("4. CUDA toolkit is not installed or not in PATH")
    
    print("\n" + "="*50)
    print("NVIDIA-SMI OUTPUT")
    print("="*50)
    print(get_nvidia_smi_output())
    
    print("\n" + "="*50)
    print("INSTALLATION INSTRUCTIONS")
    print("="*50)
    print("To install PyTorch with CUDA support, run one of these commands:")
    print("\nFor CUDA 11.8:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\nFor CUDA 12.1:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
if __name__ == "__main__":
    check_pytorch_cuda()
