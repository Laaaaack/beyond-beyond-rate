# SlayerPytorch Environment Requirements

## Python & PyTorch
- **Python 3.11** (x64)
- **PyTorch** with CUDA support — must match your CUDA toolkit version (CUDA 11.8)

## CUDA
- **CUDA Toolkit 11.8** — installed at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`
- GPU with **sm_60 or higher** (Pascal/GTX 10xx or newer; RTX 2050 = sm_86 ✅)

## Compiler (the tricky part)
- **MSVC v143 toolset version 14.36** specifically
  - Installed via VS 2022 Build Tools → Individual Components
  - CUDA 11.8 rejects anything newer (14.37+)
- Must build from **x64 Native Tools Command Prompt for VS 2022** (not plain PowerShell, not x86)

## Build Tools
- **Visual Studio 2022 Build Tools** (version 17, not 2026/version 18)
- **Ninja** build system (`pip install ninja`)

## Environment Variables (set before every build)
```cmd
set DISTUTILS_USE_SDK=1
set MSSdk=1
```

## Summary Table

| Component | Required Version |
|---|---|
| Python | 3.11 x64 |
| CUDA Toolkit | 11.8 |
| PyTorch | CUDA 11.8 build |
| VS Build Tools | 2022 (v17) |
| MSVC toolset | 14.36 specifically |
| Command Prompt | x64 Native Tools for VS 2022 |
| Ninja | any recent version |

## Build Steps

1. Open **x64 Native Tools Command Prompt for VS 2022** from the Start menu
2. Set environment variables:
   ```cmd
   set DISTUTILS_USE_SDK=1
   set MSSdk=1
   ```
3. Navigate and activate venv:
   ```cmd
   cd D:\IC_2025\IRP\workspace\slayerPytorch-master
   D:\IC_2025\IRP\workspace\venv\Scripts\activate
   ```
4. Run the install:
   ```cmd
   python setup.py install
   ```

## Notes

> **The most fragile dependency** is the **MSVC 14.36 + CUDA 11.8** pairing — CUDA 11.8's version check hard-rejects anything newer than 14.36.

> The build path must show `win-amd64` (not `win32`) — if you see `win32`, you are in a 32-bit prompt and will get `size_t` redeclaration errors.
