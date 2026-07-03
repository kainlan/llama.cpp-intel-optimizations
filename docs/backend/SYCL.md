# llama.cpp for SYCL

- [Background](#background)
- [Recommended Release](#recommended-release)
- [News](#news)
- [OS](#os)
- [Hardware](#hardware)
- [Docker](#docker)
- [Linux](#linux)
- [Windows](#windows)
- [Environment Variable](#environment-variable)
- [Memory placement](#memory-placement)
- [Cross-device KV placement contract](#cross-device-kv-placement-contract)
- [Known Issue](#known-issues)
- [Q&A](#qa)
- [TODO](#todo)

## Background

**SYCL** is a high-level parallel programming model designed to improve developers productivity writing code across various hardware accelerators such as CPUs, GPUs, and FPGAs. It is a single-source language designed for heterogeneous computing and based on standard C++17.

**oneAPI** is an open ecosystem and a standard-based specification, supporting multiple architectures including but not limited to Intel CPUs, GPUs and FPGAs. The key components of the oneAPI ecosystem include:

- **DPCPP** *(Data Parallel C++)*: The primary oneAPI SYCL implementation, which includes the icpx/icx Compilers.
- **oneAPI Libraries**: A set of highly optimized libraries targeting multiple domains *(e.g. Intel oneMKL, oneMath and oneDNN)*.
- **oneAPI LevelZero**: A high performance low level interface for fine-grained control over Intel iGPUs and dGPUs.
- **Nvidia & AMD Plugins**: These are plugins extending oneAPI's DPCPP support to SYCL on Nvidia and AMD GPU targets.

### Llama.cpp + SYCL

The llama.cpp SYCL backend is primarily designed for **Intel GPUs**.
SYCL cross-platform capabilities enable support for Nvidia GPUs as well, with limited support for AMD.

## Recommended Release

The following releases are verified and recommended:

|Commit ID|Tag|Release|Verified  Platform| Update date|
|-|-|-|-|-|
|24e86cae7219b0f3ede1d5abdf5bf3ad515cccb8|b5377 |[llama-b5377-bin-win-sycl-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b5377/llama-b5377-bin-win-sycl-x64.zip) |ArcB580/Linux/oneAPI 2025.1<br>LNL Arc GPU/Windows 11/oneAPI 2025.1.1|2025-05-15|
|3bcd40b3c593d14261fb2abfabad3c0fb5b9e318|b4040 |[llama-b4040-bin-win-sycl-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b4040/llama-b4040-bin-win-sycl-x64.zip) |Arc770/Linux/oneAPI 2024.1<br>MTL Arc GPU/Windows 11/oneAPI 2024.1| 2024-11-19|
|fb76ec31a9914b7761c1727303ab30380fd4f05c|b3038 |[llama-b3038-bin-win-sycl-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b3038/llama-b3038-bin-win-sycl-x64.zip) |Arc770/Linux/oneAPI 2024.1<br>MTL Arc GPU/Windows 11/oneAPI 2024.1||


## News

- 2025.11
  - Support malloc memory on device more than 4GB.

- 2025.2
  - Optimize MUL_MAT Q4_0 on Intel GPU for all dGPUs and built-in GPUs since MTL. Increase the performance of LLM (llama-2-7b.Q4_0.gguf) 21%-87% on Intel GPUs (MTL, ARL-H, Arc, Flex, PVC).
    |GPU|Base tokens/s|Increased tokens/s|Percent|
    |-|-|-|-|
    |PVC 1550|39|73|+87%|
    |Flex 170|39|50|+28%|
    |Arc770|42|55|+30%|
    |MTL|13|16|+23%|
    |ARL-H|14|17|+21%|

- 2024.11
  - Use syclcompat to improve the performance on some platforms. This requires to use oneAPI 2025.0 or newer (tested with 2025.3.1).

- 2024.8
  - Use oneDNN as the default GEMM library, improve the compatibility for new Intel GPUs.

- 2024.5
  - Performance is increased: 34 -> 37 tokens/s of llama-2-7b.Q4_0 on Arc770.
  - Arch Linux is verified successfully.

- 2024.4
  - Support data types: GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M.

- 2024.3
  - Release binary files of Windows.
  - A blog is published: **Run LLM on all Intel GPUs Using llama.cpp**: [intel.com](https://www.intel.com/content/www/us/en/developer/articles/technical/run-llm-on-all-gpus-using-llama-cpp-artical.html) or [medium.com](https://medium.com/@jianyu_neo/run-llm-on-all-intel-gpus-using-llama-cpp-fd2e2dcbd9bd).
  - New base line is ready: [tag b2437](https://github.com/ggml-org/llama.cpp/tree/b2437).
  - Support multiple cards: **--split-mode**: [none|layer]; not support [row], it's on developing.
  - Support to assign main GPU by **--main-gpu**, replace $GGML_SYCL_DEVICE.
  - Support detecting all GPUs with level-zero and same top **Max compute units**.
  - Support OPs
    - hardsigmoid
    - hardswish
    - pool2d

- 2024.1
  - Create SYCL backend for Intel GPU.
  - Support Windows build

## OS

| OS      | Status  | Verified                                       |
|---------|---------|------------------------------------------------|
| Linux   | Support | Ubuntu 22.04, Fedora Silverblue 39, Arch Linux |
| Windows | Support | Windows 11                                     |


## Hardware

### Intel GPU

SYCL backend supports Intel GPU Family:

- Intel Data Center Max Series
- Intel Flex Series, Arc Series
- Intel Built-in Arc GPU
- Intel iGPU in Core CPU (11th Generation Core CPU and newer, refer to [oneAPI supported GPU](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-base-toolkit-system-requirements.html#inpage-nav-1-1)).

On older Intel GPUs, you may try [OpenCL](/docs/backend/OPENCL.md) although the performance is not optimal, and some GPUs may not support OpenCL nor have any GPGPU capabilities.

#### Verified devices

| Intel GPU                     | Status  | Verified Model                        |
|-------------------------------|---------|---------------------------------------|
| Intel Data Center Max Series  | Support | Max 1550, 1100                        |
| Intel Data Center Flex Series | Support | Flex 170                              |
| Intel Arc Series              | Support | Arc 770, 730M, Arc A750, B580         |
| Intel built-in Arc GPU        | Support | built-in Arc GPU in Meteor Lake, Arrow Lake, Lunar Lake |
| Intel iGPU                    | Support | iGPU in 13700k, 13400, i5-1250P, i7-1260P, i7-1165G7  |

*Notes:*

- **Memory**
  - The device memory is a limitation when running a large model. The loaded model size, *`llm_load_tensors: buffer_size`*, is displayed in the log when running `./bin/llama-cli`.
  - Please make sure the GPU shared memory from the host is large enough to account for the model's size. For e.g. the *llama-2-7b.Q4_0* requires at least 8.0GB for integrated GPU and 4.0GB for discrete GPU.

- **Execution Unit (EU)**
  - If the iGPU has less than 80 EUs, the inference speed will likely be too slow for practical use.

### Other Vendor GPU

**Verified devices**

| Nvidia GPU               | Status    | Verified Model |
|--------------------------|-----------|----------------|
| Ampere Series            | Supported | A100, A4000    |
| Ampere Series *(Mobile)* | Supported | RTX 40 Series  |

| AMD GPU                  | Status       | Verified Model |
|--------------------------|--------------|----------------|
| Radeon Pro               | Experimental | W6800          |
| Radeon RX                | Experimental | 6700 XT        |

Note: AMD GPU support is highly experimental and is incompatible with F16.
Additionally, it only supports GPUs with a sub_group_size (warp size) of 32.

## Docker

The docker build option is currently limited to *Intel GPU* targets.

### Build image

```sh
# Using FP16
docker build -t llama-cpp-sycl --build-arg="GGML_SYCL_F16=ON" --target light -f .devops/intel.Dockerfile .

# Using FP32
docker build -t llama-cpp-sycl --build-arg="GGML_SYCL_F16=OFF" --target light -f .devops/intel.Dockerfile .
```

*Notes*:

You can also use the `.devops/llama-server-intel.Dockerfile`, which builds the *"server"* alternative.
Check the [documentation for Docker](../docker.md) to see the available images.

### Run container

```sh
# First, find all the DRI cards
ls -la /dev/dri
# Then, pick the card that you want to use (here for e.g. /dev/dri/card1).
docker run -it --rm -v "/path/to/models:/models" --device /dev/dri/renderD128:/dev/dri/renderD128 --device /dev/dri/card0:/dev/dri/card0 llama-cpp-sycl -m /models/7B/ggml-model-q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -c 4096 -s 0
```

*Notes:*
- Docker has been tested successfully on native Linux. WSL support has not been verified yet.
- You may need to install Intel GPU driver on the **host** machine *(Please refer to the [Linux configuration](#linux) for details)*.

## Linux

### I. Setup Environment

1. **Install GPU drivers**

  - **Intel GPU**

Intel data center GPUs drivers installation guide and download page can be found here: [Get intel dGPU Drivers](https://dgpu-docs.intel.com/driver/installation.html#ubuntu-install-steps).

*Note*: for client GPUs *(iGPU & Arc A-Series)*, please refer to the [client iGPU driver installation](https://dgpu-docs.intel.com/driver/client/overview.html).

Once installed, add the user(s) to the `video` and `render` groups.

```sh
sudo usermod -aG render $USER
sudo usermod -aG video $USER
```

*Note*: logout/re-login for the changes to take effect.

Verify installation through `clinfo`:

```sh
sudo apt install clinfo
sudo clinfo -l
```

Sample output:

```sh
Platform #0: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) Arc(TM) A770 Graphics

Platform #0: Intel(R) OpenCL HD Graphics
 `-- Device #0: Intel(R) Iris(R) Xe Graphics [0x9a49]
```

- **Nvidia GPU**

In order to target Nvidia GPUs through SYCL, please make sure the CUDA/CUBLAS native requirements *-found [here](README.md#cuda)-* are installed.

- **AMD GPU**

To target AMD GPUs with SYCL, the ROCm stack must be installed first.

2. **Install Intel® oneAPI Base toolkit**

SYCL backend depends on:
  - Intel® oneAPI DPC++/C++ compiler/running-time.
  - Intel® oneAPI DPC++/C++ library (oneDPL).
  - Intel® oneAPI Deep Neural Network Library (oneDNN).
  - Intel® oneAPI Math Kernel Library (oneMKL).

- **For Intel GPU**

All above are included in both **Intel® oneAPI Base toolkit** and **Intel® Deep Learning Essentials** packages.

It's recommended to install **Intel® Deep Learning Essentials** which only provides the necessary libraries with less size.

The **Intel® oneAPI Base toolkit** and **Intel® Deep Learning Essentials** can be obtained from the official [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) page.

Please follow the instructions for downloading and installing the Toolkit for Linux, and preferably keep the default installation values unchanged, notably the installation path *(`/opt/intel/oneapi` by default)*.

Following guidelines/code snippets assume the default installation values. Otherwise, please make sure the necessary changes are reflected where applicable.

Upon a successful installation, SYCL is enabled for the available intel devices, along with relevant libraries such as oneAPI oneDNN for Intel GPUs.

|Verified release|
|-|
|2025.3.1|
|2025.2.1|
|2025.1|
|2024.1|

- **Adding support to Nvidia GPUs**

**oneAPI Plugin**: In order to enable SYCL support on Nvidia GPUs, please install the [Codeplay oneAPI Plugin for Nvidia GPUs](https://developer.codeplay.com/products/oneapi/nvidia/download). User should also make sure the plugin version matches the installed base toolkit one *(previous step)* for a seamless "oneAPI on Nvidia GPU" setup.

**oneDNN**: The current oneDNN releases *(shipped with the oneAPI base-toolkit)* do not include the NVIDIA backend. Therefore, oneDNN must be compiled from source to enable the NVIDIA target:

```sh
git clone https://github.com/oneapi-src/oneDNN.git
cd oneDNN
cmake -GNinja -Bbuild-nvidia -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP -DDNNL_GPU_VENDOR=NVIDIA -DONEDNN_BUILD_GRAPH=OFF -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build-nvidia --config Release
```

- **Adding support to AMD GPUs**

**oneAPI Plugin**: In order to enable SYCL support on AMD GPUs, please install the [Codeplay oneAPI Plugin for AMD GPUs](https://developer.codeplay.com/products/oneapi/amd/download). As with Nvidia GPUs, the user should also make sure the plugin version matches the installed base toolkit.

3. **Verify installation and environment**

In order to check the available SYCL devices on the machine, please use the `sycl-ls` command.
```sh
source /opt/intel/oneapi/setvars.sh
sycl-ls
```

- **Intel GPU**

When targeting an intel GPU, the user should expect one or more devices among the available SYCL devices. Please make sure that at least one GPU is present via `sycl-ls`, for instance `[level_zero:gpu]` in the sample output below:

```
[level_zero:gpu][level_zero:0] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) Arc(TM) A770 Graphics 12.55.8 [1.3.29735+27]
[level_zero:gpu][level_zero:1] Intel(R) oneAPI Unified Runtime over Level-Zero, Intel(R) UHD Graphics 730 12.2.0 [1.3.29735+27]
[opencl:cpu][opencl:0] Intel(R) OpenCL, 13th Gen Intel(R) Core(TM) i5-13400 OpenCL 3.0 (Build 0) [2025.20.8.0.06_160000]
[opencl:gpu][opencl:1] Intel(R) OpenCL Graphics, Intel(R) Arc(TM) A770 Graphics OpenCL 3.0 NEO  [24.39.31294]
[opencl:gpu][opencl:2] Intel(R) OpenCL Graphics, Intel(R) UHD Graphics 730 OpenCL 3.0 NEO  [24.39.31294]
```

- **Nvidia GPU**

Similarly, user targeting Nvidia GPUs should expect at least one SYCL-CUDA device [`cuda:gpu`] as below:

```
[opencl:acc][opencl:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.12.0.12_195853.xmain-hotfix]
[opencl:cpu][opencl:1] Intel(R) OpenCL, Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz OpenCL 3.0 (Build 0) [2023.16.12.0.12_195853.xmain-hotfix]
[cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA A100-PCIE-40GB 8.0 [CUDA 12.5]
```

- **AMD GPU**

For AMD GPUs we should expect at least one SYCL-HIP device [`hip:gpu`]:

```
[opencl:cpu][opencl:0] Intel(R) OpenCL, 12th Gen Intel(R) Core(TM) i9-12900K OpenCL 3.0 (Build 0) [2024.18.6.0.02_160000]
[hip:gpu][hip:0] AMD HIP BACKEND, AMD Radeon PRO W6800 gfx1030 [HIP 60140.9]
```

### II. Build llama.cpp

#### Intel GPU

```sh
./examples/sycl/build.sh
```

or

```sh
# Export relevant ENV variables
source /opt/intel/oneapi/setvars.sh

# Option 1: Use FP32 (recommended for better performance in most cases)
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# Option 2: Use FP16
cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON

# build all binary
cmake --build build --config Release -j -v
```

It is possible to come across some precision issues when running tests that stem from using faster
instructions, which can be circumvented by setting the environment variable `SYCL_PROGRAM_COMPILE_OPTIONS`
as `-cl-fp32-correctly-rounded-divide-sqrt`

#### Nvidia GPU

The SYCL backend depends on [oneMath](https://github.com/uxlfoundation/oneMath) for Nvidia and AMD devices.
By default it is automatically built along with the project. A specific build can be provided by setting the CMake flag `-DoneMath_DIR=/path/to/oneMath/install/lib/cmake/oneMath`.

```sh
# Build LLAMA with Nvidia BLAS acceleration through SYCL
# Setting GGML_SYCL_DEVICE_ARCH is optional but can improve performance
GGML_SYCL_DEVICE_ARCH=sm_80 # Example architecture

# Option 1: Use FP32 (recommended for better performance in most cases)
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=NVIDIA -DGGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH} -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DDNNL_DIR=/path/to/oneDNN/build-nvidia/install/lib/cmake/dnnl

# Option 2: Use FP16
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=NVIDIA -DGGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH} -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DDNNL_DIR=/path/to/oneDNN/build-nvidia/install/lib/cmake/dnnl

# build all binary
cmake --build build --config Release -j -v
```

It is possible to come across some precision issues when running tests that stem from using faster
instructions, which can be circumvented by passing the `-fno-fast-math` flag to the compiler.

#### AMD GPU

The SYCL backend depends on [oneMath](https://github.com/uxlfoundation/oneMath) for Nvidia and AMD devices.
By default it is automatically built along with the project. A specific build can be provided by setting the CMake flag `-DoneMath_DIR=/path/to/oneMath/install/lib/cmake/oneMath`.

```sh
# Build LLAMA with rocBLAS acceleration through SYCL

## AMD
# Use FP32, FP16 is not supported
# Find your GGML_SYCL_DEVICE_ARCH with rocminfo, under the key 'Name:'
GGML_SYCL_DEVICE_ARCH=gfx90a # Example architecture
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_TARGET=AMD -DGGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH} -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx

# build all binary
cmake --build build --config Release -j -v
```

### III. Run the inference

#### Retrieve and prepare model

You can refer to the general [*Prepare and Quantize*](README.md#prepare-and-quantize) guide for model preparation, or download an already quantized model like [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf?download=true) or [Meta-Llama-3-8B-Instruct-Q4_0.gguf](https://huggingface.co/aptha/Meta-Llama-3-8B-Instruct-Q4_0-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf).

##### Check device

1. Enable oneAPI running environment

```sh
source /opt/intel/oneapi/setvars.sh
```

2. List devices information

Similar to the native `sycl-ls`, available SYCL devices can be queried as follow:

```sh
./build/bin/llama-ls-sycl-device
```

This command will only display the selected backend that is supported by SYCL. The default backend is level_zero. For example, in a system with 2 *intel GPU* it would look like the following:
```
found 2 SYCL devices:

|  |                  |                                             |Compute   |Max compute|Max work|Max sub|               |
|ID|       Device Type|                                         Name|capability|units      |group   |group  |Global mem size|
|--|------------------|---------------------------------------------|----------|-----------|--------|-------|---------------|
| 0|[level_zero:gpu:0]|               Intel(R) Arc(TM) A770 Graphics|       1.3|        512|    1024|     32|    16225243136|
| 1|[level_zero:gpu:1]|                    Intel(R) UHD Graphics 770|       1.3|         32|     512|     32|    53651849216|
```

#### Choose level-zero devices

`ONEAPI_DEVICE_SELECTOR` uses the `backend:devices` syntax. The device list is
comma-separated; semicolons separate independent filters/backends. The
`[level_zero:gpu:0]` strings printed by `llama-ls-sycl-device` are display IDs,
not valid `ONEAPI_DEVICE_SELECTOR` values.

|Chosen Device ID|Setting|
|-|-|
|0|`export ONEAPI_DEVICE_SELECTOR="level_zero:0"` or no action|
|1|`export ONEAPI_DEVICE_SELECTOR="level_zero:1"`|
|All level-zero GPUs|`export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"`|
|0 & 1|`export ONEAPI_DEVICE_SELECTOR="level_zero:0,1"`|

#### Execute

Choose one of following methods to run.

1. Script

- Use device 0:

```sh
./examples/sycl/run-llama2.sh 0
# OR
./examples/sycl/run-llama3.sh 0
```
- Use multiple devices:

```sh
./examples/sycl/run-llama2.sh
# OR
./examples/sycl/run-llama3.sh
```

2. Command line
Launch inference

There are two device selection modes:

- Single device: Use one device assigned by user. Default device id is 0.
- Multiple devices: Automatically choose the devices with the same backend.

In two device selection modes, the default SYCL backend is level_zero, you can choose other backend supported by SYCL by setting environment variable ONEAPI_DEVICE_SELECTOR.

| Device selection | Parameter                              |
|------------------|----------------------------------------|
| Single device    | --split-mode none --main-gpu DEVICE_ID |
| Multiple devices | --split-mode layer (default)           |

Examples:

- Use device 0:

```sh
ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -no-cnv -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 99 -sm none -mg 0
```

- Use multiple devices:

```sh
ZES_ENABLE_SYSMAN=1 ./build/bin/llama-cli -no-cnv -m models/llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 99 -sm layer
```

*Notes:*

- Upon execution, verify the selected device(s) ID(s) in the output log, which can for instance be displayed as follow:

```sh
detect 1 SYCL GPUs: [0] with top Max compute units:512
```
Or
```sh
use 1 SYCL GPUs: [0] with Max compute units:512
```

## Windows

### I. Setup Environment

1. Install GPU driver

Intel GPU drivers instructions guide and download page can be found here: [Get Intel GPU Drivers](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/software/drivers.html).

2. Install Visual Studio

If you already have a recent version of Microsoft Visual Studio, you can skip this step. Otherwise, please refer to the official download page for [Microsoft Visual Studio](https://visualstudio.microsoft.com/).

3. Install Intel® oneAPI Base toolkit

SYCL backend depends on:
  - Intel® oneAPI DPC++/C++ compiler/running-time.
  - Intel® oneAPI DPC++/C++ library (oneDPL).
  - Intel® oneAPI Deep Neural Network Library (oneDNN).
  - Intel® oneAPI Math Kernel Library (oneMKL).

All above are included in both **Intel® oneAPI Base toolkit** and **Intel® Deep Learning Essentials** packages.

It's recommended to install **Intel® Deep Learning Essentials** which only provides the necessary libraries with less size.

The **Intel® oneAPI Base toolkit** and **Intel® Deep Learning Essentials** can be obtained from the official [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) page.

Please follow the instructions for downloading and installing the Toolkit for Windows, and preferably keep the default installation values unchanged, notably the installation path *(`C:\Program Files (x86)\Intel\oneAPI` by default)*.

Following guidelines/code snippets assume the default installation values. Otherwise, please make sure the necessary changes are reflected where applicable.

b. Enable oneAPI running environment:

- Type "oneAPI" in the search bar, then open the `Intel oneAPI command prompt for Intel 64 for Visual Studio 2022` App.

- On the command prompt, enable the runtime environment with the following:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

- if you are using Powershell, enable the runtime environment with the following:

```
cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'
```

c. Verify installation

In the oneAPI command line, run the following to print the available SYCL devices:

```
sycl-ls.exe
```

There should be one or more *level-zero* GPU devices displayed as **[ext_oneapi_level_zero:gpu]**. Below is example of such output detecting an *intel Iris Xe* GPU as a Level-zero SYCL device:

Output (example):
```
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM), Intel(R) FPGA Emulation Device OpenCL 1.2  [2023.16.10.0.17_160000]
[opencl:cpu:1] Intel(R) OpenCL, 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz OpenCL 3.0 (Build 0) [2023.16.10.0.17_160000]
[opencl:gpu:2] Intel(R) OpenCL Graphics, Intel(R) Iris(R) Xe Graphics OpenCL 3.0 NEO  [31.0.101.5186]
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) Iris(R) Xe Graphics 1.3 [1.3.28044]
```

4. Install build tools

a. Download & install cmake for Windows: https://cmake.org/download/ (CMake can also be installed from Visual Studio Installer)
b. The new Visual Studio will install Ninja as default. (If not, please install it manually: https://ninja-build.org/)


### II. Build llama.cpp

You could download the release package for Windows directly, which including binary files and depended oneAPI dll files.

Choose one of following methods to build from source code.

#### 1. Script

```sh
.\examples\sycl\win-build-sycl.bat
```

#### 2. CMake

On the oneAPI command line window, step into the llama.cpp main directory and run the following:

```
@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force

# Option 1: Use FP32 (recommended for better performance in most cases)
cmake -B build -G "Ninja" -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx  -DCMAKE_BUILD_TYPE=Release

# Option 2: Or FP16
cmake -B build -G "Ninja" -DGGML_SYCL=ON -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=icx  -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL_F16=ON

cmake --build build --config Release -j
```

Or, use CMake presets to build:

```sh
cmake --preset x64-windows-sycl-release
cmake --build build-x64-windows-sycl-release -j --target llama-cli

cmake -DGGML_SYCL_F16=ON --preset x64-windows-sycl-release
cmake --build build-x64-windows-sycl-release -j --target llama-cli

cmake --preset x64-windows-sycl-debug
cmake --build build-x64-windows-sycl-debug -j --target llama-cli
```

#### 3. Visual Studio

You have two options to use Visual Studio to build llama.cpp:
- As CMake Project using CMake presets.
- Creating a Visual Studio solution to handle the project.

**Note**:

All following commands are executed in PowerShell.

##### - Open as a CMake Project

You can use Visual Studio to open the `llama.cpp` folder directly as a CMake project. Before compiling, select one of the SYCL CMake presets:

- `x64-windows-sycl-release`

- `x64-windows-sycl-debug`

*Notes:*
- For a minimal experimental setup, you can build only the inference executable using:

    ```Powershell
    cmake --build build --config Release -j --target llama-cli
    ```

##### - Generating a Visual Studio Solution

You can use Visual Studio solution to build and work on llama.cpp on Windows. You need to convert the CMake Project into a `.sln` file.

If you want to use the Intel C++ Compiler for the entire `llama.cpp` project, run the following command:

```Powershell
cmake -B build -G "Visual Studio 17 2022" -T "Intel C++ Compiler 2025" -A x64 -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release
```

If you prefer to use the Intel C++ Compiler only for `ggml-sycl`, ensure that `ggml` and its backend libraries are built as shared libraries ( i.e. `-DBUILD_SHARED_LIBRARIES=ON`, this is default behaviour):

```Powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_SYCL=ON -DCMAKE_BUILD_TYPE=Release \
      -DSYCL_INCLUDE_DIR="C:\Program Files (x86)\Intel\oneAPI\compiler\latest\include" \
      -DSYCL_LIBRARY_DIR="C:\Program Files (x86)\Intel\oneAPI\compiler\latest\lib"
```

If successful the build files have been written to: *path/to/llama.cpp/build*
Open the project file **build/llama.cpp.sln** with Visual Studio.

Once the Visual Studio solution is created, follow these steps:

1. Open the solution in Visual Studio.

2. Right-click on `ggml-sycl` and select **Properties**.

3. In the left column, expand **C/C++** and select **DPC++**.

4. In the right panel, find **Enable SYCL Offload** and set it to `Yes`.

5. Apply the changes and save.


*Navigation Path:*

```
Properties -> C/C++ -> DPC++ -> Enable SYCL Offload (Yes)
```

Now, you can build `llama.cpp` with the SYCL backend as a Visual Studio project.
To do it from menu: `Build -> Build Solution`.
Once it is completed, final results will be in **build/Release/bin**

*Additional Note*

- You can avoid specifying `SYCL_INCLUDE_DIR` and `SYCL_LIBRARY_DIR` in the CMake command by setting the environment variables:

    - `SYCL_INCLUDE_DIR_HINT`

    - `SYCL_LIBRARY_DIR_HINT`

- Above instruction has been tested with Visual Studio 17 Community edition and oneAPI 2025.3.1. We expect them to work also with future version if the instructions are adapted accordingly.

### III. Run the inference

#### Retrieve and prepare model

You can refer to the general [*Prepare and Quantize*](README.md#prepare-and-quantize) guide for model preparation, or download an already quantized model like [llama-2-7b.Q4_0.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_0.gguf) or [Meta-Llama-3-8B-Instruct-Q4_0.gguf](https://huggingface.co/aptha/Meta-Llama-3-8B-Instruct-Q4_0-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf).

##### Check device

1. Enable oneAPI running environment

On the oneAPI command line window, run the following and step into the llama.cpp directory:
```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
```

2. List devices information

Similar to the native `sycl-ls`, available SYCL devices can be queried as follow:

```
build\bin\llama-ls-sycl-device.exe
```

This command will only display the selected backend that is supported by SYCL. The default backend is level_zero. For example, in a system with 2 *Intel GPU* it would look like the following:
```
found 2 SYCL devices:
|  |                  |                                             |Compute   |Max compute|Max work|Max sub|               |
|ID|       Device Type|                                         Name|capability|units      |group   |group  |Global mem size|
|--|------------------|---------------------------------------------|----------|-----------|--------|-------|---------------|
| 0|[level_zero:gpu:0]|               Intel(R) Arc(TM) A770 Graphics|       1.3|        512|    1024|     32|    16225243136|
| 1|[level_zero:gpu:1]|                    Intel(R) UHD Graphics 770|       1.3|         32|     512|     32|    53651849216|

```

#### Choose level-zero devices

`ONEAPI_DEVICE_SELECTOR` uses the `backend:devices` syntax. The device list is
comma-separated; semicolons separate independent filters/backends. The
`[level_zero:gpu:0]` strings printed by `llama-ls-sycl-device` are display IDs,
not valid `ONEAPI_DEVICE_SELECTOR` values.

|Chosen Device ID|Setting|
|-|-|
|0|Default option. You may also want to `set ONEAPI_DEVICE_SELECTOR="level_zero:0"`|
|1|`set ONEAPI_DEVICE_SELECTOR="level_zero:1"`|
|All level-zero GPUs|`set ONEAPI_DEVICE_SELECTOR="level_zero:gpu"`|
|0 & 1|`set ONEAPI_DEVICE_SELECTOR="level_zero:0,1"`|

#### Execute

Choose one of following methods to run.

1. Script

```
examples\sycl\win-run-llama-2.bat
```

or

```
examples\sycl\win-run-llama-3.bat
```

2. Command line

Launch inference

There are two device selection modes:

- Single device: Use one device assigned by user. Default device id is 0.
- Multiple devices: Automatically choose the devices with the same backend.

In two device selection modes, the default SYCL backend is level_zero, you can choose other backend supported by SYCL by setting environment variable ONEAPI_DEVICE_SELECTOR.

| Device selection | Parameter                              |
|------------------|----------------------------------------|
| Single device    | --split-mode none --main-gpu DEVICE_ID |
| Multiple devices | --split-mode layer (default)           |

Examples:

- Use device 0:

```
build\bin\llama-cli.exe -no-cnv -m models\llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 99 -sm none -mg 0
```

- Use multiple devices:

```
build\bin\llama-cli.exe -no-cnv -m models\llama-2-7b.Q4_0.gguf -p "Building a website can be done in 10 simple steps:\nStep 1:" -n 400 -e -ngl 99 -sm layer
```


Note:

- Upon execution, verify the selected device(s) ID(s) in the output log, which can for instance be displayed as follow:

```sh
detect 1 SYCL GPUs: [0] with top Max compute units:512
```

Or

```sh
use 1 SYCL GPUs: [0] with Max compute units:512
```


## Environment Variable

#### Build

| Name               | Value                                 | Function                                    |
|--------------------|---------------------------------------|---------------------------------------------|
| GGML_SYCL          | ON (mandatory)                        | Enable build with SYCL code path.           |
| GGML_SYCL_TARGET   | INTEL *(default)* \| NVIDIA \| AMD    | Set the SYCL target device type.            |
| GGML_SYCL_DEVICE_ARCH | Optional (except for AMD)             | Set the SYCL device architecture, optional except for AMD. Setting the device architecture can improve the performance. See the table [--offload-arch](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/OffloadDesign.md#--offload-arch) for a list of valid architectures. |
| GGML_SYCL_F16      | OFF *(default)* \|ON *(optional)*     | Enable FP16 build with SYCL code path. (1.) |
| GGML_SYCL_GRAPH    | ON *(default)* \|OFF *(Optional)*     | Enable build with [SYCL Graph extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc). |
| GGML_SYCL_DNN      | ON *(default)* \|OFF *(Optional)*     | Enable build with oneDNN.                   |
| CMAKE_C_COMPILER   | `icx` *(Linux)*, `icx/cl` *(Windows)* | Set `icx` compiler for SYCL code path.      |
| CMAKE_CXX_COMPILER | `icpx` *(Linux)*, `icx` *(Windows)*   | Set `icpx/icx` compiler for SYCL code path. |

1. FP16 is recommended for better prompt processing performance on quantized models. Performance is equivalent in text generation but set `GGML_SYCL_F16=OFF` if you are experiencing issues with FP16 builds.

#### Runtime

| Name              | Value            | Function                                                                                                                  |
|-------------------|------------------|---------------------------------------------------------------------------------------------------------------------------|
| GGML_SYCL_DEBUG   | 0 (default) or 1 | Enable log function by macro: GGML_SYCL_DEBUG                                                                             |
| GGML_SYCL_DISABLE_GRAPH | 0 or 1 (default) | Disable running computations through SYCL Graphs feature. Disabled by default because graph performance isn't yet better than non-graph performance. |
| GGML_SYCL_DISABLE_DNN | 0 (default) or 1 | Disable running computations through oneDNN and always use oneMKL. |
| GGML_SYCL_CPU_OFFLOAD | 0 (default) or 1 | Enable CPU offload dispatch for host-resident layers when a SYCL CPU device is available. |
| GGML_SYCL_CPU_OFFLOAD_ASYNC | 1 (default) or 0 | CPU offload staging sync policy: `1` waits at staging-bank reuse/boundaries, `0` uses eager per-op draining. |
| GGML_SYCL_CPU_DEVICE_SELECTOR | e.g. `opencl:cpu` | Optional override for CPU offload queue selection when `ONEAPI_DEVICE_SELECTOR` does not expose a CPU device. |
| GGML_SYCL_CPU_BATCH_THRESHOLD | integer | Legacy global CPU batch threshold (applies to both PP and TG when set). |
| GGML_SYCL_CPU_BATCH_THRESHOLD_PP | 4 (default) | CPU offload batch threshold for prompt-processing phase. |
| GGML_SYCL_CPU_BATCH_THRESHOLD_TG | 16 (default) | CPU offload batch threshold for decode/token-generation phase. |
| GGML_SYCL_HOST_COMPUTE | 0 (default) or 1 | Use host-pinned compute buffers for CPU-offload activation tensors. Keep `0` for best/stable mixed L0+OpenCL offload throughput; `1` is opt-in for debugging/experiments. |
| GGML_SYCL_CPU_STAGING_GROW_GRANULARITY_KB | 256 (default) | CPU offload staging growth granularity in KiB (larger value reduces realloc churn at the cost of extra headroom). |
| GGML_SYCL_CPU_OFFLOAD_VECDOT_MIN_WORK | 512 (default) | Minimum vec_dot output work (`N*M`) required before enabling TBB parallelization in CPU offload `MUL_MAT`. |
| GGML_SYCL_CPU_OFFLOAD_VECDOT_MIN_ROWS_PER_TASK | 4 (default) | Minimum rows-per-task grain for TBB partitioning in CPU offload vec_dot path. |
| GGML_SYCL_CPU_OFFLOAD_VECDOT_TASKS_PER_THREAD | 2 (default) | Target number of TBB vec_dot tasks per CPU thread in the offload path. |
| GGML_SYCL_OFFLOAD_STATS | 0 (default) or 1 | Emit per-graph offload summary counters (`wait_count`, alloc counts, pool hit/miss, transfer counts/bytes, CPU/GPU dispatch counts, transition wait/elide counts). |
| GGML_SYCL_OFFLOAD_LOCALITY | 1 (default) or 0 | Enable locality smoothing for CPU/GPU offload planning to reduce ping-pong segments. |
| GGML_SYCL_OFFLOAD_HYSTERESIS | phase-aware default (PP=3, TG=2) | Minimum segment length before flipping domain in locality smoothing. |
| GGML_SYCL_OFFLOAD_CROSS_COST | phase-aware default (PP=2, TG=1) | Additional penalty for short CPU segments when locality smoothing is enabled. |
| GGML_SYCL_OFFLOAD_BOUNDARY_WAIT_BYTES | 1048576 (default) | In async offload mode, only force GPU→CPU boundary queue wait when tracked boundary bytes are at least this threshold. |
| GGML_SYCL_OFFLOAD_PLAN_DUMP | 0 (default) or 1 | Dump CPU/GPU segment plan, boundary count, and estimated boundary bytes for each graph. |
| GGML_SYCL_MOE_GATEUP_SINGLE_XMX | 0 (default) or 1 | Experimental GPT-OSS MXFP4 gate/up proof mode. Requires gate/up experts to use one persistent `GGML_LAYOUT_XMX_TILED` VRAM layout consumed by both PP and TG. No persistent SOA gate/up duplicate and no per-token gate/up prepack are allowed. Parser validation must include `--require-single-xmx-gateup --forbid-gateup-soa-fallback`. Lead-only non-dry-run B50/B580/model validation is required before promotion; workers must use dry-run-only gates. |
| GGML_SYCL_MOE_GATEUP_M2_TG1_INDEX | 0 (default) or 1 | Default-off GPT-OSS MXFP4 decode-only proof knob for the packed-Q8 M2 gate/up route. When enabled it may specialize `mxfp4.gateup.xmx_tiled_dpas_m2` indexing for `n_tokens == 1`; it must not affect prompt processing or authorize V2, bundle4, M4, prefetch, direct-q8, or default-on behavior without lead-owned correctness and throughput evidence. |
| GGML_SYCL_DMA_SLICE_MB | 1024 (default) | Unified-cache DMA streaming slice size in MB (aligned to row size). |
| GGML_SYCL_DMA_BUFFERS | 2 (default) | Unified-cache DMA streaming buffer count (staging buffers). Alias: `GGML_SYCL_DMA_SLICES`. |
| GGML_SYCL_DMA_RESERVE_MB | (auto) | VRAM headroom reserved for DMA staging buffers; overrides slice/buffer-derived default. |
| GGML_SYCL_SET_TENSOR_STREAM_FENCE | 0 (default) or 1 | In `set_tensor`, wait only on the current stream before host copy (safer than no fence, cheaper than global drain). |
| GGML_SYCL_SET_TENSOR_GLOBAL_DRAIN | 0 (default) or 1 | In `set_tensor`, force legacy global queue drain before host copy. Mainly for debugging/bisecting. |
| GGML_SYCL_COPY_TO_DEVICE_SYNC | 0 (default) or 1 | Force legacy synchronous `copy_to_device_async` behavior for bringup/debug. |
| ZES_ENABLE_SYSMAN | 0 (default) or 1 | Support to get free memory of GPU by sycl::aspect::ext_intel_free_memory.<br>Recommended to use when --split-mode = layer |
| UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS | 0 (default) or 1 | Support malloc device memory more than 4GB.|

### MXFP4 MoE TG Microbench Suite

`sycl-mxfp4-moe-bench` is a synthetic research tool for B50 GPT-OSS MXFP4 token-generation route screening. It does not change llama runtime dispatch and is worker-safe only when used in `--dry-run` mode.

Worker-safe smoke commands:

```bash
./scripts/sycl-build.sh sycl-mxfp4-moe-bench
python3 scripts/run-sycl-mxfp4-tg-microbenches.py --dry-run --out-dir /tmp/mxfp4_tg_dryrun
python3 scripts/parse-sycl-mxfp4-tg-bench.py /tmp/mxfp4_tg_dryrun/baseline.jsonl --require-route baseline
```

Lead-only non-dry-run and model validation are documented in activation/mxfp4-tg-microbench-lead-validation.md.

## Memory placement

SYCL builds use the unified cache as the memory placement authority. Model
weights, KV cache, pinned host memory, and backend-visible SYCL buffers must be
placed by the unified cache planner or by SYCL buffer types that delegate to it.

The generic llama.cpp fit path (`-fit`, `--fit-target`, and `--fit-ctx`) is
disabled for SYCL builds. That path rewrites `n_gpu_layers`, tensor split, and
tensor buffer overrides before model loading, which duplicates and can conflict
with unified cache placement. In SYCL builds, common initialization ignores
`-fit on`, `llama-bench` ignores `--fit-target`/`--fit-ctx`, and direct calls to
`common_fit_params()` return without modifying placement parameters.

#### Flash Attention oneDNN Layout Policy

The SYCL flash-attention oneDNN path uses the oneDNN Graph SDPA pattern
`MatMul(Q,K^T) -> Divide(scale) -> Add(mask) -> SoftMax -> MatMul(V)`.
The direct path is enabled only when tensor metadata can be represented safely
as oneDNN strided logical tensors:

- MHA uses 4-D logical tensors `(batch, heads, seq, head_dim)`.
- GQA/MQA uses 5-D logical tensors `(batch, kv_heads, head_repetition, seq, head_dim)`.
- Q/K/V must be f16, mask must be f16 or f32, `head_dim <= 512`, no attention sinks,
  no logit softcap, no FP8 KV, no multi-sequence batch, tensor batch must be the
  proven batch-1 descriptor class, K/V must not use paged layouts, and prompt
  batch must meet `GGML_SYCL_FA_ONEDNN_MIN_NCOLS` (default `8`).
- Direct GQA/MQA additionally requires the physical K/V token stride to match
  `head_dim`. If `nc_stride != head_dim`, the planner marks the shape as
  `MATERIALIZE_REQUIRED` and repacks K/V into dense f16 device buffers before
  oneDNN execute. Allocation or repack failure still falls back to native FA.
- oneDNN's GQA example (`/opt/intel/oneapi/dnnl/2025.3/share/doc/dnnl/examples/gqa.cpp`)
  defines 5-D GQA as Q/output `(mb, kv_heads, head_rep, seq, head_size)`,
  K/V `(mb, kv_heads, 1, seq, head_size)`, score
  `(mb, kv_heads, head_rep, seq, seq)`, and mask `(mb, 1, 1, 1, seq)`.
  The SYCL backend follows that rank/order while using explicit strides for
  GGML storage.

`GGML_SYCL_FA_FORCE_PATH=onednn` is diagnostic for unsupported layouts: it prints
the planner reason and falls back instead of bypassing safety. For proven
MATERIALIZE_REQUIRED GQA/MQA shapes it exercises the materialized oneDNN path.
`GGML_SYCL_FA_ONEDNN_ALLOW` does not override the layout planner.

Token-generation decode uses a separate native FA policy. The conservative
safe-decode fallback remains available, but the default now bypasses it for
single-query f16 descriptor classes with an integral Q-head/KV-head ratio. The
ESIMD f16 decode kernel implements stateless attention modifiers directly:
attention sinks, logit softcap, and ALiBi bias may use the fast path. FP8 KV,
paged K/V, multi-sequence routing/IDs, and multi-token decode remain on the
conservative path until those address/ownership variants have separate proof.
`GGML_SYCL_FA_SAFE_DECODE=1` forces the fallback for debugging, and
`GGML_SYCL_FA_SAFE_DECODE=0` disables the safe-decode gate entirely for A/B
testing. `GGML_SYCL_FA_DISPATCH_DEBUG=1` prints the selected decode kernel
(`esimd_f16`, `vec_f16`, `tile_f16`, `xmx_*`, or `onednn`) and the fast-decode
eligibility bit.

On B580 with Mistral 7B Q4_0, build `5b206c499-dirty`, the fixed default
`-fa 1` policy measured PP512 `2173.92 +/- 10.01` tok/s and TG128
`88.42 +/- 0.47` tok/s. The deterministic completion gate also produced the
expected `1, 2, 3, ..., 10` sequence while default decode selected `esimd_f16`.
Logs for that validation were `/tmp/kkxtv7.7-default-fa-completion.log` and
`/tmp/kkxtv7.7-default-fa-pp512-tg128.log`.

On B50 with GPT-OSS 20B MXFP4, build `5b206c499-dirty`, a decode-focused
`-p 1 -n 16 -fa 1` run selected `esimd_f16` with `fast_esimd_safe=1` for the
sinks+softcap decode path and measured TG16 `14.65 +/- 0.05` tok/s. The full
PP512/TG128 GPT-OSS B50 benchmark is currently blocked before decode by a
separate MoE CPU expert host-zone allocation abort in `MUL_MAT_ID`; see
`/tmp/kkxtv7.7-gptoss20b-b50-fa-op-relax-pp512-tg128.log`.

`GGML_SYCL_MOE_GATEUP_SINGLECOL=1` is an experimental GPT-OSS MXFP4 TG-only
gate/up candidate. It remains default-off: JD32 synthetic proof on B50 showed
`singlecol` rows were numerically exact but slower than the packed-Q8 M2
synthetic baseline, so full-model promotion was skipped. Promotion would still
require exact count correctness, `fatal.total 0`, `PP512 >= 1200`, `TG128 >=
45`, `singlecol-gateup` route evidence, and gate/up profile `<= 4.2 ms`. This
flag does not authorize prompt XMX or persistent duplicate gate/up layouts.

A same-expert multi-RHS MXFP4 gate/up benchmark-only candidate was also
rejected before runtime wiring. B50 synthetic rows validated exactly, but
`mxfp4_pair_glu_xmx_tiled_multirhs_n2_r8` measured `605.034755 us` and
`mxfp4_pair_glu_xmx_tiled_multirhs_n4_r8` measured `1323.299220 us` versus the
packed-Q8 M2 baseline at `235.588515 us`. No production
`GGML_SYCL_MOE_GATEUP_MULTIRHS` route is authorized or wired; the benchmark
result does not authorize role-column gate/up fusion or persistent duplicate
gate/up layouts.

The benchmark-only `XMX_TILED_V2` aligned-payload MXFP4 gate/up layout candidate
was rejected before runtime wiring. On B50, `/tmp/v2_gateup_synth.jsonl` showed
the current packed-Q8 M2 synthetic baseline at `237.084865 us` with
`max_abs_error=0.000000`, while
`mxfp4_pair_glu_xmx_tiled_v2_packed_r8_m2_sparse32_bias` validated exactly but
measured `251.179255 us` with `max_abs_error=0.000000`, missing the `<= 188.47
us` continue gate. Launch-timing diagnostics then showed non-device launch/drain
overhead at only `3.19%` of raw summed gate/up time, and host-only expert
histogram diagnostics produced no histogram lines in the canonical single-stream
B50 run because `ids_host` was not available. No V2 runtime route, graphlet
promotion, grouped-reuse route, or persistent duplicate gate/up layout is
authorized by this evidence.

The benchmark-only `bundle4` MXFP4 gate/up layout candidate was also rejected
for runtime follow-up after lead-owned synthetic and VTune checks in
`/tmp/sycl_mxfp4_gateup_bundle4_20260701_152716`. It was exact, but only
`1.14%` faster than the refreshed packed-Q8 M2 baseline: baseline
`274.483888 us`, V2 `282.588609 us`, and bundle4 `271.363688 us`, all with
`max_abs_error=0.000000`. VTune instruction-count mode reported baseline
`100224000` GPU instructions and bundle4 `101151360`; both had spill memory
`0`, `SIMD Utilization(%)=91.3`, `dpas.8x8=4`, and `send.ugm=65` from ocloc
assembly summaries. VTune stdout labeled the GPU as `Battlemage G21 [Arc B580]`
despite the process running with `ONEAPI_DEVICE_SELECTOR=level_zero:1`, so this
record uses the VTune data only for relative instruction/disassembly evidence
and does not make B50-specific absolute VTune claims. Decision:
`bundle4-rejected`; no runtime route, default-on behavior, production promotion,
graphlet route, or persistent duplicate gate/up layout is authorized by this
evidence.

A follow-up benchmark-only bundle4 opportunity pass added a non-bias route,
full-group A loads, and vectorized full-tile bias/output stores, then recorded
lead-owned evidence in
`/tmp/sycl_mxfp4_bundle4_opportunities_20260701_221449`. All synthetic rows were
exact, but the fastest route still missed the `10%` runtime-followup gate:
packed-Q8 M2 sparse/bias baseline `272.495794 us`, non-bias bundle4
`249.225306 us` (`9.34%` faster), and sparse/bias bundle4 `251.723551 us`
(`8.25%` faster), all with `max_abs_error=0.000000`. VTune compute-extended
active-kernel rows reported XMX-active / read-bandwidth values of `0.0%` /
`0.0 GB/s` for baseline, `0.0%` / `0.0 GB/s` for non-bias bundle4, and `6.9%`
/ `224.411295 GB/s` for sparse/bias bundle4; the zeroed baseline/non-bias rows
are retained as collected evidence but are not overinterpreted. VTune
mem-latency reported active-kernel read latency / estimated GPU cycles of
`318 cycles` / `469561326` for baseline, `526 cycles` / `479885090` for
non-bias bundle4, and `471 cycles` / `432343346` for sparse/bias bundle4. ocloc
assembly summaries kept `dpas.8x8=4` and spill memory `0`; baseline had
`send.ugm=65`, `16` scalar store comments, and no vector-store comments, while
optimized bundle4 had `send.ugm=75`, `16` scalar store comments from the
retained tail fallback, and `2` vector-store comments for the full-tile path.
VTune summaries again labeled
the target as `Battlemage G21 [Arc B580]` despite
`ONEAPI_DEVICE_SELECTOR=level_zero:1` and `target-gpu=0:7:0.0`, so these numbers
are relative benchmark evidence and not B50-specific absolute VTune claims.
Decision: `bundle4-opportunity-rejected`; no runtime route, default-on behavior,
production promotion, graphlet route, or persistent duplicate gate/up layout is
authorized by this evidence.

The planner-owned materialization contract is implemented separately from the
oneDNN execute gate:

- Source K/V are f16 GGML flash-attention views with dimensions
  `[head_dim, kv_tokens, kv_heads, batch]` and byte strides carried by
  `fattn_params`; the executable materialized path is currently limited to
  `batch == 1`.
- Target K/V are dense f16 device buffers on the planned SYCL device with token
  stride `head_dim * sizeof(f16)` and head stride
  `kv_tokens * head_dim * sizeof(f16)`.
- Target allocation ownership is through the unified-cache allocation policy and
  exposed to users as smart `mem_handle` objects. Private FA caches and raw
  untracked cross-device buffers are not valid ownership.
- Materialization is not attempted during SYCL graph recording or for paged K/V;
  native FA handles those cases.
- End-to-end oneDNN consumption of materialized GQA/MQA K/V is enabled only for
  the proven f16 descriptor class above. The output logical tensor is not dense
  `(batch, heads, seq, head_dim)` storage: it writes directly to GGML dst layout
  `[head_dim, heads, seq, batch]`, so the required output strides are
  4-D `{heads * seq * head_dim, head_dim, heads * head_dim, 1}` and 5-D
  `{heads * seq * head_dim, head_rep * head_dim, head_dim, heads * head_dim, 1}`.
  The old dense output descriptor is rejected by the synthetic descriptor test
  because it permutes query/head output and reproduces deterministic corruption.

#### CPU Offload Bench + VTune Harness

Use the helper script to run PP/TG separately with unified-cache CPU offload and optional VTune collection:

```bash
./scripts/sycl-cpu-offload-bench-vtune.sh \
  --model /Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf \
  --profile nonstream-cpuoffload
```

Defaults used by the script:
- `ONEAPI_DEVICE_SELECTOR='level_zero:0;opencl:cpu'`
- Unified cache is mandatory; no enable flag is required.
- `GGML_SYCL_CPU_OFFLOAD=1`
- `GGML_SYCL_CPU_OFFLOAD_ASYNC=1`
- `GGML_SYCL_CPU_BATCH_THRESHOLD_PP=4`
- `GGML_SYCL_CPU_BATCH_THRESHOLD_TG=16`
- `GGML_SYCL_VRAM_BUDGET_PCT=25`

The harness reports PP/TG separately and can enforce metric gates:
- `zeMemAllocHost` call count
- `zeEventHostSynchronize` call count
- `ggml_backend_sycl_buffer_set_tensor` CPU time

## Cross-device KV placement contract

This contract applies to hidden-device multi-GPU runs where the SYCL backend may
expose fewer scheduler devices than physical Level Zero GPUs. For example, the
runtime can expose `device_count == 1` to the ggml scheduler while still keeping
`total_gpu_count > device_count` physical GPUs available to the unified cache for
expert, weight, KV, or activation ownership.

The planner is the source of truth for ownership. The `kv_device` and `layer_device` fields have distinct meanings: `layer_device` identifies the
device that should execute a layer's dense work, and `kv_device` identifies the
device that owns that layer's KV roots. A non-negative `kv_device` is a physical
SYCL device ordinal in the unified-cache device namespace, even when that ordinal
is greater than or equal to scheduler-visible `device_count`. Host KV is valid
only when the planner explicitly chooses host placement, an explicit KV-host mode
is enabled, or a documented fail-fast allocation error is being reported.

`SYCL_KV_Tiered` must materialize `cache_k_l*` and `cache_v_l*` root tensors on
the planned owner. Remote device-planned KV must never be represented as ordinary host-pinned fallback. A tensor planned for physical device 1 remains device-owned
by device 1; it is not a host tensor merely because the scheduler currently runs
the graph on device 0.

Smart handles carry this ownership across roots, views, and staging decisions.
The KV root tensor must have an `extra_gpu` smart handle for the planned owner,
and views of that root must resolve through root plus `view_offs` for the
requested device. A view-local raw pointer or stale direct handle is not allowed
to override the root smart-handle owner. If execution needs data on another GPU,
the routing layer must either execute on the KV owner or explicitly stage/migrate
through unified-cache handles.

CPU fallback is invalid for device-planned F16 attention and KV. When F16 attention consumes device-planned KV, code should fail fast with ownership
diagnostics or route the operation to a GPU owner. It must not silently recover
by running CPU attention over host-pinned copies of KV that the planner assigned
to a GPU.

The safe default for hidden-device multi-GPU MoE is expert-mode execution, and hybrid/layer placement remains explicit until cross-device KV routing is implemented. In expert mode, dense activations and KV stay local to the
scheduler-visible GPU while experts use the unified cache across physical GPUs.
Hybrid/layer can assign KV to a hidden physical device and therefore requires
the full ownership and routing contract described here.

Diagnostics for future cross-device KV tasks must report, at minimum, the planned device, materialized owner, buffer type, root extra/smart-handle status, and routing decision for each affected KV layer or F16 attention operation.

## Known Issues

## Runtime Allocation Policy

- Managed runtime allocations must go through unified allocator APIs in `unified-cache.cpp`.
- Do not add new direct `sycl::malloc_device`, `sycl::malloc_host`, or `sycl::malloc_shared` calls in managed runtime paths.
- Use `./scripts/check-sycl-alloc-usage.sh` to validate policy compliance locally.
- Allowed managed placement tiers are:
  - device VRAM (`malloc_device`)
  - host pinned (`malloc_host`)
  - mmap tracked reservations (budget accounting only)
- Shared USM (`malloc_shared`) is not allowed for managed CPU-offload/host-compute paths.

- `Split-mode:[row]` is not supported.

## Q&A

- Error:  `error while loading shared libraries: libsycl.so: cannot open shared object file: No such file or directory`.

  - Potential cause: Unavailable oneAPI installation or not set ENV variables.
  - Solution: Install *oneAPI base toolkit* and enable its ENV through: `source /opt/intel/oneapi/setvars.sh`.

- General compiler error:

  - Remove **build** folder or try a clean-build.

- I can **not** see `[ext_oneapi_level_zero:gpu]` afer installing the GPU driver on Linux.

  Please double-check with `sudo sycl-ls`.

  If it's present in the list, please add video/render group to your user then **logout/login** or restart your system:

  ```
  sudo usermod -aG render $USER
  sudo usermod -aG video $USER
  ```
  Otherwise, please double-check the GPU driver installation steps.

- Can I report Ollama issue on Intel GPU to llama.cpp SYCL backend?

  No. We can't support Ollama issue directly, because we aren't familiar with Ollama.

  Sugguest reproducing on llama.cpp and report similar issue to llama.cpp. We will surpport it.

  It's same for other projects including llama.cpp SYCL backend.

- `Native API failed. Native API returns: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)`, `ggml_backend_sycl_buffer_type_alloc_buffer: can't allocate 3503030272 Bytes of memory on device`, or `failed to allocate SYCL0 buffer`

  You are running out of Device Memory.

  |Reason|Solution|
  |-|-|
  | The default context is too big. It leads to excessive memory usage.|Set `-c 8192` or a smaller value.|
  | The model is too big and requires more memory than what is available.|Choose a smaller model or change to a smaller quantization, like Q5 -> Q4;<br>Alternatively, use more than one device to load model.|

- `ggml_backend_sycl_buffer_type_alloc_buffer: can't allocate 5000000000 Bytes of memory on device`

  You need to enable to support 4GB memory malloc by:
  ```
    export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
    set UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
  ```

### **GitHub contribution**:
Please add the `SYCL :` prefix/tag in issues/PRs titles to help the SYCL contributors to check/address them without delay.

### Named SYCL kernel event profiler

Set `GGML_SYCL_KERNEL_PROFILE=1` to enable the backend-wide named kernel profiler. The profiler records SYCL event profiling timestamps under human labels assigned by llama.cpp, not VTune task names. VTune computing-task attribution is not the source of truth for this route; SYCL event profiling timestamps are.

Useful variables:

| Variable | Effect |
|---|---|
| `GGML_SYCL_KERNEL_PROFILE=1` | Enable named event collection. |
| `GGML_SYCL_KERNEL_PROFILE_OUTPUT=/tmp/sycl-kernels` | Write scriptable artifacts. `.csv` and/or `.json` suffixes are added for `both`. |
| `GGML_SYCL_KERNEL_PROFILE_FORMAT=csv,json,both` | Select artifact format. |
| `GGML_SYCL_KERNEL_PROFILE_TOP_N=40` | Number of rows in the stderr top-k summary. |
| `GGML_SYCL_KERNEL_PROFILE_FLUSH=final,window,none` | `final` harvests at explicit teardown/tool flushes, `window` waits at requested profiler flush points, and `none` records opportunistically without forcing completeness. |
| `GGML_SYCL_KERNEL_PROFILE_RAW=1` | Include raw per-event rows when supported. |

Model-free smoke command:

```bash
set +u; source /opt/intel/oneapi/setvars.sh --force; set -u
ONEAPI_DEVICE_SELECTOR=level_zero:1 \
GGML_SYCL_KERNEL_PROFILE=1 \
GGML_SYCL_KERNEL_PROFILE_OUTPUT=/tmp/sycl-kernels \
GGML_SYCL_KERNEL_PROFILE_FORMAT=both \
GGML_SYCL_KERNEL_PROFILE_FLUSH=window \
./build/bin/sycl-kernel-bench \
  --kernel=mxfp4_pair_glu_xmx_tiled_packed_r8_m2 \
  --quant=MXFP4 --dim_m=2880 --dim_n=4 --dim_k=2880 \
  --iterations=10 --warmup=2 --output=json

python3 scripts/parse-sycl-kernel-profile.py \
  --require-kernel mxfp4.gateup.xmx_tiled_dpas_m2 \
  /tmp/sycl-kernels.csv
```

### MXFP4 down-variant named profile matrix

`scripts/sycl-gptoss-down-variant-profile-matrix.sh` is a lead-only helper for comparing default-off GPT-OSS MXFP4 decode down variants with the named SYCL event profiler. It is dry-run by default and must not launch real model work unless the lead passes `--execute`.

The baseline, row-group, and atomic rows use the valid FA-on B50 GPT-OSS baseline environment: `-fa 1`, `GGML_SYCL_MOE_PHASE_MATERIALIZE=1`, `GGML_SYCL_MOE_PHASE_BULK_XMX=1`, and `GGML_SYCL_MOE_DOWN_SUM_DIRECT=1`. Cached q8-SOA rows are diagnostic only and explicitly set `GGML_SYCL_MOE_DOWN_SUM_DIRECT=0`; they are not promotion candidates for the current FA-on direct-sum baseline.

MXFP4 direct-final down remains default-off. Existing `GGML_SYCL_MOE_DOWN_SUM_*_DIRECT_FINAL*` knobs are proof/debug controls only, and direct-final promotion requires a separate approved plan plus lead-owned GPT-OSS correctness, PP preservation, and TG throughput gates.

### GPT-OSS MXFP4 end-to-end TG profile ledger

`GGML_SYCL_E2E_TG_PROFILE=1` enables a default-off decode ledger for GPT-OSS MXFP4 token generation. The ledger complements `[MXFP4-MOE-TG-PROFILE]` by reporting full decode stage evidence rather than MoE-only timing.

The summary line is:

```text
[SYCL-E2E-TG-PROFILE] tokens=1 ops=512 moe_calls=72 total_host=18.250 ms total_device=7.125 ms
```

Stage rows are:

```text
[SYCL-E2E-TG-STAGE] stage=moe calls=72 host=0.900 ms device=6.500 ms bytes=0 last_path=packed-q8-m2
[SYCL-E2E-TG-STAGE] stage=attention calls=32 host=0.450 ms device=0.500 ms bytes=1048576 last_path=xmx_v2_f16_pp_ncols32
```

Parse logs with:

```bash
python3 scripts/parse-sycl-moe-profile.py \
  --require-no-fatal-markers \
  --require-mxfp4-profile-evidence \
  --require-e2e-profile-evidence \
  --require-e2e-stage moe \
  --require-e2e-stage attention \
  /tmp/sycl_gptoss_e2e_profile_lead_20260630_000000/baseline/bench.stdout \
  /tmp/sycl_gptoss_e2e_profile_lead_20260630_000000/baseline/bench.stderr
```

For command generation, use dry-run mode first:

```bash
bash scripts/sycl-gptoss-e2e-profile-matrix.sh --dry-run
```

Real model validation is lead-owned and validates the explicitly selected device.
The harness defaults to `ONEAPI_DEVICE_SELECTOR=level_zero:1` (B50 on this
workstation); use `--device-selector level_zero:0` for B580 or another explicit
selector. Real execution requires:

```bash
bash scripts/sycl-gptoss-e2e-profile-matrix.sh --run --i-understand-this-runs-gpu-models
```

Do not use this ledger to bypass existing safety gates. Runtime optimization remains default-off until the canonical GPT-OSS correctness gate, fatal-marker parser gates, PP preservation, and TG improvement are all proven on lead-owned hardware.

## TODO

- Review ZES_ENABLE_SYSMAN: https://github.com/intel/compute-runtime/blob/master/programmers-guide/SYSMAN.md#support-and-limitations
