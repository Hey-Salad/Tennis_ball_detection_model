# Tennis Ball Detection Model

This repository contains a YOLOv5-based tennis ball detection model that has been converted for use with reCamera.

## Environment Setup

### Hardware Requirements
The conversion process was performed on a system with NVIDIA GPU support. Here's the GPU configuration used:

```bash
$ nvidia-smi
Tue Feb 11 22:23:48 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla P100-PCIE-16GB           On  |   00000000:00:04.0 Off |                    0 |
| N/A   32C    P0             27W /  250W |       3MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### Software Requirements

- Ubuntu 22.04
- Python 3.10
- Docker
- tpu-mlir v1.15.1-20250208

## Step-by-Step Conversion Guide

### 1. Setup Docker Environment

Pull the required Docker image:
```bash
$ docker pull sophgo/tpuc_dev:v3.1
```

Create and enter the Docker container:
```bash
$ docker run --privileged --name tennis_model -v $PWD:/workspace -it sophgo/tpuc_dev:v3.1
```

### 2. Install Required Dependencies

Inside the Docker container:
```bash
$ pip install tpu_mlir[all]
```

### 3. Prepare the Environment

Clone the necessary repositories and set up the environment:
```bash
$ cd /workspace
$ git clone https://github.com/sophgo/tpu-mlir.git
$ cd tpu-mlir
$ source ./envsetup.sh
$ ./build.sh
```

### 4. Model Conversion Process

#### Create Working Directory
```bash
$ mkdir model_tennis && cd model_tennis
$ cp ../Tennis_ball_detection_model/best.onnx .
$ mkdir workspace && cd workspace
```

#### Convert ONNX to MLIR
```bash
$ model_transform \
    --model_name tennis_detect \
    --model_def ../best.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean "0.0,0.0,0.0" \
    --scale "0.0039216,0.0039216,0.0039216" \
    --keep_aspect_ratio \
    --pixel_format rgb \
    --mlir tennis_detect.mlir
```

Expected output will include:
```
2025/02/12 06:54:40 - INFO : TPU-MLIR v1.15.1-20250208
2025/02/12 06:54:40 - INFO : Input_shape assigned
...
2025/02/12 06:54:41 - INFO : Mlir file generated:tennis_detect.mlir
```

#### Prepare Calibration Data
```bash
$ mkdir -p COCO2017/images
$ wget http://images.cocodataset.org/val2017/000000000139.jpg -P COCO2017/images/
$ wget http://images.cocodataset.org/val2017/000000000285.jpg -P COCO2017/images/
$ wget http://images.cocodataset.org/val2017/000000000632.jpg -P COCO2017/images/
$ wget http://images.cocodataset.org/val2017/000000000724.jpg -P COCO2017/images/
```

#### Run Calibration
```bash
$ run_calibration \
    tennis_detect.mlir \
    --dataset ../COCO2017/images \
    --input_num 4 \
    -o tennis_detect_calib_table
```

#### Convert to INT8 Model
```bash
$ model_deploy \
    --mlir tennis_detect.mlir \
    --quantize INT8 \
    --chip cv181x \
    --calibration_table tennis_detect_calib_table \
    --model tennis_detect_int8.cvimodel
```

### 5. Verify Converted Model

Check the model information:
```bash
$ model_tool --info tennis_detect_int8.cvimodel
```

Expected output will show:
```
Mlir Version: v1.15.1-20250208
Cvimodel Version: 1.4.0
tennis_detect Build at 2025-02-12 07:03:16
For cv181x chip ONLY
CviModel Need ION Memory Size: (11.73 MB)
```

## Model Specifications

### Input Requirements
- Shape: [1, 3, 640, 640]
- Format: RGB
- Preprocessing:
  - Mean: [0.0, 0.0, 0.0]
  - Scale: [0.0039216, 0.0039216, 0.0039216]
  - Keep aspect ratio: True
  - Pad type: center

### Memory Requirements
- ION Memory Size: 11.73 MB
- Private GMEM Size: 1228800 bytes
- Shared GMEM Size: 2395600 bytes

## Deployment

### On reCamera
1. Copy the converted model to your reCamera device
2. Use Node-RED for visualization:
   - Import model using model node
   - Connect to camera input
   - Add preview node for visualization

### Model Files
- Original: `best.onnx`
- Converted: `tennis_detect_int8.cvimodel`
- Intermediate: 
  - `tennis_detect.mlir`
  - `tennis_detect_calib_table`

## Troubleshooting

Common warning messages during conversion:
```
- WARNING : ConstantFolding failed.
- WARNING : onnxsim opt failed.
```
These warnings are normal and don't affect the final model functionality.

## License

[Add your license information here]

## Contributing

Feel free to open issues or submit pull requests for improvements.

## Acknowledgments

- YOLOv5 team for the original architecture
- Sophgo team for the tpu-mlir toolchain
- COCO dataset for calibration images
