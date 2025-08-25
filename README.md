
## 📐 MACs Calculation

Multiply–Accumulate Operations (MACs) are calculated as:

```
MACs = 2 × Kernel Size × Input Channels × Output Channels × Output Width × Output Height
```

Example for a `Conv2D` layer:

- Input Channels = 3 (RGB)
- Output Channels = 40
- Kernel Size = 3×3 = 9
- Stride = 2
- Padding = 1
- Input Size = 224×224

You can calculate MACs using the above formula after deriving the output feature map dimensions.

---

## 🏆 Which is Better for Satellite Image Segmentation?

- ✅ If **accuracy** is top priority: use ResNet-based DeepLabV3+, U-Net, etc.
- ⚡ If **real-time segmentation** is needed (e.g., drones): use lightweight models like **RepViT**.

---

## ⚙️ RepViTBlock Components

### 1. **Token Mixer**
- Responsible for spatial feature extraction.
- Uses:
  - `RepVGGDW`: depthwise convolution
  - `SEModule`: adds lightweight attention

### 2. **RepVGGDW**
- Depthwise conv applied per input channel.
- Efficient with reduced computational complexity.
- Derived from RepVGG architecture for faster inference.

### 3. **SEModule (Squeeze-and-Excitation)**
- Provides channel-wise attention.
- Architecture:
  - `fc1` → `ReLU` → `fc2` → `Sigmoid`
- Enhances important features while suppressing noise.

### 4. **Channel Mixer**
- Applies 1×1 convolutions for channel-wise mixing.
- Implemented as a residual block with:
  - Two `Conv2D_BN` layers
  - `GELU` activation

### 5. **GELU Activation**
Smoother than ReLU, used in transformers and modern conv nets.

```
GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))
```

---

## 🔍 RepViTBlock Layer Comparison

| Property        | Block (2)            | Block (53)            |
|----------------|----------------------|------------------------|
| Params         | 26.48K               | 1.851M                |
| MACs           | 433.848M             | 421.438M              |
| Channel Size   | 80                   | 640                   |
| SE Module      | ❌ No                | ✅ Yes                |

- **Block 2**: Lightweight and early-stage processing
- **Block 53**: High-capacity, deeper-layer block with attention

---

## 📈 Functional Purpose in Segmentation

- Early layers (e.g., Block 2):
  - Extract low-level textures and edges
  - Lightweight and fast

- Deeper layers (e.g., Block 53):
  - Capture global context and semantics
  - Use SE attention for enhanced channel representation

✅ Alternating light-heavy block structure improves overall balance between **efficiency** and **feature expressiveness**

---

## 📒 Folder Structure

<details>
<summary>Prepare the following folders to organize this repo:</summary>

```
rssegmentation
├── rsseg
├── tools
├── work_dirs
├── data
│   ├── LoveDA
│   │   ├── Train/Urban/images_png, masks_png
│   │   ├── Train/Rural/images_png, masks_png
│   │   ├── Val
│   │   ├── Test
│   ├── vaihingen
│   │   ├── ISPRS_semantic_labeling_Vaihingen
│   │   ├── ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE
│   │   ├── ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE
│   │   ├── train
│   │   ├── test
│   ├── potsdam
│   │   ├── 2_Ortho_RGB
│   │   ├── 5_Labels_all
│   │   ├── 5_Labels_all_noBoundary
│   │   ├── train
│   │   ├── test
```

</details>

---

## 🔐 Preparation

### Environment

```bash
conda create -n rsseg python=3.9
conda activate rsseg
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Data Preprocess

```bash
bash tools/vaihingen_preprocess.sh 
bash tools/potsdam_preprocess.sh 
```

📦 Preprocessed datasets available:
- [Vaihingen](https://pan.baidu.com/s/18zLMK8-gleYFyyXl-OWHrg)
- [Potsdam](https://pan.baidu.com/s/1qunEBhLH_GhqsnU6ZVK6TQ)

---

## 📚 Use Example

### 1️⃣ Training

```bash
python train.py -c configs/vaihingen/logcanplus.py
```

### 2️⃣ Testing

#### Vaihingen and Potsdam
```bash
python test.py -c configs/vaihingen/logcanplus.py --ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt
```

#### LoveDA (for online evaluation)

```bash
python online_test.py -c configs/loveda/logcanplus.py --ckpt work_dirs/logcanplus_loveda/epoch=45.ckpt
```

---

## 🛠️ Tools

### Param and FLOPs

```bash
python tools/flops_params_count.py -c configs/vaihingen/logcanplus.py
```

### Latency

```bash
python tools/latency_count.py -c configs/vaihingen/logcanplus.py --ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt
```

### Throughput

```bash
python tools/throughput_count.py -c configs/vaihingen/logcanplus.py
```

### Class Activation Map

```bash
python tools/cam.py -c configs/vaihingen/logcanplus.py --ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt --tar_layer "model.net.seghead.catconv2[-2]" --tar_category 1
```

### TSNE Map

```bash
python tools/tsne.py -c configs/vaihingen/logcanplus.py --ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt
```

---

