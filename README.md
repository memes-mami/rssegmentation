

# 📷 Introduction

**rssegmentation** is an open-source semantic segmentation toolbox, which is dedicated to reproducing and developing advanced methods for semantic segmentation of remote sensing images.

<div align="center">
  <b>Overview</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="center">
      <td>
        <b>Methods</b>
      </td>
      <td>
        <b>Datasets</b>
      </td>
      <td>
        <b>Tools</b>
      </td>
    </tr>
	<tr valign="top">
      <td>
        <ul>
          <li><a href="https://ieeexplore.ieee.org/abstract/document/10095835/">LoG-Can (ICASSP2023) </a></li>
          <li><a href="https://ieeexplore.ieee.org/abstract/document/10219583/">SACANet (ICME2023)</a></li>
       		<li><a href="https://ieeexplore.ieee.org/abstract/document/10381808/">DOCNet (GRSL2024)</a></li>
          <li><a href="https://ieeexplore.ieee.org/document/10884928/">LOGCAN++(TGRS2025)</a></li>
          <li><a href="https://www.sciencedirect.com/science/article/pii/S0924271625000255?via%3Dihub">SCSM (ISPRS2025)</a></li>
          <li>CenterSeg (Under review)</a></li>
        </ul>
      </td>
<td>
        <ul>
          <li><a href="https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx">Vaihingen </a></li>
          <li><a href="https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx">Potsdam</a></li>
       		<li><a href="https://codalab.lisn.upsaclay.fr/competitions/421">LoveDA</a></li>
          <li>iSAID(to do)</a></li>
        </ul>
      </td>
<td>
        <ul>
          <li>Training </a></li>
          <li>Testing</a></li>
       		<li>Params, FLOPs, Latency, Throughput </a></li>
			<li>Class activation map </a></li>
			<li>TSNE map </a></li>
        </ul>
      </td>
</table>




# 📒 Folder Structure

<details>
<summary>
Prepare the following folders to organize this repo:
</summary>

```
rssegmentation
├── rsseg (core code for datasets and models)
├── tools (some useful tools)
├── work_dirs (save the model weights and training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original labels)
│   │   │   ├── Rural
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original labels)
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   ├── vaihingen
│   │   ├── ISPRS_semantic_labeling_Vaihingen 
│   │   │   ├── top (original images)
│   │   ├── ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE (original labels)
│   │   ├── ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE (original noBoundary lables)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam (the same with vaihingen)
│   │   ├── 2_Ortho_RGB (original images)
│   │   ├── 5_Labels_all (original labels)
│   │   ├── 5_Labels_all_noBoundary (original noBoundary lables)
│   │   ├── train (processed)
│   │   ├── test (processed)
```
</details>


# 🔐 Preparation

- **Environment**

```shell
conda create -n rsseg python=3.9
conda activate rsseg
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

- **Data Preprocess**

```shell
# Modify img-dir and mask-dir if necessary
bash tools/vaihingen_preprocess.sh 
bash tools/potsdam_preprocess.sh 
```

Note: We also provide the preprocessed [vaihingen](https://pan.baidu.com/s/18zLMK8-gleYFyyXl-OWHrg) and [potsdam](https://pan.baidu.com/s/1qunEBhLH_GhqsnU6ZVK6TQ) dataset.



# 📚 Use example

### 1️⃣ Training

```shell
python train.py -c configs/vaihingen/logcanplus.py
```

### 2️⃣ Testing

- **Vaihingen and Potsdam**

```shell
python test.py \
-c configs/vaihingen/logcanplus.py \
--ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt \
```

- **LoveDA**

Note that since the loveda dataset needs to be evaluated online, we provide the corresponding test commands.

```shell
python online_test.py \
-c configs/loveda/logcanplus.py \
--ckpt work_dirs/logcanplus_loveda/epoch=45.ckpt \
```

### 3️⃣ Useful tools

- **Param and FLOPs**

```shell
python tools/flops_params_count.py -c configs/vaihingen/logcanplus.py 
```
- **Latency**

```shell
python tools/latency_count.py \
-c configs/vaihingen/logcanplus.py \
--ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt \
```
- **Throughput**

```shell
  python tools/throughput_count.py -c configs/vaihingen/logcanplus.py
```

- **Class activation map**

```shell
python tools/cam.py \
-c configs/vaihingen/logcanplus.py \
--ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt \
--tar_layer "model.net.seghead.catconv2[-2]" \
--tar_category 1
```

* **TSNE map**

```shell
  python tools/tsne.py \
  -c configs/vaihingen/logcanplus.py \
  --ckpt work_dirs/logcanplus_vaihingen/epoch=45.ckpt 
```

  
