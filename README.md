# Medical Image Segmentation
Integrating Active Contour Models with Attention Mechanisms in Deep Learning for Medical Image Segmentation

## Project Outline:
**Objective**: Develop a hybrid image segmentation model that integrates Active Contour Models (ACMs) within the attention mechanism of a deep learning model, targeting medical images where precise boundary detection is crucial.

**Deliverables**:
1. A literature review on deep learning models for segmentation and ACMs.
2. A prototype of a hybrid ACM-attention deep learning model.
3. Performance evaluation on medical image datasets.
4. A final report detailing the approach, results, and challenges.

## Datasets 
**One-Prompt**: One-Prompt to Segment All Medical Images, or say One-Prompt, combines the strengths of one-shot and interactive methods. In the inference stage, with just one prompted sample, it can adeptly handle the unseen task in a single forward pass.

* Repository: https://github.com/MedicineToken/one-prompt 
* Dataset: https://drive.google.com/file/d/1iXFm9M1ocrWNkEIthWUWnZYY2-1l-qya/view 
    * Skin Lesion (ISIC): https://challenge.isic-archive.com/data/#2020

## Examples
### Active Contour Models 

**ACM:** Pytorch Impelmentation of Active Contour Models:

https://github.com/noornk/ACM/tree/main

**(DALS) Deep Active Lesion Segmentation**: DALS offers a fast levelset active contour model, implemented entirely in Tensorflow, that can be paired with any CNN backbone for image segmentation.

https://github.com/ahatamiz/dals 

## Installation
```bash
pip install -r requirements.txt 
```

## Data Usage 
Different datasets are stored in `data/` 

## POC 
### Image Segmentation Usage
Call different segmentations using the following command: 
```bash
python image_segmentation/edge_based.py
```

## Model
Fix the utils/config.json to your liking with various epochs, etc. 

Then run: 
```sh
python main.py
```
