# Medical Image Segmentation
Integrating Active Contour Models with Attention Mechanisms in Deep Learning for Medical Image Segmentation

## Project Outline:
**Objective**: Develop a hybrid image segmentation model that integrates Active Contour Models (ACMs) and attention mechanisms in a baseline deep learning model (CNN), targeting medical images where precise boundary detection is crucial.

**Deliverables**:
1. A literature review on deep learning models for segmentation and ACMs.
2. A prototype of a hybrid ACM-attention deep learning model.
3. Performance evaluation on medical image datasets.
4. A final report detailing the approach, results, and challenges.

## Final Project Report
We will explain and demo some basic ideas in this README file. A more detailed and thorough explanation, experimentation, and analysis is provided in the our final project report.
[View Project Report!](Graduate_Capstone_Report.pdf)

## Datasets 
* A collection of medical image datasets: https://drive.google.com/file/d/1iXFm9M1ocrWNkEIthWUWnZYY2-1l-qya/view 
* For our final model implementation and testing we used the lung COVID-19 CT-Scans Dataset: https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset. This dataset can also be downloaded by running `data/utils.py`.
* Other datasets we utilized for some initial experiments: ISIC, DRIVE, chase_db1. We found these datasets through Kaggle as well.
* ISIC: https://challenge.isic-archive.com/data/#2016
* Ensure all these datasets are downloaded under the `data/` folder of this repo.
* Note - some of the experiments/demos below will assume these datasets have been downloaded.

## Environment Setup
```bash
pip install -r requirements.txt 
```

## Literature
### Active Contour Models 

**ACM:** Pytorch Impelmentation of Active Contour Models:

https://github.com/noornk/ACM/tree/main

**(DALS) Deep Active Lesion Segmentation**: DALS offers a fast levelset active contour model, implemented entirely in Tensorflow, that can be paired with any CNN backbone for image segmentation.

https://github.com/ahatamiz/dals 


## POC
### ACM Usage


### Image Segmentation Usage
Call other edge-based segmentation mechanisms (Roberts/Sobel) using the following command in the [POC branch](https://github.com/shairaalam19/medicalimagesegmentation/tree/POC): 
```bash
python image_segmentation/edge_based.py
```

## Model
### Running the final model
Fix the utils/config.json to your liking with various epochs, etc. 

Then run: 
```sh
python main.py
```

### Example configuration and demo