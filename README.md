# Medical Image Segmentation
Integrating Active Contour Models and Attention Mechanisms in Deep Learning for Medical Image Segmentation

## Project Outline:
**Objective**: Develop a hybrid image segmentation model that integrates Active Contour Models (ACMs) and attention mechanisms in a baseline deep learning model (CNN), targeting medical images where precise boundary detection is crucial.

**Deliverables**:
1. A literature review on deep learning models for segmentation and ACMs.
2. A prototype of a hybrid ACM-attention deep learning model.
3. Performance evaluation on medical image datasets.
4. A final report detailing the approach, results, and challenges.

## Final Project Report
We will explain and demo some basic ideas in this README file. A more detailed and thorough explanation, experimentation, analysis, and challenges description is provided in the our final project report.
[View Project Report!](Graduate_Capstone_Report.pdf)

## Datasets 
* A collection of medical image datasets: https://drive.google.com/file/d/1iXFm9M1ocrWNkEIthWUWnZYY2-1l-qya/view 
* For our final model training and testing, we used the Covid-19 lung CT scan lesion segmentation dataset: https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset. This dataset can also be downloaded by running `data/utils.py`.
* Other datasets we utilized for some initial experiments: ISIC, DRIVE, chase_db1. We found these datasets through Kaggle as well.
* ISIC: https://challenge.isic-archive.com/data/#2016
* Ensure all these datasets are downloaded under the `data/` folder of this repo.
* Note - some of the experiments/demos below will assume these datasets have been downloaded.

## Environment Setup
```bash
pip install -r requirements.txt 
```

## Active Contour Models
Active Contour Models (ACMs) are used to detect object boundaries in images by iteratively evolving curves to minimize an energy function. Although all ACMs follow the same energy minimization principle, their energy formulations and corresponding hyperparameters can differ, affecting performance across applications.

## Attention Mechanisms
**Base Attention Modules**: Scale module input using learned attention weights.

**Edge Attention**: The Edge Attention Module enhances feature learning by combining image features with edge information. First, a convolution + ReLU extracts features from the module input. Edges are obtained from module input using robert's operator logic and passed through an attention layer to generate attention weights. These weights are applied to the features, allowing the network to focus more on edge-relevant regions.

## ACM Implementation References

**Deep Active Lesion Segmentation (DALS)**: DALS offers a fast level set active contour model (LSA), implemented entirely in Tensorflow, that can be paired with any CNN backbone for image segmentation.
https://github.com/ahatamiz/dals 

**Chan-Vese (CV) ACM Implementation in Kaggle**: Region-based ACM. Region-based ACMs consider global information and segment objects by minimizing energy derived from statistical differences between regions inside and outside the contour, rather than relying solely on edge information or level-set (strip surrounding current contour) information.
https://www.kaggle.com/code/naim99/active-contour-model-python


## Proof Of Concept
### ACM Usage
We extracted ACM logic from both the DALS project and CV Kaggle implementations to create a unified pipeline for experimentation with multiple ACMs. Pipeline components:
* Preprocessing of images, ground truth masks, and initial contours
* Construction of ACM inputs and ACM execution
* Logging and saving results

The code base for these experiments is in the [acm folder of model_plus_acm branch](https://github.com/shairaalam19/medicalimagesegmentation/tree/model_plus_acm/acm). Structure:
* [Level_Set_ACM](https://github.com/shairaalam19/medicalimagesegmentation/tree/model_plus_acm/acm/Level_Set_ACM): full pipeline implementation.
* [scripts](https://github.com/shairaalam19/medicalimagesegmentation/tree/model_plus_acm/acm/scripts): scripts to set parameters and trigger experiments.
* [Results](https://github.com/shairaalam19/medicalimagesegmentation/tree/model_plus_acm/acm/Results): stored experiment outputs

Experiments are configured in the [scripts](https://github.com/shairaalam19/medicalimagesegmentation/tree/model_plus_acm/acm/scripts) folder, which call the pipeline’s core functions: `run_lsa` (runs DALS LSA) and `run_cv` (runs CV ACM). Configurable parameters include:
* Input image, ground truth segmentation, & initial contour
* Image preprocessing preferences
* ACM hyperparameters
* Output directory
* Frequency of saving intermediate ACM results

**Let's try a simple demo for Level Set ACM!**

1. Go to the correct [branch](https://github.com/shairaalam19/medicalimagesegmentation/tree/model_plus_acm).
2. `cd acm/scripts`  
3. `python unet_lsa_dals_brain_demo.py`
4. This would run two versions of the DALS level set acm (LSA) on a brain scan.
  Version 1: reproduces DALS demo results by using their provided input image, CNN generated initial contour, ground truth segmentation, and default hyperparameters ($\nu$=5, $\mu$=0.2, iterations=600).
  Version 2: The same configuration as above but uses different hyper-parameters ($\nu$=0, $\mu$=0, iterations=600).
5. In each output directory, you should be able to see all the intermediate segmentation masks for every 50 iterations of ACM, plot of evolution of IOU and DICE scores throughout the ACM iterations, and an image of initial and final contour overlayed on the input.

Results for Version 1 ($\nu$=5, $\mu$=0.2, iterations=600):
<p float="left">
  <img src="readme_images/acm_brain_demo.png" alt="ACM Contour Evolution" width="250"/>
  <img src="readme_images/acm_scores_brain_demo.png" alt="ACM Scores Progression" width="400"/>
</p>


Results for Version 2 ($\nu$=0, $\mu$=0, iterations=600):
<p float="left">
  <img src="readme_images/acm_brain_demo_2.png" alt="ACM Contour Evolution" width="250"/>
  <img src="readme_images/acm_scores_brain_demo_2.png" alt="ACM Scores Progression" width="400"/>
</p>

This demo shows that the effectiveness of ACM on an image is highly dependent on image-specific hyperparameter settings. Parameter $\mu$ controls the regularization strength, enforcing smoothness in the evolving contour and preventing noisy or irregular boundaries. Parameter $\nu$ controls the balloon force, which expands or shrinks the contour toward object boundaries. As an example, in lung lesion CT scans, the optimal values for these parameters may vary depending on the lesion characteristics:
* Small, well-defined nodules: A lower μ allows the contour to adhere closely to fine lesion boundaries, while a moderate ν helps drive the contour outward without overshooting into surrounding lung tissue.
* Large, diffuse lesions with weak boundaries: A higher μ is required to suppress noise and maintain contour smoothness, while a stronger ν is often necessary to push the contour across regions of low contrast and capture the full extent of the lesion.

*The current approach to setting these hyperparameters is manual trial and error which is infeasible. This motivates learning of ACM hyperparameters within neural networks.*

**Let's try a simple demo for Chan-Vese ACM!**

1. Go to the correct [branch](https://github.com/shairaalam19/medicalimagesegmentation/tree/model_plus_acm/acm).
2. `cd acm/scripts`  
3. `python cv_skin_lesion.py`
4. This would run a Chan Vese ACM on an image from ISIC dataset (ensure this image has been downloaded at correct path). It starts with a simple square in the middle of the image as the initial contour and has the hyperparameter settings: $\nu$=100, $\mu$=1, iterations=150.

<p float="left">
  <img src="readme_images/acm_skin_demo.png" alt="ACM Contour Evolution" width="250"/>
  <img src="readme_images/acm_scores_skin_demo.png" alt="ACM Scores Progression" width="300"/>
</p>

Refer to the [report](Graduate_Capstone_Report.pdf) for detailed analysis of the two ACMs across different medical images, preprocessing techniques, and contour & hyperparameter initializations. It also outlines the motivation for moving forward with level-set ACM (with grayscale preprocessing) and integrating it into a neural network for adaptive contour and hyperparameter settings. 

### Edge Segmentation Usage
Call edge-based segmentation mechanisms (Roberts/Sobel) using the following command in the [POC branch](https://github.com/shairaalam19/medicalimagesegmentation/tree/POC): 
```bash
python image_segmentation/edge_based.py
```

## Hybrid Model Architecture
Here is a diagram of our final neural network architecture incorporating a baseline CNN, attention layers, and ACMS. The encoder downsamples to extract features and reduces the spatial dimensions. The bottleneck further processes features and focuses on important features with more attention. The decoder upsamples features to construct an initial probability mask. Our main novelty is introducing an ”ACM Hyperparameter Generator” into the image segmentation CNN that can be trained to generate image-dependent optimal ACM hyperparameters for effective segmentation refinement. The model’s strong results on small datasets highlight its potential effectiveness, suggesting that it is well-suited for applications where high-quality segmentation is required on limited data samples. Future work can potentially focus on training efficiency with large datasets as backpropagating across ACM iterations is time-consuming.

![Hybrid Deep Learning Model for Medical Image Segmentation](readme_images/acm_plus_cnn.png)

## Model
### Running the final model
Fix the utils/config.json to your liking with various epochs, loss function criteria (IOU, DICE, BCE, etc), etc... 

Then run: 
```sh
python main.py
```

### Hybrid Model Demo

The [demo](demo) folder in the main branch has the following contents:
1. data folder -> consists of small train (8 images) and test (2 images) datasets from the Covid 19 CT Scan DataSet.
2. output folder -> to store trained model and test results from demo run.
3. config.json -> an example configuration. Some important specifications:
  - Desired functionality - use of pretrained model, training, and testing
  - Dataset properties & location
  - Enabling of acm and edge attention layers
  - [Pretrained baseline model](outputs/models/edge_attention_epoch_11.pth) location - a baseline model trained with edge attention and without acm for ~11 epochs on entire Covid 19 CT Scan DataSet. We start training our model with ACM on top of such a pretrained model.
  - 3 epochs, train batch size (4 images), test batch size (1 image)
  - Use of binary cross entropy loss (BCE) as the loss function
  - Output folder to store trained model and best and worst case test result masks.

This small demo was done to closely follow the training process on any local machine (w/o requiring GPU) and to ensure that ACM hyperparameters are being learned.
  
**Instructions**
Make sure you are in the [main branch source directory](https://github.com/shairaalam19/medicalimagesegmentation/tree/main).

```sh
cp demo/config.json utils/config.json
python main.py
```

**Terminal Output**

```
---------------------------------------------- TRAIN ----------------------------------------------
Created training session folder: demo/output/model/20250922_202702
Loading input dataset from: demo/data/train/input and targets from: demo/data/train/target
Loaded 8 images from dataset.
Using pretrained model...
Loading model information from: outputs/models/edge_attention_epoch_11.pth
Using Edge Attention Block
Using Active Contour Layer
Training model...
Saved models/EdgeSegmentationCNN.py to demo/output/model/20250922_202702/EdgeSegmentationCNN.py
Saved utils/config.json to demo/output/model/20250922_202702/config.json
Num_iters, nu, mu:  tensor(85., grad_fn=<UnbindBackward0>) tensor(7.9161, grad_fn=<UnbindBackward0>) tensor(0.1197, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(92., grad_fn=<UnbindBackward0>) tensor(8.4695, grad_fn=<UnbindBackward0>) tensor(0.1083, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(88., grad_fn=<UnbindBackward0>) tensor(8.0119, grad_fn=<UnbindBackward0>) tensor(0.1228, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(96., grad_fn=<UnbindBackward0>) tensor(9.2927, grad_fn=<UnbindBackward0>) tensor(0.0320, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(90., grad_fn=<UnbindBackward0>) tensor(8.4591, grad_fn=<UnbindBackward0>) tensor(0.0693, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(95., grad_fn=<UnbindBackward0>) tensor(9.0194, grad_fn=<UnbindBackward0>) tensor(0.0465, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(97., grad_fn=<UnbindBackward0>) tensor(9.3894, grad_fn=<UnbindBackward0>) tensor(0.0216, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(93., grad_fn=<UnbindBackward0>) tensor(8.8983, grad_fn=<UnbindBackward0>) tensor(0.0572, grad_fn=<UnbindBackward0>)
Epoch 1/3, Loss: 0.4112
Learning rate after epoch 1: [0.001]
Model saved at: demo/output/model/20250922_202702/epoch_1.pth
Num_iters, nu, mu:  tensor(97., grad_fn=<UnbindBackward0>) tensor(9.5147, grad_fn=<UnbindBackward0>) tensor(0.0151, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(99., grad_fn=<UnbindBackward0>) tensor(9.7731, grad_fn=<UnbindBackward0>) tensor(0.0033, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(95., grad_fn=<UnbindBackward0>) tensor(9.2844, grad_fn=<UnbindBackward0>) tensor(0.0257, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(93., grad_fn=<UnbindBackward0>) tensor(9.0064, grad_fn=<UnbindBackward0>) tensor(0.0304, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(94., grad_fn=<UnbindBackward0>) tensor(9.2934, grad_fn=<UnbindBackward0>) tensor(0.0157, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(99., grad_fn=<UnbindBackward0>) tensor(9.8654, grad_fn=<UnbindBackward0>) tensor(0.0013, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(91., grad_fn=<UnbindBackward0>) tensor(9.0038, grad_fn=<UnbindBackward0>) tensor(0.0308, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(98., grad_fn=<UnbindBackward0>) tensor(9.6747, grad_fn=<UnbindBackward0>) tensor(0.0066, grad_fn=<UnbindBackward0>)
Epoch 2/3, Loss: 0.3523
Learning rate after epoch 2: [0.001]
Model saved at: demo/output/model/20250922_202702/epoch_2.pth
Num_iters, nu, mu:  tensor(91., grad_fn=<UnbindBackward0>) tensor(9.1331, grad_fn=<UnbindBackward0>) tensor(0.0266, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(99., grad_fn=<UnbindBackward0>) tensor(9.9032, grad_fn=<UnbindBackward0>) tensor(0.0010, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(100., grad_fn=<UnbindBackward0>) tensor(9.9637, grad_fn=<UnbindBackward0>) tensor(0.0001, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(98., grad_fn=<UnbindBackward0>) tensor(9.7841, grad_fn=<UnbindBackward0>) tensor(0.0029, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(100., grad_fn=<UnbindBackward0>) tensor(9.9743, grad_fn=<UnbindBackward0>) tensor(6.4373e-05, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(98., grad_fn=<UnbindBackward0>) tensor(9.8396, grad_fn=<UnbindBackward0>) tensor(0.0017, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(97., grad_fn=<UnbindBackward0>) tensor(9.6808, grad_fn=<UnbindBackward0>) tensor(0.0044, grad_fn=<UnbindBackward0>)
Num_iters, nu, mu:  tensor(99., grad_fn=<UnbindBackward0>) tensor(9.8948, grad_fn=<UnbindBackward0>) tensor(0.0008, grad_fn=<UnbindBackward0>)
Epoch 3/3, Loss: 0.3471
Learning rate after epoch 3: [0.001]
Model saved at: demo/output/model/20250922_202702/epoch_3.pth
Training complete!
Final epoch: epoch_3
Model saved at: demo/output/model/20250922_202702/epoch_3.pth
Loss graph saved at: demo/output/model/20250922_202702/Training Loss per Batch
----------------------------------------------- TEST -----------------------------------------------
Loading input dataset from: demo/data/test/input and targets from: demo/data/test/target
Loaded 2 images from dataset.
Testing the model epoch_3 in demo/output/model/20250922_202702 and saving before and after images in demo/output/test_results/20250922_202702/epoch_3
Model saved at: demo/output/test_results/20250922_202702/epoch_3/model.pth
Num_iters, nu, mu:  tensor(99.) tensor(9.9449) tensor(0.0003)
Num_iters, nu, mu:  tensor(99.) tensor(9.9466) tensor(0.0003)
Saved all test metrics in demo/output/test_results/20250922_202702/epoch_3/test_metrics.json
Testing complete! Images saved to demo/output/test_results/20250922_202702/epoch_3.
```

**Output Description**
The first part depicts the training process. You can see that 3 epochs occur and in each epoch the 8 training images go through a forwards pass and the loss is backpropagated and recorded. The terminal output shows print statements of the ACM parameters (num_iters, $\nu$, $\mu$) generated for each image in the forward pass. Backpropagating over these ACM iterations is computationally intensive, which is why using an initial pretrained baseline model increases efficiency and accuracy. Once training completes the model for each epoch is stored under a timestamp subdirectory in the [demo model output folder](demo/output/model). Also a plot of the training loss over epochs is saved.

The second part depicts the testing process. The model from the final epoch is tested on the test images. The resulting lesion masks and test metrics are saved in a 'timestamp/epoch' subdirectory of the [demo test output folder](demo/output/test_results).

Attached are the training losses over epochs and the results of testing the final model on the two test images. 'a' represents intensity image, 'b' represents the predicted segmentation output of the trained model, and 'c' represents ground truth segmentation. The overall test metrics are also displayed.

<img src="readme_images/training_loss_per_batch.png" alt="Training loss per epoch" width="400"/>

<img src="readme_images/bjorke_9.png" alt="Test 1" width="500"/>
<img src="readme_images/bjorke_10.png" alt="Test 2" width="500"/>

```
"overall_metrics": {
        "Average Test Loss": 0.30593057721853256,
        "AUROC": 0.9650019157611955,
        "AUC": 0.9650019157611955,
        "Precision": 0.8577514523294225,
        "Recall (Sensitivity)": 0.8358928005074532,
        "F1 Score": 0.8466810694465638,
        "IoU": 0.7277814455245858,
        "Dice Score": 0.8422963248452655
    }
```

These are initial proof of concept demo results. The model’s strong performance on small datasets demonstrates its potential for high-quality segmentation when data is limited. For larger datasets, a more computationally efficient backbone (e.g., a CNN with attention layers) can first be trained extensively, after which an ACM hyperparameter layer can be integrated and fine-tuned on smaller subsets where precise segmentation is critical.

The [report](Graduate_Capstone_Report.pdf) contains results and analysis after more thorough training. Moreover, our [ICCV Submission](ICCV.pdf) is a more condensed version of our report.