# Cluster
https://www.hoffman2.idre.ucla.edu/Accounts/Requesting-an-account.html 

Confirmation page: https://sim.idre.ucla.edu/sim/home 

# Lab Cluster
```bash 
ssh salam@131.179.124.56
```

To move from local to ssh: 
```bash
cd medicalimagesegmentation/data
scp covid19-ct-scan-lesion-segmentation-dataset.zip salam@131.179.124.56:/CGLab/salam/medicalimagesegmentation/data
```

## GPUs 
Message from Zhi: 
Contact if you wish to install system level package.

Please store all files under the user home folder ~/ , and keep your home folder size under 500GB.

The server is equipped with two A6000 GPUs. If one GPU is in use, you can specify which GPU to use by setting the environment variable in your command, for example:
- For GPU 0: CUDA_VISIBLE_DEVICES=0 python main.py
- For GPU 1: CUDA_VISIBLE_DEVICES=1 python main.py

### Check Utilization
Check utilization of each GPU here: 
```bash
nvidia-smi
```

# AWS Sagemaker
## Package Your Code
SageMaker requires an organized script structure. Based on your project structure, you should prepare a train.py script that SageMaker can call.

Move main.py to train.py or modify it to accept command-line arguments.
Ensure all imports work relative to the script.

## Upload Data to S3
SageMaker expects data in S3. You need to upload data/ to an S3 bucket.

```bash 
aws s3 cp data/ s3://your-bucket-name/data/ --recursive
```

Replace your-bucket-name with an actual S3 bucket.

## Create a SageMaker Training Script
Modify train.py to:
- Read datasets from datasets/
- Load models from models/
- Save outputs to outputs/
- Read input arguments for hyperparameters