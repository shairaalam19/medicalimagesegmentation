# Cluster
https://www.hoffman2.idre.ucla.edu/Accounts/Requesting-an-account.html 

Confirmation page: https://sim.idre.ucla.edu/sim/home 

# Lab Cluster
ssh salam@131.179.124.56

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