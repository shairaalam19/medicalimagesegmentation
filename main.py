import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

from utils.utils import load_config
from models.EdgeSegmentationCNN import EdgeSegmentationCNN 
from models.EdgeSegmentationLoss import EdgeSegmentationLoss 
from models.utils import save_model, load_model, load_model_pretrained, test_model, train_model, demo_model
from datasets.utils import load_dataset, split_dataset, save_combined_image
from datasets.EdgeSegmentationDataset import EdgeSegmentationDataset

# -----------------------------------------------------------
# Main Execution
# -----------------------------------------------------------
def main():
    # Load Configurations
    config = load_config()

    if config["DEMO"]: 
        print(f"{('-' * ((100 - len('DEMO') - 2) // 2))} DEMO {('-' * ((100 - len('DEMO') - 2) // 2))}")
        # Preprocess 
        data = load_dataset(input_folder_path=config["INPUT_DATASET_FOLDER"], target_folder_path=config["TARGET_DATASET_FOLDER"], dataset_size=config["DEMO_DATASET_SIZE"]) # Creates a dataset class that contains the preprocessed dataset
        demo_model(data)

    model, model_folder, model_name, train_dataset, test_dataset, target_dataset = None, None, None, None, None, None

    # Load dataset 
    if config["TRAIN"] or config["TEST"]: 
        print(f"{('-' * ((100 - len('LOAD DATASET') - 2) // 2))} LOAD DATASET {('-' * ((100 - len('LOAD DATASET') - 2) // 2))}")
        data = load_dataset(input_folder_path=config["INPUT_DATASET_FOLDER"], target_folder_path=config["TARGET_DATASET_FOLDER"], dataset_size=config["MODEL_DATASET_SIZE"])
    
    # Split dataset for training and testing 
    if config["TRAIN"] and config["TEST"]: 
        print(f"{('-' * ((100 - len('SPLIT DATASET') - 2) // 2))} SPLIT DATASET {('-' * ((100 - len('SPLIT DATASET') - 2) // 2))}")
        train_dataset, test_dataset = split_dataset(data) # splits the dataset

    # print(len(data), len(train_dataset), len(test_dataset))
    # sys.exit()

    if config["TRAIN"]: 
        print(f"{('-' * ((100 - len('TRAIN') - 2) // 2))} TRAIN {('-' * ((100 - len('TRAIN') - 2) // 2))}")
        if train_dataset is None: 
            train_dataset = data
        
        train_data = DataLoader(train_dataset, batch_size=config["TRAINING_BATCH_SIZE"], shuffle=True) 

        if config["PRETRAINING"]: 
            print("Using pretrained model...")
            # no pretrained model provided 
            if config["MODEL_NAME"] == None or config["MODEL_NAME"] == "":
                print("No pretrained model provided. Exiting")
                exit()
            # load pretrained model 
            else: 
                model = load_model_pretrained(config["LOAD_MODEL_FOLDER"], config["MODEL_NAME"])

                # Optionally, freeze some layers if you don't want to train the whole model
                # for param in model.encoder.parameters():  # Example: freezing encoder layers
                #     param.requires_grad = False
        
        # no pretrained model provided 
        else: 
            print("Creating new model...")
            # Model Creation
            model = EdgeSegmentationCNN(edge_attention=config["EDGE_ATTENTION"], define_edges_before=config["DEFINE_EDGES_BEFORE"], define_edges_after=config["DEFINE_EDGES_AFTER"], use_acm=config["ACM"])
        
        # Measures how well the model's predictions match the true labels (used in the backward pass for gradient computation).
        criterion = EdgeSegmentationLoss(bce=config["BCE_LOSS"], composite=config["COMPOSITE_LOSS"], iou=config["IOU_LOSS"], dice=config["DICE_LOSS"])
        
        # Updates the model's parameters to minimize the loss based on the computed gradients (used in the optimization step).        
        optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"]) # model.parameters are the parameters of the model and lr is the step size of learning

        # Fine-tuning / Training the model on the new dataset
        model_folder, model_name = train_model(model, train_data, criterion, optimizer)
        
    if config["TEST"]: 
        print(f"{('-' * ((100 - len('TEST') - 2) // 2))} TEST {('-' * ((100 - len('TEST') - 2) // 2))}")
        if test_dataset is None: 
            test_dataset = data
        test_data = DataLoader(test_dataset, batch_size=config["TEST_BATCH_SIZE"])

        # Load Model
        if model is None and not config["TRAIN"]:
            model_folder = config["LOAD_MODEL_FOLDER"]
            model_name = config["MODEL_NAME"]
            model = load_model(model_folder, model_name)

        elif model is None and config["TRAIN"]:
            print("No model is available to test. ")

        criterion = EdgeSegmentationLoss(bce=config["BCE_LOSS"], composite=config["COMPOSITE_LOSS"], iou=config["IOU_LOSS"], dice=config["DICE_LOSS"])

        test_model(model, test_data, model_folder, model_name, criterion) 

if __name__ == "__main__":
    main()
