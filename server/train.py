import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from torch import optim

from utils.utils import load_config
from models.EdgeSegmentationCNN import EdgeSegmentationCNN 
from models.EdgeSegmentationLoss import EdgeSegmentationLoss 
from models.utils import load_model, train_model, test_model
from datasets.utils import load_dataset, split_dataset

def main():
    # Load Configurations
    config = load_config()

    # Load dataset and model paths from config
    save_model_folder = config["SAVE_MODEL_FOLDER"]  # Where trained models are saved
    model_output_folder = config["MODEL_OUTPUT_FOLDER"]  # Where test results are stored

    os.makedirs(save_model_folder, exist_ok=True)
    os.makedirs(model_output_folder, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    data = load_dataset(input_folder_path=config["INPUT_DATASET_FOLDER"], target_folder_path=config["TARGET_DATASET_FOLDER"], dataset_size=config["MODEL_DATASET_SIZE"])
    
    # Split dataset for training and testing
    train_dataset, test_dataset = split_dataset(data)

    print("Initializing DataLoader...")
    train_data = DataLoader(train_dataset, batch_size=config["TRAINING_BATCH_SIZE"], shuffle=True)

    # Load or create model
    if config["PRETRAINING"]:
        print("Loading pretrained model...")
        if not config["MODEL_NAME"]:
            raise ValueError("No pretrained model provided.")
        model = load_model(config["LOAD_MODEL_FOLDER"], config["MODEL_NAME"])
    else:
        print("Creating new model...")
        model = EdgeSegmentationCNN(
            edge_attention=config["EDGE_ATTENTION"], 
            define_edges_before=config["DEFINE_EDGES_BEFORE"], 
            define_edges_after=config["DEFINE_EDGES_AFTER"]
        )

    # Define loss function and optimizer
    criterion = EdgeSegmentationLoss(
        bce=config["BCE_LOSS"], 
        composite=config["COMPOSITE_LOSS"], 
        iou=config["IOU_LOSS"], 
        dice=config["DICE_LOSS"]
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    # Train model
    print("Training the model...")
    train_model(model, train_data, criterion, optimizer)

    # Save the trained model
    # model_save_path = os.path.join(save_model_folder, "trained_model.pth")
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Model saved to {model_save_path}")

    # Run testing and save results
    print("Testing the model...")
    test_data = DataLoader(test_dataset, batch_size=config["TEST_BATCH_SIZE"])
    test_model(model, test_data, model_output_folder, "test_results.txt", criterion)

if __name__ == "__main__":
    main()
