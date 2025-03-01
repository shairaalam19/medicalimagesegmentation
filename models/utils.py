import os
import sys
import torch
import datetime
import shutil
import json
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, jaccard_score

from utils.utils import load_config
from models.EdgeSegmentationCNN import EdgeSegmentationCNN 
from models.EdgeSegmentation import get_edges
from datasets.utils import save_combined_image

# Load Configurations
config = load_config()

# -----------------------------------------------------------
# Training Function
# -----------------------------------------------------------
def train_model(model, train_loader, criterion, optimizer):
    print("Training model...")
    # Create a timestamped folder for this training session
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_folder = save_model_folder(timestamp)

    # Save model file 
    save_file("models/EdgeSegmentationCNN.py", model_folder)
    save_file("utils/config.json", model_folder)

    # Start training 
    model.train() # enables features liek dropout or batch noramlization 

    # Apply Learning Rate Scheduler (Reduce LR on Plateau)
    scheduler = lr_scheduler.ReduceLROnPlateau( # lowers learning rate when loss stops improving
        optimizer,         # Optimizer to adjust learning rate
        mode='min',        # Look for a *decrease* in the monitored metric
        factor=0.5,        # Reduce LR by a factor of 0.5 (i.e., new LR = old LR * 0.5)
        patience=3,        # Wait for 3 epochs before reducing LR if no improvement
        threshold=1e-4,    # Minimum change in metric to qualify as an improvement
        verbose=True       # Print messages when LR is reduced
    )
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # reduces learning rate every few epochs 
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # Reduce LR by 10% every epoch

    # Early Stopping Parameters
    early_stop_patience = 7  # Stop if no significant improvement in 7 epochs
    best_loss = float('inf')
    epochs_no_improve = 0

    # Apply Gradient Clipping to Prevent Exploding Gradients
    clip_value = 1.0 

    epoch_losses = []
    last_epoch = None

    for epoch in range(config["EPOCHS"]): # iterates through the number of epochs 
        running_loss = 0.0
        for inputs, targets, __ in train_loader:
            # Forward pass
            outputs = model(inputs) # computes model's current predictions for the given input 

            # Calculate reconstruction loss (unsupervised learning)
            loss = criterion(outputs, targets)  # Compare output with the target 

            # Backward pass and optimization
            optimizer.zero_grad() # clears any previously accumulated gradients (updates based only on current mini batch)
            loss.backward() # gradients of the loss wrt model's parameters using back propagation

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Apply gradient clipping when weight updates become too large 

            optimizer.step() # updates model's parameters using calculated gradients (via optimizer)

            # Accumulate loss
            running_loss += loss.item() # loss.item() is scalar loss value of current mini batch (not tensor) and accumulates total loss for epoch 

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        # Log the loss per epoch
        print(f"Epoch {epoch + 1}/{config['EPOCHS']}, Loss: {epoch_loss:.4f}")

        # Adjust learning rate based on loss
        scheduler.step(epoch_loss) # based on what the epoch loss is (rate of change)
        # scheduler.step() # just constant scheduling 

        # Early stopping based on training loss
        if epoch_loss < best_loss - 1e-4:  # Significant improvement
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        last_epoch = epoch
        epoch_losses.append(epoch_loss)  # Store the loss for plotting

        save_model(model, f"epoch_{epoch + 1}", model_folder)

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Training stopped.")
            break

    print("Training complete!")

    if last_epoch is not None:
        model_name = f"epoch_{last_epoch + 1}"
        print(f"Final epoch: {model_name}")
        save_model(model, model_name, model_folder)
    else: 
        print("No valid epoch found. Model not saved.")
        return None, None

    save_loss_graph(epoch_losses, model_folder, title="Training Loss per Batch", file_name="training_loss.png")

    return model_folder, model_name

# -----------------------------------------------------------
# Save Model Function
# -----------------------------------------------------------
def save_model(model, model_name, folder_path):
    """Saves the model state to a file in the specified folder."""
    # Ensure the model name has a .pth extension
    if not model_name.endswith(".pth"):
        model_name += ".pth"
    model_save_path = os.path.join(folder_path, model_name)

    torch.save(model.state_dict(), model_save_path)

    print(f"Model saved at: {model_save_path}")

# -----------------------------------------------------------
# Save File Function
# -----------------------------------------------------------
def save_file(file, folder_path):
    """Copies the specified model file to the session folder."""
    if os.path.exists(file):
        destination = os.path.join(folder_path, os.path.basename(file))
        shutil.copy(file, destination)
        print(f"Saved {file} to {destination}")
    else:
        print(f"Warning: {file} not found. Skipping model script backup.")

# -----------------------------------------------------------
# Save Model Folder Function
# -----------------------------------------------------------
def save_model_folder(folder_name):
    # Generates folder with folder name 
    folder_path = os.path.join(config["SAVE_MODEL_FOLDER"], folder_name)
    os.makedirs(folder_path, exist_ok=True)

    print(f"Created training session folder: {folder_path}")

    return folder_path

# -----------------------------------------------------------
# Save Loss Graph Function
# -----------------------------------------------------------
def save_loss_graph(batch_losses, output_folder, title, file_name):
    """Save the graph of loss vs. batches."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(batch_losses) + 1), batch_losses, marker='o', label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    loss_graph_path = os.path.join(output_folder, file_name)
    plt.savefig(loss_graph_path)
    plt.close()
    print(f"Loss graph saved at {loss_graph_path}")

# -----------------------------------------------------------
# Load Model Function
# -----------------------------------------------------------
def load_model(model_path, model_name):
    model_path = os.path.join(model_path, model_name)

    print(f"Loading model from: {model_path}")

    # Instantiating model architecture 
    model = EdgeSegmentationCNN(edge_attention=config["EDGE_ATTENTION"], define_edges_before=config["DEFINE_EDGES_BEFORE"], define_edges_after=config["DEFINE_EDGES_AFTER"])

    # Loading saved model onto model architecture 
    model.load_state_dict(torch.load(model_path))

    return model

# -----------------------------------------------------------
# Test Function
# -----------------------------------------------------------
def test_model(model, test_loader, model_path, model_name, criterion):
    # Extract timestamp and epoch information from the model path
    timestamp = os.path.basename(model_path)
    epoch = model_name.split(".")[0]  # Extract the epoch number (e.g., epoch_20)

    # Create output folder structure: config["MODEL_OUTPUT_FOLDER"]/{timestamp}/epoch_{}
    output_folder = os.path.join(config["MODEL_OUTPUT_FOLDER"], timestamp, f"{epoch}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Testing the model {model_name} in {model_path} and saving before and after images in {output_folder}")
    save_model(model, f"model", output_folder)

    # Loss tracking
    batch_losses = []
    total_loss = 0.0

    # Store outputs and targets for metric computation
    all_targets = []
    all_outputs = []
    all_file_names = []

    model.eval()
    with torch.no_grad(): # disables gradient computation for computational efficiency
        for i, (inputs, targets, file_names) in enumerate(test_loader):
            outputs = model(inputs) # forward pass to generate predictions 

            # Future visualization and calculations 
            # Calculate the loss
            loss = criterion(outputs, targets)  
            total_loss += loss.item()
            batch_losses.append(loss.item())

            # Convert tensors to numpy for metric computation
            outputs_np = torch.sigmoid(outputs).cpu().numpy()  # Convert logits to probabilities
            targets_np = targets.cpu().numpy()
            
            # Store for overall metrics
            all_outputs.append(outputs_np)
            all_targets.append(targets_np)
            all_file_names.extend(file_names)

            # Save individual results
            for j, file_name in enumerate(file_names):
                original_filename = os.path.basename(file_name)
                output_filename = os.path.join(output_folder, f"{original_filename}")
                save_combined_image(inputs[j], outputs[j], targets[j], output_filename)
    
    # Calculate average loss
    average_loss = total_loss / len(test_loader)

    compute_metrics(all_targets=all_targets, all_outputs=all_outputs, all_file_names=all_file_names, batch_losses=batch_losses, average_loss=average_loss, output_folder=output_folder)

    print(f"Testing complete! Images saved to {output_folder}.")

# -----------------------------------------------------------
# Demo Function
# -----------------------------------------------------------
def demo_model(demo_loader): 
    print("Demoing model...")

    for input, target, file_name in demo_loader:
        # Ensure image is a tensor
        image = image.unsqueeze(0) if image.ndimension() == 3 else image  # Add batch dimension if necessary

        # Process the image
        output = get_edges(image)

        # Convert outputs to NumPy arrays for visualization
        original_image = image.squeeze().cpu().numpy()
        edges_image = output.squeeze().cpu().numpy()

        # Ensure both images are 2D (either original or edges)
        if original_image.ndim == 2:  # If the image is already 2D (1024, 1024)
            original_image = original_image
        else:
            original_image = original_image[0]  # Select the first channel if 3D

        # Ensure the edges image has the correct shape
        if edges_image.ndim == 3 and edges_image.shape[0] > 1:
            # If the output has more than one channel, select the first channel (edge detection)
            edges_image = edges_image[0]

        print("Displaying image. Exit out of popup whenever you want to end execution...")

        # Display the original and processed images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Original Image: {file_name}")
        plt.imshow(original_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Edge Detection (Filtered)")
        plt.imshow(edges_image, cmap="gray")
        plt.axis("off")

        plt.show()

    print("Demo Complete!")

import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, jaccard_score

def compute_metrics(all_targets, all_outputs, all_file_names, batch_losses, average_loss, output_folder, threshold=0.5):
    """
    Compute performance metrics for binary segmentation on a **per-image** basis 
    and for the entire test dataset.
    
    Args:
        all_targets (list of numpy arrays): List of ground truth binary masks.
        all_outputs (list of numpy arrays): List of model-predicted probabilities.
        all_file_names (list): List of file names corresponding to test images.
        batch_losses (list): List of batch losses.
        average_loss (float): Overall test loss.
        output_folder (str): Path to save results.
        threshold (float): Threshold for converting probabilities to binary predictions.

    Returns:
        None (Saves metrics to JSON files).
    """
    all_metrics = {}

    # Compute metrics for each image
    for idx, file_name in enumerate(all_file_names):
        target = (all_targets[idx].flatten() > threshold).astype(int)  # Convert to binary mask
        output = all_outputs[idx].flatten()  # Model outputs (probabilities)
        binary_output = (output > threshold).astype(int)  # Convert to binary mask

        try:
            auroc = roc_auc_score(target, output)
        except ValueError:
            auroc = None  # Handle case where only one class exists in the target

        # Compute classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(target, binary_output, average='binary', zero_division=0)
        iou = jaccard_score(target, binary_output, average='binary', zero_division=0)
        dice = (2 * iou) / (1 + iou) if iou > 0 else 0  # Avoid division by zero

        image_metrics = {
            "AUROC": auroc,
            "AUC": auroc,  
            "Precision": precision,
            "Recall (Sensitivity)": recall,
            "F1 Score": f1,
            "IoU": iou,
            "Dice Score": dice
        }
        
        # Save individual image metrics
        image_metrics_path = os.path.join(output_folder, f"{os.path.basename(file_name)}_metrics.json")
        with open(image_metrics_path, "w") as f:
            json.dump(image_metrics, f, indent=4)

        all_metrics[file_name] = image_metrics

    # Compute overall metrics (aggregate across all images)
    all_outputs_flat = np.concatenate(all_outputs, axis=0).flatten()
    all_targets_flat = (np.concatenate(all_targets, axis=0).flatten() > threshold).astype(int)  # Ensure binary targets
    binary_outputs = (all_outputs_flat > threshold).astype(int)

    # Overall AUROC
    try:
        overall_auroc = roc_auc_score(all_targets_flat, all_outputs_flat)
    except ValueError:
        overall_auroc = None  # Handle case where only one class exists in the targets

    # Overall Precision, Recall, F1, IoU, Dice
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_targets_flat, binary_outputs, average='binary', zero_division=0
    )
    overall_iou = jaccard_score(all_targets_flat, binary_outputs, average='binary', zero_division=0)
    overall_dice = (2 * overall_iou) / (1 + overall_iou) if overall_iou > 0 else 0

    overall_metrics = {
        "Average Test Loss": average_loss,
        "AUROC": overall_auroc,
        "AUC": overall_auroc,
        "Precision": overall_precision,
        "Recall (Sensitivity)": overall_recall,
        "F1 Score": overall_f1,
        "IoU": overall_iou,
        "Dice Score": overall_dice
    }

    # Save overall metrics
    overall_metrics_path = os.path.join(output_folder, "overall_test_metrics.json")
    with open(overall_metrics_path, "w") as f:
        json.dump(overall_metrics, f, indent=4)

    # Save loss graph
    save_loss_graph(batch_losses, output_folder, title="Test Loss per Batch", file_name="test_loss.png")

    print(f"Saved individual and overall test metrics to {output_folder}")

