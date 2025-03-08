import os
import sys
import torch
import datetime
import shutil
import json
import re
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
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
def train_model(model, train_loader, criterion, optimizer, model_folder):
    print("Training model...")
    
    # Save model file 
    save_file("models/EdgeSegmentationCNN.py", model_folder)
    save_file("utils/config.json", model_folder)
    # torch.autograd.set_detect_anomaly(True)

    # Start training 
    model.train() # enables features liek dropout or batch noramlization 

    # Apply Learning Rate Scheduler (Reduce LR on Plateau)
    scheduler = lr_scheduler.ReduceLROnPlateau( # lowers learning rate when loss stops improving
        optimizer,         # Optimizer to adjust learning rate
        mode='min',        # Look for a *decrease* in the monitored metric
        factor=0.5,        # Reduce LR by a factor of 0.5 (i.e., new LR = old LR * 0.5)
        patience=3,        # Wait for 3 epochs before reducing LR if no improvement
        threshold=1e-4,    # Minimum change in metric to qualify as an improvement
    )
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # reduces learning rate every few epochs 
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)  # Reduce LR by 10% every epoch

    # Early Stopping Parameters
    early_stop_patience = config["PATIENCE"]  # Stop if no significant improvement in 7 epochs
    best_loss = float('inf')
    epochs_no_improve = 0

    # Apply Gradient Clipping to Prevent Exploding Gradients
    clip_value = 1.0 

    # epoch_losses = []
    # last_epoch = None

    # Prepare loss tracking file
    loss_file_path = os.path.join(model_folder, "epoch_losses.txt")

    # Open file in append mode (to prevent loss in case of shutdown)
    with open(loss_file_path, "a") as loss_file:
        for epoch in range(config["EPOCHS"]): # iterates through the number of epochs 
            running_loss = 0.0
            for inputs, targets, __ in train_loader:
                optimizer.zero_grad() # clears any previously accumulated gradients (updates based only on current mini batch)
                
                # Forward pass
                outputs = model(inputs) # computes model's current predictions for the given input 

                # Calculate reconstruction loss (unsupervised learning)
                loss = criterion(outputs, targets)  # Compare output with the target 

                # Backward pass and optimization
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

            learning_rate = scheduler.get_last_lr()
            print(f"Learning rate after epoch {epoch + 1}: {learning_rate}")

            # Save the epoch loss to the file after each epoch
            loss_file.write(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.6f}, Learning Rate: {learning_rate}\n")
            loss_file.flush()  # Ensure data is written immediately

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
# Save Loss Graph Function
# -----------------------------------------------------------
def save_loss_graph(loss_file_path, output_folder, title):
    # Read epoch losses from the file
    epochs = []
    losses = []
    
    with open(loss_file_path, "r") as file:
        for line in file:
            match = re.search(r"Epoch: (\d+), Loss: ([\d\.]+)", line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                epochs.append(epoch)
                losses.append(loss)

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o", linestyle="-", color="b", label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save plot
    output_path = os.path.join(output_folder, title)
    plt.savefig(output_path)
    plt.close()

    print(f"Loss graph saved at: {output_path}")
# -----------------------------------------------------------
# Load Model Function
# -----------------------------------------------------------
def load_model(model_path, model_name):
    model_path = os.path.join(model_path, model_name)
    print(f"Loading model from: {model_path}")
    model = EdgeSegmentationCNN(edge_attention=config["EDGE_ATTENTION"], define_edges_before=config["DEFINE_EDGES_BEFORE"], define_edges_after=config["DEFINE_EDGES_AFTER"], use_acm=config["ACM"])
    model.load_state_dict(torch.load(model_path))
    return model

def load_model_pretrained(model_path, model_name):
    
    pretrained_model_path = os.path.join(model_path, model_name)
    print(f"Loading model information from: {pretrained_model_path}")
    pretrained_state_dict = torch.load(pretrained_model_path)

    model = EdgeSegmentationCNN(edge_attention=config["EDGE_ATTENTION"], define_edges_before=config["DEFINE_EDGES_BEFORE"], define_edges_after=config["DEFINE_EDGES_AFTER"], use_acm=config["ACM"]) # Latest architecture
    #model = EdgeSegmentationCNN(edge_attention=config["EDGE_ATTENTION"]) # Latest architecture
    # Get the state dict of the current model architecture (ex: with acm layers added)
    model_state_dict = model.state_dict()

    # Merge weights from the pretrained model
    merged_state_dict = {}
    for name, param in model_state_dict.items():
        if name in pretrained_state_dict and pretrained_state_dict[name].size() == param.size():
            merged_state_dict[name] = pretrained_state_dict[name]
        else:
            merged_state_dict[name] = param  # Keep new layers initialized as they are

    # Load the merged state dict into the model
    model.load_state_dict(merged_state_dict)

    return model

# -----------------------------------------------------------
# Test Function
# -----------------------------------------------------------
def test_model(model, test_loader, model_path, model_name, criterion):
    # Extract timestamp and epoch information from the model path
    timestamp = os.path.basename(model_path)
    epoch = model_name.split(".")[0]  # Extract the epoch number (e.g., epoch_20)

    # Create output folder structure: config["TEST_RESULTS_FOLDER"]/{timestamp}/epoch_{}
    output_folder = os.path.join(config["TEST_RESULTS_FOLDER"], timestamp, f"{epoch}")
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
            # outputs_np = torch.sigmoid(outputs).cpu().numpy()  # Convert logits to probabilities
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Store for overall metrics
            all_outputs.append(outputs_np)
            all_targets.append(targets_np)
            all_file_names.extend(file_names)

            # Save individual results
            for j, file_name in enumerate(file_names):
                original_filename = os.path.basename(file_name)
                output_filename = os.path.join(output_folder, f"{original_filename}")

                # fix this output to print the binary
                # output = all_outputs[idx].flatten()  # Model outputs (probabilities)
                # binary_output = (output > threshold).astype(int)  # Convert to binary mask

                save_combined_image(inputs[j], torch.round(outputs[j]), targets[j], output_filename)
    
    # Calculate average loss
    average_loss = total_loss / len(test_loader)

    # Threshold the outputs so it can turn it into a mask 

    compute_metrics(all_targets=all_targets, all_outputs=all_outputs, all_file_names=all_file_names, batch_losses=batch_losses, average_loss=average_loss, output_folder=output_folder)

    print(f"Testing complete! Images saved to {output_folder}.")


# -----------------------------------------------------------
# Demo Function
# -----------------------------------------------------------
def demo_model(demo_loader, output_folder): 
    print("Demoing model...")

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for input, target, file_name in demo_loader:
        # Ensure input is a tensor and rename it to image for clarity
        image = input  # Assign input to image

        # Ensure image has a batch dimension
        image = image.unsqueeze(0) if image.ndimension() == 3 else image  

        # Process the image
        output = get_edges(image)

        # Convert outputs to NumPy arrays
        original_image = image.squeeze().cpu().detach().numpy()
        edges_image = output.squeeze().cpu().detach().numpy()
        target_image = target.squeeze().cpu().detach().numpy()

        # Ensure all images are at least 3D for consistency
        if original_image.ndim == 2:
            original_image = original_image[:, :, np.newaxis]
        if edges_image.ndim == 2:
            edges_image = edges_image[:, :, np.newaxis]
        if target_image.ndim == 2:
            target_image = target_image[:, :, np.newaxis]

        images = [original_image, edges_image, target_image]
        titles = ["(1) Input", "(2) Model Output Image", "(3) Target Image"]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        
        # Define the output file path
        save_path = os.path.join(output_folder, f"{file_name}_combined.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Saved combined image: {save_path}")

    print("Demo Complete!")


import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

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
        None (Saves all metrics in a single JSON file).
    """
    metrics_dict = {}
    iou_list = []
    dice_list = []

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

        # Compute Intersection and Union
        intersection = np.logical_and(target, binary_output).sum()  # True Positives
        union = np.logical_or(target, binary_output).sum()  # Union of both masks

        # Compute IoU and Dice coefficient
        iou = intersection / union if union > 0 else 0
        dice = (2 * intersection) / (np.sum(target) + np.sum(binary_output)) if (np.sum(target) + np.sum(binary_output)) > 0 else 0

        iou_list.append(iou)
        dice_list.append(dice)

        metrics_dict[file_name] = {
            "AUROC": auroc,
            "AUC": auroc,  
            "Precision": precision,
            "Recall (Sensitivity)": recall,
            "F1 Score": f1,
            "IoU": iou,
            "Dice Score": dice
        }

    # Compute overall metrics (aggregate across all images)
    all_outputs_flat = np.concatenate([arr.flatten() for arr in all_outputs])
    all_targets_flat = (np.concatenate([arr.flatten() for arr in all_targets]) > threshold).astype(int)  # Ensure binary targets
    binary_outputs = (all_outputs_flat > threshold).astype(int)

    # Overall AUROC
    try:
        overall_auroc = roc_auc_score(all_targets_flat, all_outputs_flat)
    except ValueError:
        overall_auroc = None  # Handle case where only one class exists in the targets

    # Overall Precision, Recall, F1
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_targets_flat, binary_outputs, average='binary', zero_division=0
    )

    # Compute IoU and Dice as averages of per-image values
    overall_iou = np.mean(iou_list) if iou_list else 0
    overall_dice = np.mean(dice_list) if dice_list else 0

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save overall metrics
    metrics_dict["overall_metrics"] = {
        "Average Test Loss": average_loss,
        "AUROC": overall_auroc,
        "AUC": overall_auroc,
        "Precision": overall_precision,
        "Recall (Sensitivity)": overall_recall,
        "F1 Score": overall_f1,
        "IoU": overall_iou,
        "Dice Score": overall_dice
    }

    # Save all metrics to a single JSON file
    metrics_path = os.path.join(output_folder, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Saved all test metrics in {metrics_path}")
