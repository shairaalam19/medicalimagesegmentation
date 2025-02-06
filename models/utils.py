import os
import sys
import torch
import datetime
import shutil
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=1e-4, verbose=True) # lowers learning rate when loss stops improving

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
        scheduler.step(epoch_loss)

        if epoch_loss < 0: 
            print(f"Epoch Loss is less than zero ({epoch_loss:.4f}). Ending training. ")
            break

        last_epoch = epoch
        epoch_losses.append(epoch_loss)  # Store the loss for plotting

        if (epoch + 1) % 5 == 0: 
            save_model(model, f"epoch_{epoch + 1}", model_folder)

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
    plt.xlabel("Batch")
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

    model = EdgeSegmentationCNN()
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

    model.eval()
    for i, (inputs, targets, file_names) in enumerate(test_loader):
        with torch.no_grad(): # disables gradient computation for computational efficiency
            outputs = model(inputs) # forward pass to generate predictions 

            # Future visualization and calculations 
            # Calculate the loss
            loss = criterion(outputs, targets)  
            total_loss += loss.item()
            batch_losses.append(loss.item())

        # Save individual results
        for j, file_name in enumerate(file_names):
            original_filename = os.path.basename(file_name)
            output_filename = os.path.join(output_folder, f"{original_filename}")
            save_combined_image(inputs[j], outputs[j], targets[j], output_filename)

    # Calculate and log the average loss
    average_loss = total_loss / len(test_loader)
    print(f"Average Test Loss: {average_loss:.4f}")

    # Save the loss graph
    save_loss_graph(batch_losses, output_folder, title="Test Loss per Batch", file_name="test_loss.png")

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
