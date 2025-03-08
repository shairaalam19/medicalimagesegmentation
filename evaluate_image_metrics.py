import json
import os
import matplotlib.pyplot as plt

def find_best_and_worst_image(metrics_file, metric):
    # Load the JSON data from the file
    with open(metrics_file, 'r') as f:
        data = json.load(f)

    # Extract the overall metric before deleting it
    overall_value = data["overall_metrics"].get(metric, None)

    # Ignore the overall metrics
    del data["overall_metrics"]

    # Initialize variables to store the best and worst image
    best_image = None
    worst_image = None
    best_value = float('-inf')  # Best value starts low
    worst_value = float('inf')  # Worst value starts high
    
    # Store all metric values for box plot
    all_metric_values = []

    # Iterate over the images and their metrics
    for image, metrics in data.items():
        # Check if the metric exists in the image's data
        if metric in metrics:
            value = metrics[metric]
            all_metric_values.append(value)  # Store for box plot
            
            # Check if the current image has the best or worst value
            if value > best_value:
                best_value = value
                best_image = image
            
            if value < worst_value:
                worst_value = value
                worst_image = image

    return best_image, best_value, worst_image, worst_value, overall_value, all_metric_values

def plot_box_plot(best_value, worst_value, overall_value, all_values, metric, output_folder):
    plt.figure(figsize=(6, 4))

    # Create a box plot
    plt.boxplot(all_values, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))

    # Add best, worst, and overall values as points
    plt.scatter(1, best_value, color='green', label='Best', zorder=3)
    plt.scatter(1, worst_value, color='red', label='Worst', zorder=3)
    plt.scatter(1, overall_value, color='orange', label='Overall', zorder=3)

    # Formatting
    plt.ylabel(metric)
    plt.title(f"Box Plot of {metric} Scores")
    plt.legend()

    # Save the plot
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, f"{metric}_boxplot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Box plot saved at {plot_path}")

# Example usage
metrics_file = 'outputs/results/models/training/20250305_231254/epoch_32/test_metrics.json'
metric_to_check = 'Dice Score'  # Change to any metric like 'Recall', 'F1-score', etc.
output_folder = 'outputs/results/models/training/20250305_231254/epoch_32'  # Folder to save the box plot

best_img, best_val, worst_img, worst_val, overall_val, all_vals = find_best_and_worst_image(metrics_file, metric_to_check)
print(f"Best Image: {best_img}, {metric_to_check}: {best_val}")
print(f"Worst Image: {worst_img}, {metric_to_check}: {worst_val}")
print(f"Overall {metric_to_check}: {overall_val}")

# Generate and save box plot
plot_box_plot(best_val, worst_val, overall_val, all_vals, metric_to_check, output_folder)
