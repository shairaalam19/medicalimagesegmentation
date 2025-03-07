import json

def find_best_and_worst_image(metrics_file, metric):
    # Load the JSON data from the file
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    # Ignore the overall_metrics object
    del data["overall_metrics"]
    
    # Initialize variables to store the best and worst image
    best_image = None
    worst_image = None
    best_value = float('-inf')  # Best value starts low
    worst_value = float('inf')  # Worst value starts high
    
    # Iterate over the images and their metrics
    for image, metrics in data.items():
        # Check if the metric exists in the image's data
        if metric in metrics:
            value = metrics[metric]
            
            # Check if the current image has the best or worst value
            if value > best_value:
                best_value = value
                best_image = image
            
            if value < worst_value:
                worst_value = value
                worst_image = image
    
    return best_image, best_value, worst_image, worst_value

# Example usage
metrics_file = 'outputs/results/models/training/20250305_231254/epoch_25/test_metrics.json'
metric_to_check = 'IoU'  # You can change this to any metric, like 'Recall (Sensitivity)', 'F1 Score', etc.

best_image, best_value, worst_image, worst_value = find_best_and_worst_image(metrics_file, metric_to_check)

print(f"Best Image: {best_image}, {metric_to_check}: {best_value}")
print(f"Worst Image: {worst_image}, {metric_to_check}: {worst_value}")
