import kagglehub

# Download latest version
path = kagglehub.dataset_download("maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset")

print("Path to dataset files:", path)