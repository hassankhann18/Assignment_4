from sklearn.datasets import load_breast_cancer
import numpy as np


data = load_breast_cancer()

# Constructing feature matrix X and target vector y
X = data.data
y = data.target

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

print("Class names:", data.target_names)

# Count samples in each class
class_counts = np.bincount(y)
print("The number of samples in each class:")
print("Malignant :", class_counts[0])
print("Benign :", class_counts[1])

# Discussion

# The dataset is slightly imbalanced because there are more benign cases than malignant cases.
# Class balance matters because a model may favor the larger class, which can make accuracy misleading.