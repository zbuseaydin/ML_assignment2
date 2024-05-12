import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
from collections import defaultdict
# Load MNIST data
(train_X, train_y), (test_X, test_y) = mnist.load_data()
apply_pca = True

# Filter for digits 2, 3, 8, and 9
digits = [2, 3, 8, 9]
train_mask = np.isin(train_y, digits)
test_mask = np.isin(test_y, digits)

train_X, train_y = train_X[train_mask], train_y[train_mask]
test_X, test_y = test_X[test_mask], test_y[test_mask]

# Flatten the images
train_X = train_X.reshape(-1, 784)  # 28x28 pixels = 784 features
test_X = test_X.reshape(-1, 784)

if apply_pca:
    pca = PCA(n_components=100)
    train_X = pca.fit_transform(train_X)
    test_X = pca.transform(test_X)

# Normalize the data
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf']  # Radial Basis Function (RBF) kernel
}

# Create the SVC object
svc = SVC()

# Set up the grid search with cross-validation
grid_search = GridSearchCV(svc, param_grid, cv=3, scoring='accuracy', verbose=2)

start = time.time()
# Train the model
grid_search.fit(train_X, train_y)
end = time.time()

# Retrieve the best model
best_svc = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Predictions on the training set
train_preds = best_svc.predict(train_X)
train_accuracy = accuracy_score(train_y, train_preds)

# Predictions on the test set
test_preds = best_svc.predict(test_X)
test_accuracy = accuracy_score(test_y, test_preds)

# Write the results to a file
if apply_pca:
    file = open("results_part_d_pca.txt", "w")
else:
    file = open("results_part_d.txt", "w")
file.write(f"Training Accuracy: {train_accuracy}\n")
file.write(f"Test Accuracy: {test_accuracy}\n")
file.write(f"Training Time: {end - start} seconds")
file.close()

# Function to display images in a grid format
def plot_images(images, labels, nrows, ncols, title):
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
    axes = axes.flatten()
    for img, ax, label in zip(images, axes, labels):
        ax.imshow(img.reshape(28, 28), cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.savefig(f"{title}.png")

def sample_balanced_vectors(vectors_by_label, num_per_class=2):  # Fewer vectors per class
    balanced_vectors = []
    balanced_labels = []
    for label, vectors in vectors_by_label.items():
        if len(vectors) >= num_per_class:
            chosen_vectors = vectors[:num_per_class]
        else:
            chosen_vectors = vectors
        balanced_vectors.extend(chosen_vectors)
        balanced_labels.extend([label] * len(chosen_vectors))
    return balanced_vectors, balanced_labels

# Get support vectors
support_indices = best_svc.support_
support_vectors = best_svc.support_vectors_
support_labels = train_y[support_indices]

# Organize and sample support vectors
support_vectors_by_label = defaultdict(list)
for vector, label in zip(best_svc.support_vectors_, support_labels):
    support_vectors_by_label[label].append(vector)
balanced_support_vectors, balanced_support_labels = sample_balanced_vectors(support_vectors_by_label)

# Get non-support vectors
non_support_indices = [i for i in range(len(train_X)) if i not in support_indices]
non_support_vectors = train_X[non_support_indices]
non_support_labels = train_y[non_support_indices]

# Organize and sample non-support vectors
non_support_vectors_by_label = defaultdict(list)
for vector, label in zip(non_support_vectors, non_support_labels):
    non_support_vectors_by_label[label].append(vector)
balanced_non_support_vectors, balanced_non_support_labels = sample_balanced_vectors(non_support_vectors_by_label)

# Plot balanced support vectors
plot_images(balanced_support_vectors, balanced_support_labels, nrows=4, ncols=2, title='Balanced Support Vector Images')

# Plot balanced non-support vectors
plot_images(balanced_non_support_vectors, balanced_non_support_labels, nrows=4, ncols=2, title='Balanced Non-Support Vector Images')