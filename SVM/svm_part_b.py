import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import time

# Load data
(train_X, train_y), (test_X, test_y) = mnist.load_data()
apply_pca = True

# Filter for digits 2, 3, 8, 9
digits = [2, 3, 8, 9]
train_mask = np.isin(train_y, digits)
test_mask = np.isin(test_y, digits)

train_X, train_y = train_X[train_mask], train_y[train_mask]
test_X, test_y = test_X[test_mask], test_y[test_mask]

# Flatten the images
train_X = train_X.reshape(-1, 784)  # 28*28 = 784
test_X = test_X.reshape(-1, 784)

if apply_pca:
    pca = PCA(n_components=100)
    train_X = pca.fit_transform(train_X)
    test_X = pca.transform(test_X)

# Normalize the data
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

# Set up the parameter grid
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

# Create the LinearSVC object
svc = LinearSVC(dual=False)

# Set up the grid search with cross-validation
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', verbose=2)

start = time.time()
# Train the model
grid_search.fit(train_X, train_y)
end = time.time()

# Retrieve the best model
best_svc = grid_search.best_estimator_
print("Best C parameter:", grid_search.best_params_)

# Evaluate on training set
train_preds = best_svc.predict(train_X)
train_accuracy = accuracy_score(train_y, train_preds)

# Evaluate on test set
test_preds = best_svc.predict(test_X)
test_accuracy = accuracy_score(test_y, test_preds)

# create a file to write the results
if apply_pca:
    file = open("results_part_b_pca.txt", "w")
else:
    file = open("results_part_b.txt", "w")
file.write(f"Training Accuracy: {train_accuracy}\n")
file.write(f"Test Accuracy: {test_accuracy}\n")
file.write(f"Training Time: {end - start} seconds")
file.close()
