import numpy as np
from cvxopt import matrix, solvers
from keras.datasets import mnist
from sklearn.decomposition import PCA
import time

# Helper function to load models from a file
def load_models(filename):
    models = {}
    with open(filename, "r") as f:
        for line in f:
            C_val, model_str = line.split(':')
            model = np.array([float(x) for x in model_str.split(',')])
            if C_val not in models:
                models[C_val] = []
            models[C_val].append(model)
    return models

# Function to setup the QP for soft margin SVM
def setup_soft_margin_svm_qp(X, y, C):
    N, d = X.shape
    Q = np.zeros((d + 1 + N, d + 1 + N))  # b, w, and epsilon
    Q[1:d+1, 1:d+1] = np.eye(d)
    p = np.hstack([np.zeros(d + 1), C * np.ones(N)])
    A = np.zeros((2 * N, d + 1 + N))
    c = np.zeros(2 * N)
    for i in range(N):
        A[i, 0] = y[i]
        A[i, 1:d+1] = y[i] * X[i]
        A[i, d+1+i] = 1
        c[i] = 1
        A[N+i, d+1+i] = 1

    return matrix(Q), matrix(p), matrix(A), matrix(c)

# Function to train a soft margin SVM
def train_soft_margin_svm(X, y, C, digit):
    start = time.time()
    y_binary = np.where(y == digit, 1, -1)
    Q, p, A, c = setup_soft_margin_svm_qp(X, y_binary, C)
    sol = solvers.qp(Q, p, -A, -c)
    end = time.time()
    return sol['x'], end - start

# Function to predict the labels of the data
def predict(X, model):
    # Extracting b, w from model;
    b = model[0]
    w = model[1:X.shape[1]+1]
    predictions = np.dot(X, w) + b
    return np.sign(predictions).flatten()

# Function to evaluate the accuracy of the model
def evaluate_accuracy(X, y, models, digits):
    # Store predictions for each SVM
    predictions = np.zeros((len(models), len(y)))
    for i, model in enumerate(models):
        predictions[i] = predict(X, model)

    # Voting mechanism to choose the most common prediction
    final_predictions = []
    for j in range(len(y)):
        # Get the digit corresponding to the model with the highest score
        digit_votes = np.argmax(predictions[:, j])
        final_predictions.append(digits[digit_votes])

    # Calculate accuracy
    final_predictions = np.array(final_predictions)
    correct = (final_predictions == y)
    accuracy = np.mean(correct)
    return accuracy

#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()
apply_pca = False

digits = np.array([2, 3, 8, 9])

filter_array_train = np.isin(train_y, digits)
filter_array_test = np.isin(test_y, digits)

# Filtering the data
train_X_filtered = train_X[filter_array_train]
train_y_filtered = train_y[filter_array_train]
test_X_filtered = test_X[filter_array_test]
test_y_filtered = test_y[filter_array_test]

# Flattening the data
train_X_flattened = train_X_filtered.reshape(-1, 28*28)
test_X_flattened = test_X_filtered.reshape(-1, 28*28)

# Normalizing the data
train_X_flattened = train_X_flattened / 255
test_X_flattened = test_X_flattened / 255

if apply_pca:
    pca = PCA(n_components=100)
    train_X_flattened = pca.fit_transform(train_X_flattened)
    test_X_flattened = pca.transform(test_X_flattened)

# Train soft margin SVM
C_vals = [0.1]
models = {}
train_times = {}
for C_val in C_vals:
    temp = []
    time_tmp = []
    for digit in digits:
        model, train_time = train_soft_margin_svm(train_X_flattened, train_y_filtered, C_val, digit)
        temp.append(model)
        time_tmp.append(train_time)
    models[C_val] = temp
    train_times[C_val] = time_tmp

# Save models to file
if apply_pca:
    file = open("models_part_a_pca.txt", "w")
else:
    file = open("models_part_a.txt", "w")
for C_val, model_list in models.items():
    for model in model_list:
        # Convert model elements to string and join with commas
        model_str = ",".join(map(str, model))
        file.write(f"{C_val}:{model_str}\n")
file.close()

# Evaluate on training data
train_accuracies = []
for C_val in C_vals:
    train_accuracy = evaluate_accuracy(train_X_flattened, train_y_filtered, models[C_val], digits)
    train_accuracies.append(train_accuracy)

# Evaluate on test data
test_accuracies = []
for C_val in C_vals:
    test_accuracy = evaluate_accuracy(test_X_flattened, test_y_filtered, models[C_val], digits)
    test_accuracies.append(test_accuracy)

# Write results to file
if apply_pca:
    file = open("results_part_a_pca.txt", "w")
else:
    file = open("results_part_a.txt", "w")
file.write("Train Accuracies\n")
for i in range(len(C_vals)):
    file.write(f"C = {C_vals[i]}: {train_accuracies[i]}\n")

file.write("Test Accuracies\n")
for i in range(len(C_vals)):
    file.write(f"C = {C_vals[i]}: {test_accuracies[i]}\n")

file.write("Train Times\n")
for i in range(len(C_vals)):
    file.write(f"C = {C_vals[i]}\n")
    for j in range(len(digits)):
        file.write(f"Digit {digits[j]}: {train_times[C_vals[i]][j]}\n")
file.close()
