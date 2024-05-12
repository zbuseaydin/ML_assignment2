import numpy as np
from cvxopt import matrix, solvers
from keras.datasets import mnist
from sklearn.decomposition import PCA
import time

def load_models(filename):
    models = {}
    with open(filename, "r") as f:
        for line in f:
            # Split the line into key (C_val) and model data
            C_val, model_str = line.split(':')
            # Convert string back to numpy array
            model = np.array([float(x) for x in model_str.split(',')])
            if C_val not in models:
                models[C_val] = []
            models[C_val].append(model)
    return models

def compute_rbf_kernel(X1, X2, gamma):
    X1_norm = np.sum(X1**2, axis=1)
    X2_norm = np.sum(X2**2, axis=1)
    K = X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)
    K = np.exp(-gamma * K)
    return K


def compute_bias(y_binary, alphas, K):
    support_vector_indices = (alphas > 1e-5)
    if not support_vector_indices.any():
        return 0  # Default bias if no support vectors

    # Filter everything according to support vectors
    alphas_sv = alphas[support_vector_indices]
    y_sv = y_binary[support_vector_indices]
    K_sv = K[support_vector_indices][:, support_vector_indices]

    # Compute biases using only the support vectors' data
    biases = y_sv - np.dot(K_sv, alphas_sv * y_sv)
    return np.mean(biases)


def setup_soft_margin_dual(X, y, C):
    N, d = X.shape
    gamma = 1.0 / d
    K = compute_rbf_kernel(X, X, gamma)
    P = matrix(np.outer(y, y) * K)
    q = matrix(-np.ones((N , 1)))
    G = matrix(np.vstack((-np.eye(N), np.eye(N))))
    h = matrix(np.hstack((np.zeros(N), np.ones(N) * C)))
    A = matrix(y.reshape(1, -1), tc='d')
    b = matrix(np.array([0.0]), tc='d')
    return P, q, G, h, A, b, K, gamma

def train_soft_margin_dual(X, y, C, digit):
    start = time.time()
    y_binary = np.where(y == digit, 1, -1)
    P, q, G, h, A, b, K, gamma = setup_soft_margin_dual(X, y_binary, C)
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x']).flatten()
    sv_indices = alphas > 1e-5

    if not sv_indices.any():
        b = 0  # or handle no support vectors scenario as needed
    else:
        alphas_sv = alphas[sv_indices]
        K_sv = K[sv_indices][:, sv_indices]
        b = compute_bias(y_binary[sv_indices], alphas_sv, K_sv)

    end = time.time()
    model = {
        'alphas': alphas[sv_indices],  # Support vector coefficients
        'support_vectors': X[sv_indices],  # Support vectors
        'support_vector_labels': y_binary[sv_indices],  # Labels of support vectors
        'b': b,  # Bias term
        'gamma': gamma,  # Gamma for the RBF kernel
    }
    return model, end - start



def predict(X, model):
    if len(model['alphas']) == 0:
        # Handle case where no support vectors exist
        return np.zeros(len(X))  # Or some other default prediction

    # Compute the RBF kernel between input X and the support vectors
    K = compute_rbf_kernel(X, model['support_vectors'], model['gamma'])

    # Ensure that alphas and labels are in the correct shape
    alphas = model['alphas'][:, np.newaxis]  # Ensure alphas is a column vector
    labels = model['support_vector_labels'][:, np.newaxis]  # Ensure labels is a column vector

    # Prediction calculation using the kernel dot product, support vector labels, alphas and bias
    predictions = np.dot(K, (alphas * labels)) + model['b']
    return np.sign(predictions).flatten()  # Flatten to ensure it is a 1D array

def evaluate_accuracy(X, y, models, digits):
    # Store predictions from each SVM
    predictions = np.zeros((len(models), len(y)))
    for i, model in enumerate(models):
        predictions[i] = predict(X, model)

    # Voting mechanism for final prediction
    final_predictions = []
    for j in range(len(y)):
        digit_votes = np.argmax(predictions[:, j])
        final_predictions.append(digits[digit_votes])

    final_predictions = np.array(final_predictions)
    accuracy = np.mean(final_predictions == y)
    return accuracy

#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()
apply_pca = False


digits = np.array([2, 3, 8, 9])

filter_array_train = np.isin(train_y, digits)
filter_array_test = np.isin(test_y, digits)

train_X_filtered = train_X[filter_array_train]
train_y_filtered = train_y[filter_array_train]
test_X_filtered = test_X[filter_array_test]
test_y_filtered = test_y[filter_array_test]

train_X_flattened = train_X_filtered.reshape(-1, 28*28)
test_X_flattened = test_X_filtered.reshape(-1, 28*28)

train_X_flattened = train_X_flattened/255
test_X_flattened = test_X_flattened/255

if apply_pca:
    pca = PCA(n_components=100)
    train_X_flattened = pca.fit_transform(train_X_flattened)
    test_X_flattened = pca.transform(test_X_flattened)

C_vals = [0.1]
models = {}
train_times = {}
for C_val in C_vals:
    temp = []
    time_tmp = []
    for digit in digits:
        model, train_time = train_soft_margin_dual(train_X_flattened, train_y_filtered, C_val, digit)
        temp.append(model)
        time_tmp.append(train_time)
    models[C_val] = temp
    train_times[C_val] = time_tmp

if apply_pca:
    file = open("models_part_c_pca.txt", "w")
else:
    file = open("models_part_c.txt", "w")
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

test_accuracies = []
for C_val in C_vals:
    test_accuracy = evaluate_accuracy(test_X_flattened, test_y_filtered, models[C_val], digits)
    test_accuracies.append(test_accuracy)

if apply_pca:
    file = open("results_part_c_pca.txt", "w")
else:
    file = open("results_part_c.txt", "w")
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

