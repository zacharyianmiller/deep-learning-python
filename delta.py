import numpy as np
import numpy.typing
import random
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot_date

import activation.sigmoid as sigmoid

# Naive tensor
Tensor = np.typing.NDArray[np.float64]

# ====================================== #

def delta_sgd(W: Tensor, X: Tensor, D: Tensor):
    alpha = 0.9  # learning rate

    for k in range(X.shape[0]):
        x = np.vstack(X[k, :])
        d = D[0, k]

        v = W @ x # weighted sum
        y = sigmoid.calc(v)

        error = d - y # correct - output
        delta = sigmoid.calc_dydx(v) * error

        dW = alpha * delta * x # delta rule

        for w in range(X.shape[1]):
            W[w] += dW[w].item()

    return W

def delta_batch(W: Tensor, X: Tensor, D: Tensor):
    alpha = 0.9 # learning rate

    dW_sum = np.zeros((X.shape[1], 1))
    N = X.shape[0]
    for k in range(N):
        x = np.vstack(X[k, :])
        d = D[0, k]

        v = W @ x # weighted sum
        y = sigmoid.calc(v)

        error = d - y  # correct - output
        delta = sigmoid.calc_dydx(v) * error

        dW = alpha * delta * x # delta rule
        dW_sum += dW

    dW_avg = dW_sum / N
    W += dW_avg[:, 0]

    return W


def delta_main(processor_type: str, num_epochs: int):
    # Training inputs
    X: Tensor = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    # Correct outputs(supervised)
    D: Tensor = np.array([
        [0, 0, 1, 1]
    ])

    # Initial weights (random between - 1 and 1)
    W: Tensor = np.zeros(X.shape[1])
    for n in range(W.shape[0]):
        W[n] = 2 * (random.random() - 0.5)

    # Train model
    train_start = time.perf_counter()
    for epoch in range(num_epochs):
        if processor_type == "sgd":
            W = delta_sgd(W, X, D)
        elif processor_type == "batch":
            W = delta_batch(W, X, D)
    train_end = time.perf_counter()
    print(f"Training performed in {train_end - train_start:0.4f} seconds")

    # Infer data
    y = np.zeros(X.shape[0])
    for k in range(X.shape[0]):
        x = np.vstack(X[k, :]) # each row of X, transformed
        v = W @ x
        y[k] = sigmoid.calc(v.item())

    # Output inferred / predicted values
    print("Predicted output values:\n")
    for val in y: print(f"{val:.4f}")
    print()

    return 0

# Main function for comparison
def abtest(num_epochs: int = 1000):
    # Training inputs
    X: Tensor = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    # Correct outputs(supervised)
    D: Tensor = np.array([
        [0, 0, 1, 1]
    ])

    error = [0.0, 0.0]
    E1 = np.zeros(num_epochs)
    E2 = E1.copy()

    # Initial weights (random between - 1 and 1)
    W1: Tensor = np.zeros(X.shape[1])
    W2: Tensor = np.zeros(X.shape[1])
    for n in range(W1.shape[0]):
        W1[n] = W2[n] = 2 * (random.random() - 0.5)

    # Infer data
    y2 = y1 = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        W1 = delta_sgd(W1, X, D)
        W2 = delta_batch(W2, X, D)

        es2 = es1 = 0.0

        N = X.shape[0]
        for k in range(N):
            x = np.vstack(X[k, :])
            d = D[0, k]

            v1 = W1 @ x # weighted sum
            y1 = sigmoid.calc(v1)
            es1 += (d - y1) ** 2

            v2 = W2 @ x
            y2 = sigmoid.calc(v2)
            es2 += (d - y2) ** 2

        E1[epoch] = es1.item() / N
        E2[epoch] = es2.item() / N

    # Plot performance difference
    plt.plot(E1)
    plt.plot(E2)
    plt.xlabel('Epochs')
    plt.ylabel('Average of training error')
    plt.title('Learning Rate: SGD vs Batch Methods')
    plt.legend(['SGD', 'Batch'])
    plt.show()

# ====================================== #

if __name__ == "__main__":
    print("Available options:")
    print("1. Delta rule w/ stochastic gradient descent (\"sgd\")")
    print("2. Delta rule w/ batch (\"batch\")")
    print("3. Compare both (\"both\")")
    choice = input("Enter choice: ")

    if choice == "sgd" or choice == "batch":
        delta_main(choice, 10000 if choice == "sgd" else 40000)
    elif choice == "both":
        abtest()
    else:
        print("ERROR: Invalid type entered. Aborting.")