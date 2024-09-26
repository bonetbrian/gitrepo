import numpy as np
import matplotlib.pyplot as plt
import model as myModel
from utils import load_data, shuffle_data, get_batches

LEARNING_RATE = 0.02
EPOCHS = 10
BATCH_SIZE = 64
HIDDEN_SIZE = 128  # number of neurons in hidden layer

def compute_accuracy(y_true, y_pred):
    #compute accuracy of predictions.
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y_true, axis=1)
    accuracy = np.mean(y_pred_class == y_true_class)
    return accuracy

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def write_test_scores(test_loss, test_accuracy):
    # write test loss and accuracy to a text file
    file_path = r'E:\pycodes\NeuralNetwork\Bscores.txt'

    with open(file_path, 'a') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n")

def train():
    # load MNIST dataset
    X_train, y_train, X_test, y_test = load_data()

    # split into training and validation sets (80% train, 20% validation)
    split_idx = int(0.8 * X_train.shape[0])
    X_val, y_val = X_train[split_idx:], y_train[split_idx:]
    X_train, y_train = X_train[:split_idx], y_train[:split_idx]

    # initialize neural network
    input_size = X_train.shape[1]
    output_size = 10  # 10 classes for MNIST
    model = myModel.NeuralNetwork(input_size, HIDDEN_SIZE, output_size, LEARNING_RATE)

    # lists to store losses and accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # training loop
    for epoch in range(EPOCHS):
        X_train, y_train = shuffle_data(X_train, y_train)
        total_loss = 0

        for X_batch, y_batch in get_batches(X_train, y_train, BATCH_SIZE):
            y_pred = model.forward(X_batch)
            loss = model.compute_loss(y_batch, y_pred)
            total_loss += loss
            model.backward(X_batch, y_batch, y_pred)

        # calculate training accuracy
        y_train_pred = model.forward(X_train)
        train_accuracy = compute_accuracy(y_train, y_train_pred)

        # Cclculate validation loss and accuracy
        y_val_pred = model.forward(X_val)
        val_loss = model.compute_loss(y_val, y_val_pred)  # calculate validation loss
        val_accuracy = compute_accuracy(y_val, y_val_pred)

        # append losses and accuracies to lists
        train_losses.append(total_loss)
        val_losses.append(val_loss)  # store the calculated validation loss
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # evaluate on test set
    y_test_pred = model.forward(X_test)
    test_loss = model.compute_loss(y_test, y_test_pred)
    test_accuracy = compute_accuracy(y_test, y_test_pred)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    write_test_scores(test_loss, test_accuracy)

    # plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)


if __name__ == "__main__":
    train()
