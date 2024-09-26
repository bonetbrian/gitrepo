import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x):
        self.Z1 = np.dot(x, self.weights1) + self.bias1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.weights2) + self.bias2
        output = self.softmax(self.Z2)
        return output

    def compute_loss(self, y_true, y_pred):
        n_samples = y_true.shape[0]
        logp = -np.log(y_pred[range(n_samples), np.argmax(y_true, axis=1)])
        loss = np.sum(logp) / n_samples
        return loss

    def backward(self, X, y_true, y_pred):
        n_samples = X.shape[0]

        dZ2 = y_pred - y_true
        dW2 = np.dot(self.A1.T, dZ2) / n_samples
        db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples

        dA1 = np.dot(dZ2, self.weights2.T)
        dZ1 = dA1 * self.relu_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / n_samples
        db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples

        # Update weights and biases
        self.weights1 -= self.learning_rate * dW1
        self.bias1 -= self.learning_rate * db1
        self.weights2 -= self.learning_rate * dW2
        self.bias2 -= self.learning_rate * db2
    

    def train(self, X_train, y_train, epochs=100):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X_train)

            # Compute the loss
            loss = self.compute_loss(y_train, y_pred)

            # Backward pass
            self.backward(X_train, y_train, y_pred)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
