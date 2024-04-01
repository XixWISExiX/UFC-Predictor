import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def PreProcessData():
    # Get data from csv
    X = pd.read_csv('./ufc_data/ML_fighter_stats.csv')

    # Replace '--' with NaN
    X.replace('--', pd.NA, inplace=True)

    # Drop rows containing NaN
    X.dropna(inplace=True)

    # Get the dependent variable
    X.dropna(axis=0, subset=['WINNER'], inplace=True)
    y = X.WINNER
    X.drop(['WINNER'], axis=1, inplace=True)

    # Columns to Encode
    encode_cols = ['STANCE 1', 'STANCE 2', 'FIGHTER 1', 'FIGHTER 2']

    # Columns encoded
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[encode_cols]))

    # Remove categorical columns (will replace with one-hot encoding)
    num_X = X.drop(encode_cols, axis=1)
    cols = num_X.columns

    # Create StandardScaler object
    scaler = StandardScaler()

    # Fit the scaler to your data and transform it
    X_normalized = pd.DataFrame(scaler.fit_transform(num_X), columns=cols)

    # Add one-hot encoded columns to numerical features
    total_X = pd.concat([X_normalized, OH_cols], axis=1)

    # Ensure all columns have string type
    total_X.columns = total_X.columns.astype(str)

    # Break off  set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(total_X, y, train_size=0.8, test_size=0.2, random_state=0, shuffle=False)

    return X_train, X_valid, y_train, y_valid


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted -y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X, threshold=0.5):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted_per = self.sigmoid(linear_model)
        y_predicted_bool = [1 if i > threshold else 0 for i in y_predicted_per]
        return y_predicted_per, y_predicted_bool

    def validate(self, y_pred, y_valid):
        num_of_correct_predictions = 0
        for i in range(len(y_valid)):
            if y_pred[i] == y_valid[i]:
                num_of_correct_predictions += 1
        accuracy = num_of_correct_predictions / len(y_valid)
        return accuracy

    def precision(self, y_pred, y_valid):
        num_of_correct_positives = 0
        num_of_positives = 0
        for i in range(len(y_valid)):
            if y_pred[i] == y_valid[i] and y_valid[i]:
                num_of_correct_positives += 1
            if y_pred[i]:
                num_of_positives += 1
        accuracy = num_of_correct_positives / num_of_positives
        return accuracy

    def recall(self, y_pred, y_valid):
        TP_and_FN = 0
        TP = 0
        for i in range(len(y_valid)):
            if y_pred[i] == y_valid[i] and y_valid[i]:
                TP += 1
            if (y_valid[i] and y_pred[i] == y_valid[i]) or (not y_pred[i] and y_pred[i] != y_valid[i]):
                TP_and_FN += 1
        accuracy = TP / TP_and_FN 
        return accuracy

    def specificity(self, y_pred, y_valid):
        TN_and_FP = 0
        TN = 0
        for i in range(len(y_valid)):
            if y_pred[i] == y_valid[i] and not y_valid[i]:
                TN += 1
            if (not y_valid[i] and y_pred[i] == y_valid[i]) or (y_pred[i] and y_pred[i] != y_valid[i]):
                TN_and_FP += 1
        accuracy = TN / TN_and_FP 
        return accuracy

    def plot(self, y_pred, y_valid):
        y_pred = [0 if y < 0.5 else 1 for y in y_pred]
        # Calculate confusion matrix
        cm = confusion_matrix(y_valid, y_pred)

        # Define labels for the confusion matrix
        labels = [['True Negative', 'False Positive'], ['False Negative', 'True Positive']]

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

        # Annotate cells with labels
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.3, labels[i][j], ha='center', va='center', color='red')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix with Labels')
        plt.show()



class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.biases_input_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.biases_hidden_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x >= 0, 1, 0)

    def feedforward(self, X):
        # Input to hidden layer
        self.hidden_sum = np.dot(X, self.weights_input_hidden) + self.biases_input_hidden
        # self.hidden_activation = self.relu(self.hidden_sum)
        self.hidden_activation = self.sigmoid(self.hidden_sum)

        # Hidden to output layer
        self.output_sum = np.dot(self.hidden_activation, self.weights_hidden_output) + self.biases_hidden_output
        self.output_activation = self.sigmoid(self.output_sum)

        return self.output_activation

    def backward(self, X, y, output):
        # Output layer
        error_output = y - output
        # delta_output = error_output * self.relu_derivative(output)
        delta_output = error_output * self.sigmoid_derivative(output)

        # Hidden layer
        error_hidden = error_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_activation)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_activation.T.dot(delta_output) * self.learning_rate
        self.biases_hidden_output += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(delta_hidden) * self.learning_rate
        self.biases_input_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        y = y.reshape(-1,1)
        for epoch in range(epochs):
            # Forward pass
            output = self.feedforward(X)

            # Backpropagation
            self.backward(X, y, output)

            # Print loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                loss = -(y * np.log(output) + (1 - y) * np.log(1 - output))
                loss = np.mean(loss)
                print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
                # print(f'Epoch {epoch + 1}, Loss: {loss}')

    def predict(self, X):
        return self.feedforward(X)

    def validate(self, y_pred, y_valid):
        num_of_correct_predictions = np.sum((y_pred >= 0.5) == y_valid)
        accuracy = num_of_correct_predictions / len(y_valid)
        return accuracy

    def plot(self, y_pred, y_valid):
        y_pred = [0 if y < 0.5 else 1 for y in y_pred]
        # Calculate confusion matrix
        cm = confusion_matrix(y_valid, y_pred)

        # Define labels for the confusion matrix
        labels = [['True Negative', 'False Positive'], ['False Negative', 'True Positive']]

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

        # Annotate cells with labels
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.3, labels[i][j], ha='center', va='center', color='red')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix with Labels')
        plt.show()



class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_class = len(np.unique(y))

        # Check termination conditions
        if depth == self.max_depth or num_class == 1:
            return np.bincount(y).argmax()

        # Find the best split
        best_split = self._find_best_split(X, y)

        if best_split['impurity'] == 0:
            return np.bincount(y).argmax()

        left_indices = X[:, best_split['feature_index']] <= best_split['threshold']
        right_indices = ~left_indices

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature_index': best_split['feature_index'],
                'threshold': best_split['threshold'],
                'left': left_subtree,
                'right': right_subtree}

    def _calculate_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_split = {'impurity': 0}
        best_gini = float('inf')

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = ~left_indices

                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue

                left_gini = self._calculate_gini(y[left_indices])
                right_gini = self._calculate_gini(y[right_indices])

                gini = (len(y[left_indices]) / num_samples) * left_gini \
                       + (len(y[right_indices]) / num_samples) * right_gini

                if gini < best_gini:
                    best_split = {'feature_index': feature_index,
                                  'threshold': threshold,
                                  'impurity': gini}
                    best_gini = gini

        return best_split

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if isinstance(node, int):
            return node

        if isinstance(node, dict):
            if x[node['feature_index']] <= node['threshold']:
                return self._traverse_tree(x, node['left'])
            else:
                return self._traverse_tree(x, node['right'])
        else:
            # If the node is not a dictionary, return it as it is
            return node

    def validate(self, y_pred, y_valid):
        num_of_correct_predictions = 0
        for i in range(len(y_valid)):
            if y_pred[i] == y_valid[i]:
                num_of_correct_predictions += 1
        accuracy = num_of_correct_predictions / len(y_valid)
        return accuracy

    def plot(self, y_pred, y_valid):
        y_pred = [0 if y < 0.5 else 1 for y in y_pred]
        # Calculate confusion matrix
        cm = confusion_matrix(y_valid, y_pred)

        # Define labels for the confusion matrix
        labels = [['True Negative', 'False Positive'], ['False Negative', 'True Positive']]

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

        # Annotate cells with labels
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.3, labels[i][j], ha='center', va='center', color='red')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix with Labels')
        plt.show()



class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.estimators = []

    def fit(self, X, y):
        self.estimators = []
        for _ in range(self.n_estimators):
            # Create a decision tree
            tree = DecisionTree(max_depth=self.max_depth)
            
            # Randomly sample the data with replacement
            sample_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sampled = X[sample_indices]
            y_sampled = y[sample_indices]
            
            # Fit the decision tree on the sampled data
            tree.fit(X_sampled, y_sampled)
            
            # Append the decision tree to the list of estimators
            self.estimators.append(tree)

    def predict(self, X):
        # Make predictions using each decision tree and take the majority vote
        predictions = np.array([tree.predict(X) for tree in self.estimators])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    def validate(self, y_pred, y_valid):
        num_of_correct_predictions = 0
        for i in range(len(y_valid)):
            if y_pred[i] == y_valid[i]:
                num_of_correct_predictions += 1
        accuracy = num_of_correct_predictions / len(y_valid)
        return accuracy

    def plot(self, y_pred, y_valid):
        y_pred = [0 if y < 0.5 else 1 for y in y_pred]
        # Calculate confusion matrix
        cm = confusion_matrix(y_valid, y_pred)

        # Define labels for the confusion matrix
        labels = [['True Negative', 'False Positive'], ['False Negative', 'True Positive']]

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

        # Annotate cells with labels
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.3, labels[i][j], ha='center', va='center', color='red')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix with Labels')
        plt.show()
