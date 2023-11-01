# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None
        self.w = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        np.random.seed(0)

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        y = y.reshape(-1, 1)

        self.w = np.random.randn(X.shape[1], 1)

        for i in range(self.iteration):
            # 求出預測值
            y_pred = np.dot(X, self.w)
            y_pred = self.sigmoid(y_pred)
            # 求error
            epsilon = 1e-15
            error = (y * np.log(y_pred + epsilon)) + ((1 - y) * np.log(1 - y_pred + epsilon))
            cost = -1 * np.sum(error) / y.size
            # 求gradient
            gradient = -1 / y.size * np.dot(X.T, (y-y_pred))
            # update weights and intercept
            self.w -= self.learning_rate * gradient
            self.intercept = self.w[0, 0]
            self.weights = self.w[1:, 0]
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        y_pred = self.sigmoid(np.dot(X, self.w))

        return y_pred > 0.5

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        c0 = X[y == 0]
        c1 = X[y == 1]
        self.m0 = np.mean(c0, axis=0)
        self.m1 = np.mean(c1, axis=0)

        self.sw = np.zeros((X.shape[1], X.shape[1]))
        for x in c0:
            self.sw += np.dot((x - self.m0).reshape(2, 1), (x - self.m0).reshape(1, 2))
        for x in c1:
            self.sw += np.dot((x - self.m1).reshape(2, 1), (x - self.m1).reshape(1, 2))

        self.sb = np.dot((self.m1 - self.m0).reshape(2, 1), (self.m1 - self.m0).reshape(1, 2))

        self.w = np.dot(np.linalg.inv(self.sw), (self.m1 - self.m0).reshape(2, 1))

        self.slope = self.w[1][0] / self.w[0][0]

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        projected_points = np.dot(X, self.w)
        projected_mean_0 = np.dot(self.m0, self.w)
        projected_mean_1 = np.dot(self.m1, self.w)
        y_pred = []

        for point in projected_points:
            distance_0 = np.abs(point - projected_mean_0)
            distance_1 = np.abs(point - projected_mean_1)
            if distance_0 < distance_1:
                y_pred.append(0)
            else:
                y_pred.append(1)

        return np.array(y_pred)

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        y_pred = self.predict(X)
        
        m = self.slope
        b = 100

        c0 = X[y_pred == 0]
        c1 = X[y_pred == 1]

        c0_x = c0[:, 0]
        c0_y = c0[:, 1]
        c1_x = c1[:, 0]
        c1_y = c1[:, 1]

        p0_x = (m * c0_y + c0_x - m * b) / (m**2 + 1)
        p0_y = (m**2 * c0_y + m * c0_x + b) / (m**2 +1)
        p1_x = (m * c1_y + c1_x - m * b) / (m**2 + 1)
        p1_y = (m**2 * c1_y + m * c1_x + b) / (m**2 + 1)

        

        X = np.linspace(-25, 10, 100)
        Y = m * X + b

        plt.title(f'Projection Line: w={m}, b={b}')
        plt.plot(X, Y, c='blue', linewidth=0.8)

        plt.plot([c1_x, p1_x], [c1_y, p1_y], c='red', linewidth=0.02)
        plt.plot([c0_x, p0_x], [c0_y, p0_y], c='green', linewidth=0.02)

        plt.plot(c1_x, c1_y, '.', c='red', markersize=1)
        plt.plot(c0_x, c0_y, '.', c='green', markersize=1)
        plt.plot(p1_x, p1_y, '.', c='red', markersize=1)
        plt.plot(p0_x, p0_y, '.', c='green', markersize=1)

        plt.savefig("projection.png")
        plt.show()
     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=3e-4, iteration=300000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"
    # FLD.plot_projection(X_test)
