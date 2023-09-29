# You are not allowed to import any additional packages/libraries.
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.closed_form_w = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None
        self.gradient_descent_weight_matrix = None
        self.gradient_descent_loss = []
        
    # This function computes the closed-form solution of linear regression.
    def closed_form_fit(self, X, y):
        # Compute closed-form solution.
        # Save the weights and intercept to self.closed_form_weights and self.closed_form_intercept
        X_train = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1), X], axis = 1)
        y_train = y.reshape(-1, 1)

        self.closed_form_w = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T.dot(y_train))

        self.closed_form_intercept = self.closed_form_w[0, 0]
        self.closed_form_weights = self.closed_form_w[1:, 0]

    # This function computes the gradient descent solution of linear regression.
    def gradient_descent_fit(self, X, y, lr, epochs):
        # Compute the solution by gradient descent.
        # Save the weights and intercept to self.gradient_descent_weights and self.gradient_descent_intercept
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)
        y = y.reshape(-1, 1)
        self.gradient_descent_weight_matrix = np.random.randn(X.shape[1], 1)
        #self.gradient_descent_weight_matrix = np.zeros(X.shape[1]).reshape(-1,1)
        np.random.seed(1)

        for i in range(epochs):
            idx = np.random.randint(0, X.shape[0], 500)

            X_batch = X[idx]
            y_batch = y[idx]
            y_pred = X_batch @ self.gradient_descent_weight_matrix
            error = y_pred - y_batch
            self.gradient_descent_weight_matrix -= lr * 2 * (X_batch.T @ error) / X.shape[0] 
            loss = self.get_mse_loss(y_pred, y_batch)
            self.gradient_descent_loss.append(loss)
        
        self.gradient_descent_intercept = self.gradient_descent_weight_matrix[0, 0]
        self.gradient_descent_weights = self.gradient_descent_weight_matrix[1:, 0]

    # This function compute the MSE loss value between your prediction and ground truth.
    def get_mse_loss(self, prediction, ground_truth):
        # Return the value.
        loss = np.sum((prediction - ground_truth) ** 2) / prediction.size

        return loss

    # This function takes the input data X and predicts the y values according to your closed-form solution.
    def closed_form_predict(self, X):
        # Return the prediction.
        X_pred = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1), X], axis = 1)
        y_pred = X_pred.dot(self.closed_form_w)
        y_pred = y_pred.reshape(-1)


        return y_pred

    # This function takes the input data X and predicts the y values according to your gradient descent solution.
    def gradient_descent_predict(self, X):
        # Return the prediction.
        X = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1), X],axis=1)
        y_pred = X.dot(self.gradient_descent_weight_matrix)
        y_pred = y_pred.reshape(-1)


        return y_pred
    
    # This function takes the input data X and predicts the y values according to your closed-form solution, 
    # and return the MSE loss between the prediction and the input y values.
    def closed_form_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.closed_form_predict(X), y)

    # This function takes the input data X and predicts the y values according to your gradient descent solution, 
    # and return the MSE loss between the prediction and the input y values.
    def gradient_descent_evaluate(self, X, y):
        # This function is finished for you.
        return self.get_mse_loss(self.gradient_descent_predict(X), y)
        
    # This function use matplotlib to plot and show the learning curve (x-axis: epoch, y-axis: training loss) of your gradient descent solution.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_learning_curve(self):
        plt.plot(range(len(self.gradient_descent_loss)), self.gradient_descent_loss, ',-' ,color='red')
        plt.title('Training Loss')
        plt.xlabel('epochs', color='black')
        plt.ylabel('loss', color='black')
        plt.xticks(color='black')
        plt.yticks(color='black')
        plt.savefig("loss.png")
        plt.show()

# Do not modify the main function architecture.
# You can only modify the arguments of your gradient descent fitting function.
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    
    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    LR.gradient_descent_fit(train_x, train_y, lr=3e-3, epochs=4500000)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")

    LR.plot_learning_curve()