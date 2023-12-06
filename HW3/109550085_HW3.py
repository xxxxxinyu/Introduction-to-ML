# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    return 1 - np.sum(prob ** 2)

# This function computes the entropy of a label array.
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    return -np.sum(prob * np.log2(prob))

class Node:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        
# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth 
        self.tree = None
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return self.gini(y)
        elif self.criterion == 'entropy':
            return self.entropy(y)
        
    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return 1 - np.sum(prob ** 2)

    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return -np.sum(prob * np.log2(prob))
    
    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y, sample_weights=None):
        self.tree = self._fit(X, y, sample_weights, depth=0)


    def _fit(self, X, y, sample_weights, depth):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        if len(unique_classes) == 1:
            return Node(value=unique_classes[0])
        
        if self.max_depth is not None and depth == self.max_depth:
            return Node(value=self._most_common_class(y, sample_weights))
        
        if sample_weights is None:
            sample_weights = np.ones(num_samples) / num_samples

        current_impurity = self.impurity(y)

        best_impurity = float('inf')
        best_criteria = None
        best_sets = None

        for feature_index in range(num_features):
            values = np.unique(X[:, feature_index])
            for threshold in values:
                sets = self._split(X, y, feature_index, threshold)
                weighted_impurity = self._cal_weighted_impurity(sets, sample_weights, y)

                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_criteria = (feature_index, threshold)
                    best_sets = sets

        if current_impurity - best_impurity < 1e-10:
            return Node(value=self._most_common_class(y, sample_weights))
        
        left = self._fit(X[best_sets[0]], y[best_sets[0]], sample_weights[best_sets[0]], depth+1)
        right = self._fit(X[best_sets[1]], y[best_sets[1]], sample_weights[best_sets[1]], depth+1)

        return Node(feature_index=best_criteria[0], threshold=best_criteria[1], left=left, right=right)
    
    def _split(self, X, y, feature_index, threshold):
        mask = X[:, feature_index] <= threshold
        return [np.where(mask)[0], np.where(~mask)[0]]
    
    def _cal_weighted_impurity(self, sets, sample_weights, y):
        total_samples = np.sum(sample_weights)
        weighted_impurity = 0.0
        for subset in sets:
            subset_weight = np.sum(sample_weights[subset]) / total_samples
            weighted_impurity += subset_weight * self.impurity(y[subset])
        return weighted_impurity
    
    def _most_common_class(self, y, sample_weights):
        classes, counts = np.unique(y, return_counts=True)
        index = np.argmax(counts)
        return classes[index]

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x, node=None):
        if node is None:
            node = self.tree

        if node.value is not None:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)
        
    def _cal_feature_importance(self, node, columns):
        if node is None:
            return np.zeros(len(columns))
        
        if node.left is None and node.right is None:
            return np.zeros(len(columns))
        
        importance = np.zeros(len(columns))
        importance[node.feature_index] = 1.0

        importance += self._cal_feature_importance(node.left, columns)
        importance += self._cal_feature_importance(node.right, columns)

        return importance
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        importances = self._cal_feature_importance(self.tree, columns)
        indices = np.argsort(importances)

        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [columns[i] for i in indices])
        plt.title("Feature Importance")
        plt.savefig("feature_importance.png")
        plt.show()

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.alphas = []
        self.classifiers = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        num_samples, num_features = X.shape
        weights = np.ones(num_samples) / num_samples

        for _ in range(self.n_estimators):
            weak_classifier = DecisionTree(criterion=self.criterion, max_depth=2)
            weak_classifier.fit(X, y, sample_weights=weights)
            predictions = weak_classifier.predict(X)

            error = np.sum(weights * (predictions != y))

            alpha = 0.5 * np.log((1 - error + 1e-10) / (error + 1e-10))
            #alpha = 0.0 if np.isnan(alpha) else alpha

            self.alphas.append(alpha)
            self.classifiers.append(weak_classifier)

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)


    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        predictions = np.zeros(X.shape[0])

        for alpha, classifier in zip(self.alphas, self.classifiers):
            predictions += alpha * classifier.predict(X)
        
        final_predictions = (predictions > 0).astype(int)

        return final_predictions

# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    columns = X_train.columns

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))

    tree = DecisionTree(criterion='gini', max_depth=15)
    tree.fit(X_train, y_train)
    tree.plot_feature_importance_img(columns)

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='entropy', n_estimators=10)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    

    
