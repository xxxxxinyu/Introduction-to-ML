# HW3: Decision tree & Adaboost
## Goals
### Decision Tree
- Implement gini index and entropy for measuring the best split of the data. 
- Implement the decision tree classifier with the following two arguments: 
    - criterion: The function to measure the quality of a split of the data. Your model should support "gini" and "entropy".
    - max_depth: The maximum depth of the tree.

### Adaboost
- Implement the Adaboost algorithm by using the decision tree classifier (max_depth=1) you just implemented as the weak classifier.
- The Adaboost model should include the following two arguments:
    - criterion: The function to measure the quality of a split of the data. Your model should support "gini" and "entropy".
    - n_estimators: The total number of weak classifiers.