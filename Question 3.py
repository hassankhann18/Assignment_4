# NAME: HASSAN ADNAN
# UCID: 30217418

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Constrained decision tree
constrained_tree = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    random_state=42
)

constrained_tree.fit(X_train, y_train)

# Predictions
y_train_prediction = constrained_tree.predict(X_train)
y_test_prediction = constrained_tree.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_prediction)
test_accuracy = accuracy_score(y_test, y_test_prediction)

print("Training Accuracy is :", train_accuracy)
print("Test Accuracy is :", test_accuracy)

# Top 5 features
importances = constrained_tree.feature_importances_
top_indices = np.argsort(importances)[::-1][:5]

print("\nTop 5 Most Important Features are:")
for i in top_indices:
    print(feature_names[i], ":", importances[i])


# Controlling model complexity helps reduce overfitting because
# a smaller tree is less likely to memorize the training data.

# Feature importance improves interpretability because it shows
# which features had the biggest influence on the model's decisions.