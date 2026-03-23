# NAME: HASSAN ADNAN
# UCID: 30217418

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Using entropy to train model
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)

# Predictions
y_train_prediction = dt_model.predict(X_train)
y_test_prediction = dt_model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_prediction)
test_accuracy = accuracy_score(y_test, y_test_prediction)

print("The training accuracy is :", train_accuracy)
print("The test accuracy is :", test_accuracy)

# Discussion
# Entropy measures how mixed or uncertain the classes are in a node.

# As the training accuracy is higher than the test accuracy, the model shows some overfitting rather than perfect generalization.