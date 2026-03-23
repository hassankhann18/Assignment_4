# NAME: HASSAN ADNAN
# UCID: 30217418

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Same 80/20 split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build neural network
nn_model = Sequential()
nn_model.add(InputLayer(shape=(X_train_scaled.shape[1],)))
nn_model.add(Dense(16, activation='relu'))     # hidden layer
nn_model.add(Dense(1, activation='sigmoid'))   # output layer for binary classification

# Compile model
nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    verbose=0
)

# Evaluate accuracy
train_loss, train_accuracy = nn_model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_accuracy = nn_model.evaluate(X_test_scaled, y_test, verbose=0)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)