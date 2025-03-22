import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load features and labels from the previously saved files
features = np.load("features.npy")
labels = np.load("labels.npy")
print(features.shape)

# Optionally: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Optionally: Apply PCA for dimensionality reduction (uncomment if needed)
# pca = PCA(n_components=64)  # Reduce to 50 dimensions or adjust as needed
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)  # Use the same PCA transformation on the test set

# Split training data further into training and validation sets
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Prepare DMatrix for training (XGBoost's internal format)
dtrain = xgb.DMatrix(X_train_sub, label=y_train_sub)
dval = xgb.DMatrix(X_val, label=y_val)

# Set hyperparameters
params = {
    'objective': 'binary:logistic',  # for binary classification
    'eval_metric': 'logloss',  # evaluation metric
    'max_depth': 9,  # Randomly chosen
    'learning_rate': 0.01,  # Randomly chosen
    'subsample': 0.9,  # Randomly chosen
    'colsample_bytree': 0.3,  # Randomly chosen
    'gamma': 0.1  # Randomly chosen
}

# Create a list of evaluation data
evals = [(dtrain, 'train'), (dval, 'eval')]

# Train the model with early stopping
num_round = 200  # Number of boosting rounds (estimators)
early_stopping_rounds = 10  # Stop after 10 rounds with no improvement

bst = xgb.train(
    params,
    dtrain,
    num_round,
    evals=evals,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=True
)

# Get the best number of estimators (trees) used
print(f"Best number of estimators: {bst.best_iteration}")

# Use the trained model for prediction (use best iteration)
y_pred = bst.predict(xgb.DMatrix(X_test)) > 0.5  # Binary classification threshold

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy:.4f}")

# Optionally: Save the trained model for later use
bst.save_model("xgboost_model_random.json")
