import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models

# --- GPU CHECK ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    print("⚠️ No GPU detected. Using CPU.")

# --- LOAD AND PREP DATA ---
# --- LOAD AND PREP DATA ---

# Load full dataset
df = pd.read_csv(r'/final_fc_dataset.csv')

# Filter out OCD rows
df_no_ocd = df[df['disorder'] != 'OCD']

# --- HOLD OUT A FEW HC SAMPLES FOR OCD TESTING ---
hc_holdout = df_no_ocd[df_no_ocd['is_HC'] == 1].sample(n=10, random_state=42)

# OCD rows from original dataset
df_ocd = df[df['disorder'] == 'OCD']

# Combine OCD + held-out HCs
df_ocd_with_hc = pd.concat([df_ocd, hc_holdout], axis=0).reset_index(drop=True)

# Save to CSV
output_path = r'/ocd_plus_heldout_hc.csv'
df_ocd_with_hc.to_csv(output_path, index=False)

print(f"✅ OCD + Held-out HC saved to: {output_path}")

# Now drop held-out HCs from training pool
df_no_ocd = df_no_ocd.drop(index=hc_holdout.index)

# Separate features and target
X = df_no_ocd.drop(columns=['disorder', 'is_HC'])
y = df_no_ocd['is_HC']

# Train/test split
# --- SPLIT INTO TRAIN (80%), VAL (10%), TEST (10%) ---

# First split into train (80%) and temp (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Then split temp into val (10%) and test (10%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42
)


# Feature scaling
scaler = StandardScaler()
X_train_scaled = X_train #scaler.fit_transform(X_train)
X_test_scaled = X_test #scaler.transform(X_test)


# --- MODEL ---
from tensorflow.keras import layers, models, regularizers

model = models.Sequential([
    layers.Input(shape=(9730,)),

    # Block 1: Wide entry
    layers.Dense(2048, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Block 2
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Block 3
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Block 4
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Block 5
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Output layer
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- TRAIN ---
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val, y_val),
    epochs=45,
    batch_size=32
)


# --- EVAL ---
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")



# --- SAVE MODEL ---
model_path = r'/fc_isHC_model.h5'
model.save(model_path)
print(f"Model saved to: {model_path}")

