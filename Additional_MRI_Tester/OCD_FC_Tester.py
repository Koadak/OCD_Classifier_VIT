from tensorflow.keras.models import load_model
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- LOAD MODEL ---
model = load_model(r'/fc_isHC_model.h5')

# --- LOAD DATA ---
df = pd.read_csv(r'/ocd_plus_heldout_hc.csv')

# Split separately by class
df_hc = df[df['is_HC'] == 1]
df_ocd = df[df['is_HC'] == 0]

# Split HC: 8 train, 2 test
df_hc_train, df_hc_test = train_test_split(df_hc, test_size=2, random_state=42)

# Split OCD: 8 train, 4 test
df_ocd_train, df_ocd_test = train_test_split(df_ocd, test_size=4, random_state=42)

# Combine to form balanced train/test sets
df_train = pd.concat([df_hc_train, df_ocd_train]).sample(frac=1, random_state=42).reset_index(drop=True)
df_test = pd.concat([df_hc_test, df_ocd_test]).sample(frac=1, random_state=42).reset_index(drop=True)

# --- Separate features and targets ---
X_ocd_train = df_train.drop(columns=['disorder', 'is_HC'])
y_ocd_train = df_train['is_HC']

X_ocd_test = df_test.drop(columns=['disorder', 'is_HC'])
y_ocd_test = df_test['is_HC']

# --- SCALE with training data (from original non-OCD set) ---
df_no_ocd = df[df['disorder'] != 'OCD']
X_train_reference = df_no_ocd.drop(columns=['disorder', 'is_HC'])
scaler = StandardScaler().fit(X_train_reference)

X_ocd_train_scaled = X_ocd_train# scaler.transform(X_ocd_train)
X_ocd_test_scaled = X_ocd_test# scaler.transform(X_ocd_test)

# --- FINE-TUNE MODEL ON OCD SUBSET ---
for layer in model.layers:
    layer.trainable = True  # Fine-tune all layers

# Recompile with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train on the small OCD training set
history = model.fit(
    X_ocd_train_scaled, y_ocd_train,
    epochs=50,
    batch_size=4,
    verbose=1
)

# --- Evaluate on the Held-Out OCD Samples ---
loss, acc = model.evaluate(X_ocd_test_scaled, y_ocd_test)
print(f"\nHeld-Out OCD Accuracy (low samples): {acc:.4f}")
