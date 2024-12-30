#%%
import os
from scapy.all import rdpcap
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

#%%
# Path to the folder containing the CSV files
DATA_FOLDER = "MachineLearningCVE"

# Function to load all CSV files and combine into a single DataFrame
def load_csv_files(data_folder):
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    dataframes = []

    for csv_file in csv_files:
        file_path = os.path.join(data_folder, csv_file)
        print(f"Loading {file_path}...")
        df = pd.read_csv(file_path)
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)

# Load the data
data = load_csv_files(DATA_FOLDER)

# Explore the data structure
print("Data Columns:", data.columns)
print("Sample Rows:")
print(data.head())

#%%

data.columns

#%%

data.columns = data.columns.str.strip() 
X = data.drop(columns=['Label'])  # Features
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)  # Replace NaN with column means
y = data['Label']  # Target labels
# Encode the target labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

#%%

# Build a Neural Network with TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='sigmoid', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='swish'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='swish'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='relu')
])

#%%

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%

# Add early stopping and learning rate scheduler callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,  
    patience=3,  # Number of epochs with no improvement before reducing the learning rate
    min_lr=1e-6  # Minimum learning rate
)

# Train the model
print("Training the neural network with early stopping and learning rate scheduler...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=100,  # Set a high number of epochs; early stopping will prevent overtraining
    batch_size=128, 
    validation_split=0.2, 
    verbose=2, 
    callbacks=[early_stopping, reduce_lr]
)

#%%

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {accuracy}")

#%%
# Generate predictions and classification report
y_pred = np.argmax(model.predict(X_test), axis=1)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

#%%

