#%%
import os
from scapy.all import rdpcap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import resample

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

#%%

data.columns

#%%

from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

data.columns = data.columns.str.strip()
data = data.query("Label != 'Heartbleed' & Label != 'Infiltration'")

# Ratio-Based Features
data['Fwd to Bwd Packet Ratio'] = data['Total Fwd Packets'] / (data['Total Backward Packets'] + 1e-9)
data['Bwd to Fwd Packet Ratio'] = data['Total Backward Packets'] / (data['Total Fwd Packets'] + 1e-9)
data['Fwd to Bwd Length Ratio'] = data['Total Length of Fwd Packets'] / (data['Total Length of Bwd Packets'] + 1e-9)
data['Bwd to Fwd Length Ratio'] = data['Total Length of Bwd Packets'] / (data['Total Length of Fwd Packets'] + 1e-9)

# Statistical Features
data['Total Packet Length'] = data['Total Length of Fwd Packets'] + data['Total Length of Bwd Packets']
data['Total Packet Count'] = data['Total Fwd Packets'] + data['Total Backward Packets']
data['Flow IAT Range'] = data['Flow IAT Max'] - data['Flow IAT Min']

# Flag Ratios
data['FIN Ratio'] = data['FIN Flag Count'] / (data['Total Packet Count'] + 1e-9)
data['SYN Ratio'] = data['SYN Flag Count'] / (data['Total Packet Count'] + 1e-9)
data['ACK Ratio'] = data['ACK Flag Count'] / (data['Total Packet Count'] + 1e-9)
data['PSH Ratio'] = data['PSH Flag Count'] / (data['Total Packet Count'] + 1e-9)
data['URG Ratio'] = data['URG Flag Count'] / (data['Total Packet Count'] + 1e-9)

# Time-Based Features
data['Idle Time Range'] = data['Idle Max'] - data['Idle Min']
data['Active Time Range'] = data['Active Max'] - data['Active Min']
data['IAT Range'] = data['Flow IAT Max'] - data['Flow IAT Min']
data['Activity-to-Idle Ratio'] = (data['Active Max'] + data['Active Min']) / (data['Idle Max'] + data['Idle Min'] + 1e-9)

# Flow Characteristics
data['Packets Per Flow'] = data['Total Packet Count'] / (data['Flow Duration'] + 1e-9)
data['Flow Header to Length Ratio'] = (data['Fwd Header Length'] + data['Bwd Header Length']) / (data['Total Packet Length'] + 1e-9)

# Interaction Features
data['Flow Intensity'] = data['Flow Bytes/s'] * data['Flow Packets/s']
data['Packet Size Intensity'] = data['Avg Fwd Segment Size'] * data['Avg Bwd Segment Size']
data['Avg Packet Length'] = (data['Total Length of Fwd Packets'] + data['Total Length of Bwd Packets']) / data['Total Packet Count']
data['Header to Packet Length Ratio'] = (data['Fwd Header Length'] + data['Bwd Header Length']) / data['Total Packet Length']
data['Flow Intensity'] = data['Flow Bytes/s'] * data['Flow Packets/s']


# Remove any rows with NaN or infinite values after feature engineering
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)


# Separate the BENIGN class (label: 0) and other classes
benign_data = data[data['Label'] == 'BENIGN']
non_benign_data = data[data['Label'] != 'BENIGN']

# Downsample the BENIGN class
benign_downsampled = resample(
    benign_data,
    replace=False,  # Sample without replacement
    n_samples=int(len(benign_data) * 0.25),  # Keep 25% of BENIGN samples
    random_state=42
)

# Combine downsampled BENIGN class with other classes
data_balanced = pd.concat([benign_downsampled, non_benign_data], axis=0)

# Extract features and labels
X = data_balanced.drop(columns=['Label'])  # Features
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.mean(), inplace=True)  # Replace NaN with column means
y = data_balanced['Label']  # Target labels

# Encode the target labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the original data into training/validation and holdout sets to have data to predict that was never viewed by the model during training
X_initial_train, X_holdout, y_initial_train, y_holdout = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# Apply SMOTE only to the training/validation data
X_resampled, y_resampled = SMOTE(random_state=43).fit_resample(X_initial_train, y_initial_train)

# Downsample the resampled data
downsample_indices = np.random.choice(len(X_resampled), size=250000, replace=False)
X_resampled = X_resampled[downsample_indices]
y_resampled = y_resampled[downsample_indices]

# Split the resampled data into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(
    X_resampled, y_resampled, test_size=0.02, random_state=42
)

#%%

# Build a simplified neural network with regularization
model = Sequential([
    Dense(128, activation='swish', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='swish', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(32, activation='swish', kernel_regularizer=l2(0.01)),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

#%%

# Compile the model with an adjusted learning rate
opt = Adam(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

#%%

# Add early stopping and learning rate scheduler
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=12,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001
)

# Train the model
print("Training the neural network with early stopping and learning rate scheduler...")
history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=500,
    validation_data=(X_validation, y_validation),
    verbose=2,
    callbacks=[early_stopping, reduce_lr]
)


#%%

# Evaluate the model on the holdout set
y_holdout_pred_prob = model.predict(X_holdout)
y_holdout_pred = np.argmax(y_holdout_pred_prob, axis=1)

print("Holdout Set Classification Report:")
print(classification_report(y_holdout, y_holdout_pred, target_names=label_encoder.classes_))

#%%
conf_matrix_holdout = confusion_matrix(y_holdout, y_holdout_pred)
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(conf_matrix_holdout, annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
plt.title("Holdout Set Confusion Matrix", fontsize=16)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.show()

#%%

