# EEG Sleep Stage Classification Challenge
# TSAC, 2024/2025

"""
This notebook implements a state-of-the-art solution for classifying EEG sleep stages.
Classes: Wake, NREM (E1, E2, E3), and REM

The approach combines several high-performing techniques:
1. Advanced feature extraction
2. Multiple machine learning models
3. Gradient boosting and deep learning
4. Stacked ensemble for maximum accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import time
from scipy import signal, stats
from scipy.fft import fft
import pywt
import warnings
import os
from google.colab import drive
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
DATA_PATH = '/content/drive/MyDrive/EEG_Data/'
RESULTS_DIR = '/content/drive/MyDrive/EEG_Models/'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Helper functions for feature extraction

def extract_statistical_features(X):
    """Extract statistical features from time series data."""
    features = []
    
    for i in range(X.shape[0]):
        x = X[i]
        
        # Basic statistics
        mean = np.mean(x)
        std = np.std(x)
        minimum = np.min(x)
        maximum = np.max(x)
        median = np.median(x)
        skewness = stats.skew(x)
        kurtosis = stats.kurtosis(x)
        
        # Percentiles
        q25 = np.percentile(x, 25)
        q75 = np.percentile(x, 75)
        iqr = q75 - q25
        
        # Zero crossings and peaks
        zero_crossings = np.sum(np.diff(np.signbit(x)))
        peaks, _ = signal.find_peaks(x)
        n_peaks = len(peaks)
        
        # Energy and power
        energy = np.sum(x**2)
        rms = np.sqrt(np.mean(x**2))
        
        # First and second derivatives
        dx = np.diff(x)
        dx_mean = np.mean(np.abs(dx))
        dx_std = np.std(dx)
        
        if len(dx) > 1:
            d2x = np.diff(dx)
            d2x_mean = np.mean(np.abs(d2x))
            d2x_std = np.std(d2x)
        else:
            d2x_mean = 0
            d2x_std = 0
        
        # Autocorrelation features
        if len(x) > 1:
            autocorr = np.correlate(x, x, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr_decay = np.sum(np.diff(autocorr[:min(10, len(autocorr))]))
        else:
            autocorr_decay = 0
        
        row_features = [
            mean, std, minimum, maximum, median, 
            q25, q75, iqr, skewness, kurtosis,
            zero_crossings, n_peaks, energy, rms,
            dx_mean, dx_std, d2x_mean, d2x_std, autocorr_decay
        ]
        
        features.append(row_features)
    
    column_names = [
        'mean', 'std', 'min', 'max', 'median', 
        'q25', 'q75', 'iqr', 'skewness', 'kurtosis',
        'zero_crossings', 'n_peaks', 'energy', 'rms',
        'dx_mean', 'dx_std', 'd2x_mean', 'd2x_std', 'autocorr_decay'
    ]
    
    return np.array(features), column_names

def extract_frequency_features(X, fs=100):
    """Extract frequency domain features."""
    features = []
    
    # Define frequency bands (Hz)
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 50)
    }
    
    for i in range(X.shape[0]):
        x = X[i]
        
        # Compute FFT
        n = len(x)
        fft_vals = np.abs(fft(x))
        fft_freq = np.fft.fftfreq(n, d=1/fs)
        
        # Only take positive frequencies
        fft_vals = fft_vals[:n//2]
        fft_freq = fft_freq[:n//2]
        
        # Power spectral density
        psd = fft_vals**2 / n
        
        # Total power
        total_power = np.sum(psd)
        
        # Band powers
        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (fft_freq >= low) & (fft_freq <= high)
            band_power = np.sum(psd[band_mask])
            band_powers[band_name] = band_power
            band_powers[f'{band_name}_rel'] = band_power / total_power if total_power > 0 else 0
        
        # Spectral edge frequency (95% of power)
        cumulative_power = np.cumsum(psd)
        if np.sum(psd) > 0:
            spectral_edge_95 = fft_freq[np.where(cumulative_power >= 0.95 * cumulative_power[-1])[0][0]]
        else:
            spectral_edge_95 = 0
        
        # Mean frequency and spectral entropy
        if len(fft_freq) > 0 and np.sum(psd) > 0:
            mean_freq = np.sum(fft_freq * psd) / np.sum(psd)
            norm_psd = psd / np.sum(psd)
            spectral_entropy = -np.sum(norm_psd * np.log2(norm_psd + 1e-10))
        else:
            mean_freq = 0
            spectral_entropy = 0
        
        row_features = [
            total_power, 
            band_powers['delta'], band_powers['delta_rel'],
            band_powers['theta'], band_powers['theta_rel'],
            band_powers['alpha'], band_powers['alpha_rel'],
            band_powers['beta'], band_powers['beta_rel'],
            band_powers['gamma'], band_powers['gamma_rel'],
            spectral_edge_95, mean_freq, spectral_entropy
        ]
        
        features.append(row_features)
    
    column_names = [
        'total_power', 
        'delta_power', 'delta_power_rel',
        'theta_power', 'theta_power_rel',
        'alpha_power', 'alpha_power_rel',
        'beta_power', 'beta_power_rel',
        'gamma_power', 'gamma_power_rel',
        'spectral_edge_95', 'mean_freq', 'spectral_entropy'
    ]
    
    return np.array(features), column_names

def extract_wavelet_features(X):
    """Extract wavelet-based features."""
    features = []
    
    for i in range(X.shape[0]):
        x = X[i]
        
        # Apply wavelet decomposition
        coeffs = pywt.wavedec(x, 'db4', level=4)
        
        # Extract features from each level
        wavelet_features = []
        for coef in coeffs:
            wavelet_features.extend([
                np.mean(np.abs(coef)),
                np.std(coef),
                np.max(np.abs(coef)),
                np.sum(coef**2)
            ])
        
        features.append(wavelet_features)
    
    # Generate column names
    column_names = []
    for i, level_name in enumerate(['approx'] + [f'detail_{j}' for j in range(4, 0, -1)]):
        for stat in ['mean_abs', 'std', 'max_abs', 'energy']:
            column_names.append(f'wavelet_{level_name}_{stat}')
    
    return np.array(features), column_names

def extract_hjorth_features(X):
    """Extract Hjorth parameters (activity, mobility, complexity)."""
    features = []
    
    for i in range(X.shape[0]):
        x = X[i]
        
        # First derivative
        dx = np.diff(x, prepend=x[0])
        
        # Second derivative
        d2x = np.diff(dx, prepend=dx[0])
        
        # Activity (variance of the signal)
        activity = np.var(x)
        
        # Mobility
        mobility = np.sqrt(np.var(dx) / np.var(x)) if np.var(x) > 0 else 0
        
        # Complexity
        complexity = np.sqrt(np.var(d2x) / np.var(dx)) / mobility if mobility > 0 and np.var(dx) > 0 else 0
        
        features.append([activity, mobility, complexity])
    
    column_names = ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']
    
    return np.array(features), column_names

def extract_features(X):
    """Extract all features and combine them."""
    stat_features, stat_names = extract_statistical_features(X)
    freq_features, freq_names = extract_frequency_features(X)
    wavelet_features, wavelet_names = extract_wavelet_features(X)
    hjorth_features, hjorth_names = extract_hjorth_features(X)
    
    # Combine all features
    all_features = np.hstack([stat_features, freq_features, wavelet_features, hjorth_features])
    all_names = stat_names + freq_names + wavelet_names + hjorth_names
    
    return all_features, all_names

def preprocess_data(X_train, y_train=None, X_test=None, feature_extraction=True):
    """Preprocess data with robust scaling and feature extraction."""
    
    # Initialize scalers and encoders
    scaler = RobustScaler()
    le = None
    
    if y_train is not None:
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
    
    # If we only have labels to process
    if X_train is None:
        if y_train is not None:
            return None, y_train_encoded, None, le, None
        return None, None, None, None, None
    
    # Scale the input data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Extract features if requested
    if feature_extraction:
        X_train_features, feature_names = extract_features(X_train_scaled)
    else:
        X_train_features = X_train_scaled
        feature_names = [f"feature_{i}" for i in range(X_train_scaled.shape[1])]
    
    # Process test data if provided
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        
        if feature_extraction:
            X_test_features, _ = extract_features(X_test_scaled)
        else:
            X_test_features = X_test_scaled
        
        if y_train is not None:
            return X_train_features, y_train_encoded, X_test_features, le, feature_names
        else:
            return X_train_features, X_test_features, feature_names
    
    if y_train is not None:
        return X_train_features, y_train_encoded, feature_names
    else:
        return X_train_features, feature_names

# Model definitions

def create_cnn_model(input_shape, num_classes):
    """Create a 1D CNN model for classification."""
    model = Sequential([
        Input(shape=input_shape),
        
        # First convolutional block
        Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Second convolutional block
        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Third convolutional block
        Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Output layers
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_ensemble_models():
    """Create a set of diverse models for ensemble learning."""
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            n_jobs=-1,
            random_state=SEED
        ),
        
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt',
            random_state=SEED
        ),
        
        'xgboost': xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=2,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=0.01,
            scale_pos_weight=1,
            objective='multi:softprob',
            n_jobs=-1,
            random_state=SEED
        ),
        
        'lightgbm': lgb.LGBMClassifier(
            n_estimators=200,        # Increased back to 200
            learning_rate=0.05,      # Balanced learning rate
            max_depth=5,             # Moderate depth
            num_leaves=31,           # Back to default
            min_child_samples=30,    # Reduced to allow more splits
            subsample=0.8,           # Back to default
            colsample_bytree=0.8,    # Back to default
            reg_alpha=0.05,          # Reduced regularization
            reg_lambda=0.05,         # Reduced regularization
            min_split_gain=0.001,    # Reduced minimum gain
            min_child_weight=10,     # Reduced minimum weight
            force_col_wise=True,     # Force column-wise processing
            boosting_type='gbdt',    # Explicit boosting type
            importance_type='gain',  # Use gain for feature importance
            verbose=-1,              # Reduce verbosity
            n_jobs=-1,
            random_state=SEED
        ),
        
        'svm': SVC(
            C=10.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=SEED
        ),
        
        'mlp': MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=SEED
        )
    }
    
    return models

def cross_val_model_selection(X_train, y_train, models, cv=5):
    """Select best models using cross-validation."""
    print("Evaluating models using cross-validation...")
    
    results = {}
    for name, model in models.items():
        start_time = time.time()
        
        # Run cross-validation
        scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED),
            scoring='accuracy',
            n_jobs=-1 if name != 'mlp' else 1  # Avoid parallelization issues with MLPClassifier
        )
        
        # Store results
        results[name] = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'time': time.time() - start_time
        }
        
        print(f"{name}: Accuracy={scores.mean():.4f} (±{scores.std():.4f}), Time={results[name]['time']:.2f}s")
    
    return results

def train_stacked_ensemble(X_train, y_train, X_val, y_val, models):
    """Train a stacked ensemble of models."""
    print("Training stacked ensemble...")
    
    # Train each base model
    base_models = {}
    base_predictions_train = np.zeros((X_train.shape[0], len(models), len(np.unique(y_train))))
    base_predictions_val = np.zeros((X_val.shape[0], len(models), len(np.unique(y_train))))
    
    for i, (name, model) in enumerate(models.items()):
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        base_models[name] = model
        
        # Get probability predictions
        if hasattr(model, "predict_proba"):
            base_predictions_train[:, i, :] = model.predict_proba(X_train)
            base_predictions_val[:, i, :] = model.predict_proba(X_val)
        else:
            # For models without predict_proba, use decision_function if available
            if hasattr(model, "decision_function"):
                decision_values = model.decision_function(X_train)
                if decision_values.ndim == 1:  # Binary case
                    # Convert to probabilities via sigmoid
                    pos_probs = 1 / (1 + np.exp(-decision_values))
                    base_predictions_train[:, i, 0] = 1 - pos_probs
                    base_predictions_train[:, i, 1] = pos_probs
                    
                    val_decision_values = model.decision_function(X_val)
                    val_pos_probs = 1 / (1 + np.exp(-val_decision_values))
                    base_predictions_val[:, i, 0] = 1 - val_pos_probs
                    base_predictions_val[:, i, 1] = val_pos_probs
                else:  # Multi-class case
                    # Convert to probabilities via softmax
                    exp_values = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
                    base_predictions_train[:, i, :] = exp_values / np.sum(exp_values, axis=1, keepdims=True)
                    
                    val_exp_values = np.exp(model.decision_function(X_val) - 
                                         np.max(model.decision_function(X_val), axis=1, keepdims=True))
                    base_predictions_val[:, i, :] = val_exp_values / np.sum(val_exp_values, axis=1, keepdims=True)
    
    # Reshape for meta-model
    meta_X_train = base_predictions_train.reshape(X_train.shape[0], -1)
    meta_X_val = base_predictions_val.reshape(X_val.shape[0], -1)
    
    # Train meta-model
    meta_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        n_jobs=-1,
        random_state=SEED
    )
    
    meta_model.fit(meta_X_train, y_train)
    
    # Evaluate ensemble
    val_pred = meta_model.predict(meta_X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Stacked Ensemble Validation Accuracy: {val_accuracy:.4f}")
    
    return base_models, meta_model

def train_deep_learning_model(X_train, y_train, X_val, y_val):
    """Train CNN model on raw time series data."""
    print("Training CNN model...")
    
    # Reshape input for CNN
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    # Define model
    num_classes = len(np.unique(y_train))
    cnn_model = create_cnn_model((X_train.shape[1], 1), num_classes)
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    history = cnn_model.fit(
        X_train_reshaped, y_train,
        validation_data=(X_val_reshaped, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, lr_reducer],
        verbose=1
    )
    
    # Evaluate
    _, val_accuracy = cnn_model.evaluate(X_val_reshaped, y_val, verbose=0)
    print(f"CNN Model Validation Accuracy: {val_accuracy:.4f}")
    
    return cnn_model, history

def final_ensemble_prediction(X, base_models, meta_model, cnn_model, le=None):
    """Make predictions using the full ensemble."""
    # Get base model predictions
    base_predictions = np.zeros((X.shape[0], len(base_models), meta_model.n_classes_))
    
    for i, (name, model) in enumerate(base_models.items()):
        if hasattr(model, "predict_proba"):
            base_predictions[:, i, :] = model.predict_proba(X)
        else:
            # Handle models without predict_proba
            if hasattr(model, "decision_function"):
                decision_values = model.decision_function(X)
                if decision_values.ndim == 1:  # Binary case
                    pos_probs = 1 / (1 + np.exp(-decision_values))
                    base_predictions[:, i, 0] = 1 - pos_probs
                    base_predictions[:, i, 1] = pos_probs
                else:  # Multi-class case
                    exp_values = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
                    base_predictions[:, i, :] = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    # Reshape for meta-model
    meta_X = base_predictions.reshape(X.shape[0], -1)
    
    # Get CNN predictions
    # Reshape input to match the CNN's expected input shape
    X_reshaped = X.reshape(X.shape[0], -1, 1)
    cnn_pred_proba = cnn_model.predict(X_reshaped, verbose=0)
    
    # Combine meta-model and CNN predictions (weighted average)
    meta_pred_proba = meta_model.predict_proba(meta_X)
    
    # Use 60% weight for meta-model and 40% for CNN
    final_pred_proba = 0.6 * meta_pred_proba + 0.4 * cnn_pred_proba
    final_pred = np.argmax(final_pred_proba, axis=1)
    
    # Convert to original labels if label encoder is provided
    if le is not None:
        final_pred = le.inverse_transform(final_pred)
    
    return final_pred, final_pred_proba

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
    plt.close()

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        idx = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importance')
        plt.bar(range(top_n), importance[idx])
        plt.xticks(range(top_n), [feature_names[i] for i in idx], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/feature_importance.png")
        plt.close()

def evaluate_model(y_true, y_pred, class_names):
    """Evaluate model performance with multiple metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    
    # Convert class names to strings if they're not already
    if isinstance(class_names[0], (np.integer, int)):
        class_names = [str(i) for i in class_names]
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open(f"{RESULTS_DIR}/classification_report.txt", 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    return accuracy

def main():
    """Main execution function."""
    print("Starting EEG Sleep Stage Classification...")
    
    try:
        # Load data
        print("Loading data...")
        data = pd.read_csv(f"{DATA_PATH}train_data.csv", header=None)
        
        # Separate features and target
        y = data.iloc[:, 0].values
        X = data.iloc[:, 1:].values
        
        print(f"Loaded data with shape: {X.shape}")
        print(f"Class distribution: {np.unique(y, return_counts=True)}")
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        
        # Preprocess data
        print("Preprocessing data and extracting features...")
        X_train_features, y_train_encoded, X_val_features, le, feature_names = preprocess_data(
            X_train, y_train, X_val
        )
        
        # Encode validation labels
        y_val_encoded = le.transform(y_val)
        
        # Get class names as strings
        class_names = [str(label) for label in le.classes_]
        print(f"Classes: {class_names}")
        
        # Create ensemble models
        models = create_ensemble_models()
        
        # Evaluate models with cross-validation
        cv_results = cross_val_model_selection(X_train_features, y_train_encoded, models)
        
        # Train stacked ensemble
        base_models, meta_model = train_stacked_ensemble(
            X_train_features, y_train_encoded, X_val_features, y_val_encoded, models
        )
        
        # Train CNN on raw time series
        num_classes = len(class_names)
        input_shape = (X_train_features.shape[1], 1)  # Use the same shape as the feature data
        cnn_model = create_cnn_model(input_shape, num_classes)
        
        # Reshape data for CNN
        X_train_reshaped = X_train_features.reshape(X_train_features.shape[0], X_train_features.shape[1], 1)
        X_val_reshaped = X_val_features.reshape(X_val_features.shape[0], X_val_features.shape[1], 1)
        
        # Train CNN
        cnn_model.fit(
            X_train_reshaped, y_train_encoded,
            validation_data=(X_val_reshaped, y_val_encoded),
            epochs=100,
            batch_size=32,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ],
            verbose=1
        )
        
        # Final evaluation on validation set
        print("Generating final predictions...")
        val_preds, _ = final_ensemble_prediction(
            X_val_features, base_models, meta_model, cnn_model
        )
        
        # Evaluate final model
        print("\nFinal Model Evaluation:")
        val_accuracy = evaluate_model(y_val_encoded, val_preds, class_names)
        
        # Plot feature importance for interpretability
        plot_feature_importance(models['random_forest'], feature_names)
        
        # Save models
        import pickle
        with open(f"{RESULTS_DIR}/base_models.pkl", 'wb') as f:
            pickle.dump(base_models, f)
        
        with open(f"{RESULTS_DIR}/meta_model.pkl", 'wb') as f:
            pickle.dump(meta_model, f)
        
        cnn_model.save(f"{RESULTS_DIR}/cnn_model.h5")
        
        with open(f"{RESULTS_DIR}/label_encoder.pkl", 'wb') as f:
            pickle.dump(le, f)
        
        print("Training complete! All models saved.")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

def predict(test_file, output_file):
    """Generate predictions for test data using trained models."""
    print(f"Generating predictions for {test_file}...")
    
    try:
        # Import pickle at the start of the function
        import pickle
        from tensorflow.keras.models import load_model
        
        # Check if models exist
        if not os.path.exists(f"{RESULTS_DIR}/base_models.pkl"):
            print("Error: Models not found. Please run training first.")
            return
        
        # Load test data
        test_data = pd.read_csv(test_file, header=None)
        X_test = test_data.iloc[:, 1:].values
        
        # Load models and preprocessing objects
        with open(f"{RESULTS_DIR}/base_models.pkl", 'rb') as f:
            base_models = pickle.load(f)
        
        with open(f"{RESULTS_DIR}/meta_model.pkl", 'rb') as f:
            meta_model = pickle.load(f)
        
        with open(f"{RESULTS_DIR}/label_encoder.pkl", 'rb') as f:
            le = pickle.load(f)
            
        # Load CNN model
        cnn_model = load_model(f"{RESULTS_DIR}/cnn_model.h5")
        
        # Preprocess test data
        # When only processing test data, preprocess_data returns (X_test_processed, feature_names)
        X_test_processed, feature_names = preprocess_data(X_test)
        
        # Generate predictions
        test_preds, test_probs = final_ensemble_prediction(
            X_test_processed, base_models, meta_model, cnn_model
        )
        
        # Convert predictions back to original labels
        test_preds_original = le.inverse_transform(test_preds)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'predicted_class': test_preds_original,
            'confidence': np.max(test_probs, axis=1)
        })
        predictions_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run enhanced version
    main()
    
    # Example prediction on test data
    test_file = f"{DATA_PATH}test_data.csv"
    output_file = f"{DATA_PATH}predictions.csv"
    
    if os.path.exists(test_file):
        predict(test_file, output_file) 
    else:
        print(f"Test file {test_file} not found. Skipping prediction step.")
