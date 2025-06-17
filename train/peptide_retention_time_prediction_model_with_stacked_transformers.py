
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from typing import List
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import json
import argparse # Import argparse for command-line argument parsing

"""# Set random seeds for reproducibility"""
tf.random.set_seed(42)
np.random.seed(42)

"""## Logging Configuration"""
# Configure basic logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""## Parameters"""
# These parameters will now be assigned from argparse if provided, otherwise defaults are used.
MAX_PEPTIDE_LENGTH = 52 # Maximum length of the peptide sequences
EMBED_DIM = 32 # Embedding size for each token
NUM_HEADS = 8 # Number of attention heads
FF_DIM = 64 # Hidden layer size in feed forward network inside transformer
BATCH_SIZE = 32
EPOCH = 500
DROPOUT_RATE = 0.1
NUM_TRANSFORMER_BLOCKS = 2 # Number of stacked Transformer blocks

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

"""## Convert peptide sequences to integers"""

# Define the amino acid vocabulary and mappings
AA_VOCAB = ['<PAD>', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'm', 'c',
            '<STOP>','<UNK>']
VOCAB_SIZE = len(AA_VOCAB) # Correctly derive VOCAB_SIZE from the vocabulary list
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_VOCAB)}

def peptide_to_indices(sequence: str) -> List[int]:
    """
    Converts a peptide sequence string to a list of integer indices,
    using the AA_TO_INDEX mapping. Pads the sequence with <PAD>
    if it is shorter than MAX_PEPTIDE_LENGTH. Unknown amino acids are
    mapped to '<UNK>'.

    Args:
        sequence: The peptide sequence string (e.g., "PEPTIDE").

    Returns:
        A list of integer indices representing the sequence, padded to
        MAX_PEPTIDE_LENGTH.
    """
    indices = [AA_TO_INDEX.get(aa, AA_TO_INDEX['<UNK>']) for aa in sequence]

    # Truncate if too long
    if len(indices) > MAX_PEPTIDE_LENGTH:
        indices = indices[:MAX_PEPTIDE_LENGTH]

    # Pad if too short
    padding_length = MAX_PEPTIDE_LENGTH - len(indices)
    indices.extend([AA_TO_INDEX['<PAD>']] * padding_length)
    return np.array(indices, dtype=np.int32)

"""## Load the dataset and preprocess"""

# Fixed min/max for RT scaling, as per original code's intent
min_rt, max_rt = 10, 90

def min_max_scale(x, min_val, max_val):
    """
    Scales a value x to a range [0, 1] using fixed min_val and max_val.
    """
    new_x = 1.0 * (x - min_val) / (max_val - min_val)
    return new_x

def inverse_min_max_scale(scaled_x, min_val, max_val):
    """
    Inverses the min-max scaling to convert a scaled value back to its original range.
    """
    return scaled_x * (max_val - min_val) + min_val

def load_and_preprocess_data(FilePath="..\Data\DataPreProcessing\liverpool_all_data.csv"):
    """
    Loads peptide data from a CSV, filters peptides by maximum length,
    and scales retention times using a fixed min/max range.

    Args:
        FilePath (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Processed DataFrame with 'aligned_rt' column.
    """
    logger.info(f"Loading data from: {FilePath}")
    data = pd.read_csv(FilePath)
    initial_rows = len(data)

    # Filter out peptides longer than MAX_PEPTIDE_LENGTH
    data = data[data.Peptide.apply(len) <= MAX_PEPTIDE_LENGTH]
    logger.info(f"Filtered {initial_rows - len(data)} rows due to peptide length > {MAX_PEPTIDE_LENGTH}.")

    _min_rt_data, _max_rt_data = data.rt.min(), data.rt.max()
    logger.info(f"Original RT range in data: {_min_rt_data:.2f} to {_max_rt_data:.2f}")

    # Apply min-max scaling using the fixed global bounds
    data['aligned_rt'] = [min_max_scale(x, min_rt, max_rt) for x in data.rt]
    logger.info(f"RTs scaled using fixed bounds ({min_rt}, {max_rt}). Scaled range: {data.aligned_rt.min():.2f} to {data.aligned_rt.max():.2f}")

    return data

"""## Model Architecture"""

@tf.keras.utils.register_keras_serializable(package='TransformerBlock')
class TransformerBlock(layers.Layer):
    """
    A single Transformer Block composed of multi-head self-attention
    and a position-wise feed-forward network, with residual connections
    and layer normalization.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Multi-head self-attention layer
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )

        # Feed-forward network with GELU activation
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
        ])

        # Layer normalization and dropout layers for residual connections
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        # Apply self-attention with a residual connection
        attn_output = self.attention(inputs, inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Apply feed-forward network with a residual connection
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        # Implement get_config to enable serialization of the custom layer
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config

@tf.keras.utils.register_keras_serializable(package='TokenAndPositionEmbedding')
class TokenAndPositionEmbedding(layers.Layer):
    """
    Combines token embeddings and positional embeddings for sequence input.
    Supports masking for padding tokens.
    """
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            mask_zero=True  # Enable masking for padding tokens
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.dropout = layers.Dropout(0.1)

    def call(self, x, training=False):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        output = x + positions
        return self.dropout(output, training=training)

    def get_config(self):
        # Implement get_config to enable serialization of the custom layer
        config = super().get_config()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        })
        return config

def build_model():
    """
    Builds the Transformer-based peptide retention time prediction model.
    The model consists of a token and position embedding layer,
    stacked Transformer blocks, global average pooling, and a
    feed-forward regression head.
    """
    inputs_pep = layers.Input(shape=(MAX_PEPTIDE_LENGTH,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(MAX_PEPTIDE_LENGTH, VOCAB_SIZE, EMBED_DIM)
    x = embedding_layer(inputs_pep)

    # --- START OF STACKED TRANSFORMER BLOCKS ---
    logger.info(f"Building model with {NUM_TRANSFORMER_BLOCKS} stacked Transformer Blocks.")
    for i in range(NUM_TRANSFORMER_BLOCKS):
        logger.debug(f"Adding Transformer Block {i+1}")
        transformer_block = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT_RATE)
        x = transformer_block(x)
    # --- END OF STACKED TRANSFORMER BLOCKS ---

    x = layers.GlobalAveragePooling1D()(x) # Condense sequence info into a fixed-size vector
    x = layers.Dropout(DROPOUT_RATE)(x) # Apply dropout after pooling

    # Feed-Forward Downstream (FFD) regression head
    x = layers.Dense(512, activation="gelu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(1, activation="linear")(x) # Output a single regression value

    model = keras.Model(inputs=inputs_pep, outputs=outputs)
    return model

"""## Callbacks and Metadata Saving"""

class TrainingHistoryLogger(keras.callbacks.Callback):
    """
    Custom Keras callback to log training progress and save detailed history
    to a CSV file.
    """
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.history_data = []
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"TrainingHistoryLogger initialized. Logs will be saved to: {log_dir}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_msg = f"Epoch {epoch + 1:3d}/{self.params['epochs']} - "
        log_msg += " - ".join([f"{k}: {v:.6f}" for k, v in logs.items()])
        logger.info(log_msg) # Use the configured logger

        # Store history data
        epoch_data = {'epoch': epoch + 1}
        epoch_data.update(logs)
        self.history_data.append(epoch_data)

        # Save history to CSV every 10 epochs or at the end
        if (epoch + 1) % 10 == 0:
            self.save_history()

    def on_train_end(self, logs=None):
        # Final save of history at the end of training
        self.save_history()
        logger.info(f"Training completed. Final history saved to {self.log_dir}")

    def save_history(self):
        """Save current training history data to a CSV file."""
        if self.history_data:
            df = pd.DataFrame(self.history_data)
            csv_path = os.path.join(self.log_dir, f'training_history_{TIMESTAMP}.csv')
            df.to_csv(csv_path, index=False)
            logger.debug(f"History saved to {csv_path}")

def create_callbacks():
    """
    Creates a list of Keras callbacks for training, including custom logging,
    learning rate scheduling, early stopping, model checkpointing,
    TensorBoard integration, and CSV logging.
    """
    log_dir = f"logs/training_{TIMESTAMP}"
    model_dir = f"models/model_{TIMESTAMP}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger.info(f"Creating callbacks with log_dir: {log_dir}")
    logger.info(f"Model checkpoints will be saved to: {model_dir}")

    callbacks = [
        TrainingHistoryLogger(log_dir), # Custom detailed history logger
        # keras.callbacks.ReduceLROnPlateau(
        #     monitor='val_loss',
        #     factor=0.5,
        #     patience=10, # Reduce LR if val_loss doesn't improve for 10 epochs
        #     min_lr=1e-7,
        #     verbose=1
        # ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=100, # Stop training if val_loss doesn't improve for 20 epochs
            restore_best_weights=True, # Restore model weights from the epoch with the best val_loss
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'), # Recommended .keras extension for Keras 3+
            monitor='val_loss',
            save_best_only=True, # Only save the model if validation loss improves
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'model_epoch_{epoch:03d}.keras'),
            save_freq='epoch', # Save model at the end of every epoch
            verbose=0
        ),
        keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=5, # Log histograms of weights and biases every 5 epochs
            write_graph=True, # Write the model graph to TensorBoard
            write_images=True, # Write model weights as images
            update_freq='epoch' # Log metrics every epoch
        ),
        keras.callbacks.CSVLogger(
            os.path.join(log_dir, f'keras_training_log_{TIMESTAMP}.csv'),
            append=True # Append to the file if it already exists
        )
    ]
    return callbacks

def save_training_metadata(model, history, rt_scaling_info, data_info, test_results):
    """
    Saves comprehensive training metadata including model architecture,
    training configuration, RT scaling parameters, and test results.

    Args:
        model (tf.keras.Model): The trained Keras model.
        history (tf.keras.callbacks.History): The history object returned by model.fit.
        rt_scaling_info (dict): Dictionary containing min/max values used for RT scaling.
        data_info (dict): Dictionary with information about the dataset splits.
        test_results (dict): Dictionary with evaluation metrics on the test set.
    """
    metadata_dir = f"logs/training_{TIMESTAMP}"
    os.makedirs(metadata_dir, exist_ok=True)
    logger.info(f"Saving training metadata to: {metadata_dir}")

    # Save model architecture
    try:
        model_json = model.to_json()
        with open(os.path.join(metadata_dir, 'model_architecture.json'), 'w') as f:
            f.write(model_json)
        logger.debug("Model architecture saved.")
    except Exception as e:
        logger.warning(f"Could not save model architecture to JSON: {e}")

    # Save model summary
    with open(os.path.join(metadata_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    logger.debug("Model summary saved.")

    # Save RT scaling parameters
    with open(os.path.join(metadata_dir, 'rt_scaling_parameters.json'), 'w') as f:
        json.dump(rt_scaling_info, f, indent=2)
    logger.debug("RT scaling parameters saved.")

    # Save training configuration
    config = {
        'timestamp': TIMESTAMP,
        'max_peptide_length': MAX_PEPTIDE_LENGTH,
        'embed_dim': EMBED_DIM,
        'num_heads': NUM_HEADS,
        'ff_dim': FF_DIM,
        'dropout_rate': DROPOUT_RATE,
        'num_transformer_blocks': NUM_TRANSFORMER_BLOCKS, # Include new parameter
        'batch_size': BATCH_SIZE,
        'vocab_size': VOCAB_SIZE,
        'vocab': AA_VOCAB,
        'data_info': data_info,
        'test_results': test_results
    }
    with open(os.path.join(metadata_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    logger.debug("Training configuration saved.")

    # Save final training history as CSV
    if hasattr(history, 'history'):
        history_df = pd.DataFrame(history.history)
        history_df['epoch'] = range(1, len(history_df) + 1)
        history_df.to_csv(os.path.join(metadata_dir, f'final_training_history_{TIMESTAMP}.csv'), index=False)
        logger.debug("Final training history saved.")

def analyze_predictions(model, X_test, y_test, min_rt_bound, max_rt_bound):
    """
    Analyzes model predictions by converting them back to the original RT scale,
    calculating key regression metrics, and generating diagnostic plots.
    The plots and prediction results are saved to disk.

    Args:
        model (tf.keras.Model): The trained Keras model.
        X_test (np.ndarray): Test features (peptide indices).
        y_test (np.ndarray): True test labels (scaled RTs).
        min_rt_bound (float): The minimum RT value used for min-max scaling.
        max_rt_bound (float): The maximum RT value used for min-max scaling.

    Returns:
        dict: A dictionary containing MAE, MSE, RMSE, R-squared,
              and statistical measures of residuals on the original scale.
    """
    logger.info("Analyzing model predictions")

    # Make predictions
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Convert back to original scale for interpretation
    y_test_orig = inverse_min_max_scale(y_test, min_rt_bound, max_rt_bound)
    y_pred_orig = inverse_min_max_scale(y_pred, min_rt_bound, max_rt_bound)

    # Calculate metrics in original scale
    residuals = y_test_orig - y_pred_orig
    absolute_errors = np.abs(residuals)

    mae_orig = np.mean(absolute_errors)
    mse_orig = np.mean(residuals**2)
    rmse_orig = np.sqrt(mse_orig)
    # Calculate R-squared. Using a small epsilon to avoid division by zero for constant y_test_orig
    ss_total = np.sum((y_test_orig - np.mean(y_test_orig))**2)
    if ss_total > 1e-6: # Check if there's variance in actuals
        r2_orig = 1 - (np.sum(residuals**2) / ss_total)
    else:
        r2_orig = np.nan # R2 is undefined if actuals are constant

    # Calculate additional statistics for residuals and absolute errors
    mean_residuals = np.mean(residuals)
    median_residuals = np.median(residuals)
    std_residuals = np.std(residuals)

    mean_abs_error = np.mean(absolute_errors)
    median_abs_error = np.median(absolute_errors)
    std_abs_error = np.std(absolute_errors)


    logger.info("Prediction analysis (original scale):")
    logger.info(f"  MAE: {mae_orig:.4f} minutes")
    logger.info(f"  MSE: {mse_orig:.4f}")
    logger.info(f"  RMSE: {rmse_orig:.4f} minutes")
    logger.info(f"  R²: {r2_orig:.4f}")
    logger.info(f"  Residuals - Mean: {mean_residuals:.4f}, Median: {median_residuals:.4f}, Std: {std_residuals:.4f} minutes")
    logger.info(f"  Absolute Errors - Mean: {mean_abs_error:.4f}, Median: {median_abs_error:.4f}, Std: {std_abs_error:.4f} minutes")


    # Create diagnostic plots
    plots_dir = f"plots/training_{TIMESTAMP}"
    os.makedirs(plots_dir, exist_ok=True)
    logger.info(f"Saving diagnostic plots to: {plots_dir}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Prediction Diagnostics', fontsize=18)

    # Prediction vs Actual scatter plot
    axes[0, 0].scatter(y_test_orig, y_pred_orig, alpha=0.6, s=20)
    # Plot perfect prediction line
    min_val_plot = min(y_test_orig.min(), y_pred_orig.min())
    max_val_plot = max(y_test_orig.max(), y_pred_orig.max())
    axes[0, 0].plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Retention Time (min)')
    axes[0, 0].set_ylabel('Predicted Retention Time (min)')
    axes[0, 0].set_title(f'Predictions vs Actual (R² = {r2_orig:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Residuals plot
    axes[0, 1].scatter(y_pred_orig, residuals, alpha=0.6, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Residuals')
    axes[0, 1].set_xlabel('Predicted Retention Time (min)')
    axes[0, 1].set_ylabel('Residuals (min)')
    axes[0, 1].set_title(f'Residual Plot (RMSE = {rmse_orig:.3f} min)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Residuals histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Residuals')
    axes[1, 0].set_xlabel('Residuals (min)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Error distribution by retention time range
    # Ensure there's enough data for sensible binning
    if len(y_test_orig) > 50: # Arbitrary threshold for meaningful binning
        rt_bins = np.linspace(y_test_orig.min(), y_test_orig.max(), 10)
        bin_centers = (rt_bins[:-1] + rt_bins[1:]) / 2
        bin_errors = []

        for i in range(len(rt_bins)-1):
            mask = (y_test_orig >= rt_bins[i]) & (y_test_orig < rt_bins[i+1])
            if np.sum(mask) > 0:
                bin_errors.append(np.mean(np.abs(residuals[mask])))
            else:
                bin_errors.append(0)

        axes[1, 1].bar(bin_centers, bin_errors, width=(rt_bins[1]-rt_bins[0])*0.8, alpha=0.7)
        axes[1, 1].set_xlabel('Retention Time Range (min)')
        axes[1, 1].set_ylabel('Mean Absolute Error (min)')
        axes[1, 1].set_title('Error Distribution by RT Range')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].set_title('Error Distribution by RT Range (Insufficient Data)')
        axes[1, 1].text(0.5, 0.5, 'Not enough data for meaningful binning.',
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    diag_plot_path = os.path.join(plots_dir, f'prediction_diagnostics_{TIMESTAMP}.png')
    plt.savefig(diag_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Diagnostic plots saved to: {diag_plot_path}")
    plt.show()

    # Save prediction results
    results_df = pd.DataFrame({
        'actual_rt_scaled': y_test,
        'predicted_rt_scaled': y_pred,
        'actual_rt_original': y_test_orig,
        'predicted_rt_original': y_pred_orig,
        'residuals_original': residuals,
        'absolute_error_original': np.abs(residuals)
    })

    results_path = os.path.join(f"logs/training_{TIMESTAMP}", f'prediction_results_{TIMESTAMP}.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Prediction results saved to: {results_path}")

    return {
        'mae_orig': mae_orig,
        'mse_orig': mse_orig,
        'rmse_orig': rmse_orig,
        'r2_orig': r2_orig,
        'mean_residuals': mean_residuals,
        'median_residuals': median_residuals,
        'std_residuals': std_residuals,
        'mean_abs_error': mean_abs_error,
        'median_abs_error': median_abs_error,
        'std_abs_error': std_abs_error
    }


"""## Main Training Function"""

def train_model():
    """
    Main function to load data, build, train, and evaluate the peptide
    retention time prediction model. It also handles saving comprehensive
    metadata and the final model.
    """
    logger.info("Starting model training process.")

    # Load and preprocess data
    data = load_and_preprocess_data()
    x = np.vstack(data.Peptide.apply(peptide_to_indices))
    y = np.array(data.aligned_rt)

    # Split data into training, validation, and test sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, random_state=42)

    logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)} samples.")

    # Build model
    model = build_model()

    # Configure optimizer with learning rate scheduling
    initial_learning_rate = 1e-3
    lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=initial_learning_rate,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )

    # Compile model with Huber loss and common regression metrics
    model.compile(
        optimizer=optimizer,
        loss='huber',  # Huber loss is robust to outliers
        metrics=['mae', 'mse']
    )

    logger.info("Model summary:")
    model.summary(print_fn=logger.info) # Print summary using the logger

    # Train model
    logger.info(f"Starting model training for {EPOCH} epochs with BATCH_SIZE={BATCH_SIZE}.")
    start_time = datetime.now()
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        validation_data=(X_val, y_val),
        callbacks=create_callbacks(),
        verbose=1 # Show progress bar during training
    )
    end_time = datetime.now()
    training_duration = end_time - start_time
    logger.info(f"Training completed. Total training time: {training_duration}")

    # Evaluate on test set
    logger.info("Evaluating model on test set.")
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    test_rmse = np.sqrt(test_mse)

    test_results = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'test_mse': float(test_mse),
        'test_rmse': float(test_rmse)
    }

    logger.info("="*50)
    logger.info("TEST SET RESULTS:")
    logger.info(f"Test Loss (Huber): {test_loss:.6f}")
    logger.info(f"Test MAE: {test_mae:.6f}")
    logger.info(f"Test MSE: {test_mse:.6f}")
    logger.info(f"Test RMSE: {test_rmse:.6f}")
    logger.info("="*50)

    # Analyze predictions in original scale
    original_scale_metrics = analyze_predictions(model, X_test, y_test, min_rt, max_rt)
    # Merge original scale metrics into test_results for comprehensive metadata saving
    test_results.update({f'original_scale_{k}': v for k, v in original_scale_metrics.items()})

    # Prepare data info for metadata saving
    data_info = {
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'peptide_length_range': [int(data.Peptide.apply(len).min()), int(data.Peptide.apply(len).max())],
        'rt_range_original_data': [float(data.rt.min()), float(data.rt.max())],
        'rt_range_scaled_bounds_used': [float(min_rt), float(max_rt)], # Explicitly state the bounds used for scaling
        'rt_range_normalized_data': [float(data.aligned_rt.min()), float(data.aligned_rt.max())],
        'training_duration_seconds': int(training_duration.total_seconds()),
        'epochs_completed': len(history.history['loss'])
    }
    # Pass scaling info to metadata function
    rt_scaling_info = {'min_rt_bound': float(min_rt), 'max_rt_bound': float(max_rt)}

    # Save comprehensive metadata
    save_training_metadata(model, history, rt_scaling_info, data_info, test_results)

    # Save final model
    model_path = f"models/final_model_{TIMESTAMP}.keras"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    logger.info(f"Final model saved to: {model_path}")

    return model, history, (X_test, y_test)

"""## Plotting Training History"""
def plot_training_history(history):
    """
    Plots the training and validation loss, MAE, MSE, and learning rate
    over epochs from the training history. The plot is also saved to disk.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training History', fontsize=16)

    # Loss plot
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # MAE plot
    axes[0, 1].plot(history.history['mae'], label='Training MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # MSE plot
    axes[1, 0].plot(history.history['mse'], label='Training MSE')
    axes[1, 0].plot(history.history['val_mse'], label='Validation MSE')
    axes[1, 0].set_title('Mean Squared Error')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Learning Rate plot
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log') # Log scale is often useful for LR plots
        axes[1, 1].grid(True)
    else:
        axes[1,1].set_visible(False) # Hide if LR is not available in history

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save the training history plot
    plots_dir = f"plots/training_{TIMESTAMP}"
    os.makedirs(plots_dir, exist_ok=True)
    history_plot_path = os.path.join(plots_dir, f'training_history_plot_{TIMESTAMP}.png')
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training history plot saved to: {history_plot_path}")

    plt.show()


# Main execution block
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Train a Transformer model for peptide retention time prediction.")

    # Add arguments for each parameter with default values
    parser.add_argument('--max_peptide_length', type=int, default=52,
                        help='Maximum length of the peptide sequences.')
    parser.add_argument('--embed_dim', type=int, default=52,
                        help='Embedding size for each token.')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads.')
    parser.add_argument('--ff_dim', type=int, default=32,
                        help='Hidden layer size in feed forward network inside transformer.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training.')
    parser.add_argument('--epoch', type=int, default=5,
                        help='Number of training epochs.')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate for regularization.')
    parser.add_argument('--num_transformer_blocks', type=int, default=2,
                        help='Number of stacked Transformer blocks in the model.')

    # Parse the arguments
    args = parser.parse_args()

    # Assign parsed arguments to global variables
    MAX_PEPTIDE_LENGTH = args.max_peptide_length
    EMBED_DIM = args.embed_dim
    NUM_HEADS = args.num_heads
    FF_DIM = args.ff_dim
    BATCH_SIZE = args.batch_size
    EPOCH = args.epoch
    DROPOUT_RATE = args.dropout_rate
    NUM_TRANSFORMER_BLOCKS = args.num_transformer_blocks

    logger.info(f"Using parameters from command line or defaults:")
    logger.info(f"  MAX_PEPTIDE_LENGTH: {MAX_PEPTIDE_LENGTH}")
    logger.info(f"  EMBED_DIM: {EMBED_DIM}")
    logger.info(f"  NUM_HEADS: {NUM_HEADS}")
    logger.info(f"  FF_DIM: {FF_DIM}")
    logger.info(f"  BATCH_SIZE: {BATCH_SIZE}")
    logger.info(f"  EPOCH: {EPOCH}")
    logger.info(f"  DROPOUT_RATE: {DROPOUT_RATE}")
    logger.info(f"  NUM_TRANSFORMER_BLOCKS: {NUM_TRANSFORMER_BLOCKS}")


    model, history, test_data = train_model()
    plot_training_history(history)


