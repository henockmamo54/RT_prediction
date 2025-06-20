# requirements.txt
# Flask==2.3.2
# numpy==1.24.4
# pandas==2.0.3
# tensorflow==2.13.0 # or your specific TensorFlow version
# scikit-learn==1.3.0 # NEW: Added for PolynomialFeatures and LinearRegression

# app.py
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Configure basic logging for the Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Global Variables for Model and Preprocessing Data ---
# These will be loaded once when the Flask app starts
MODEL = None
AA_TO_INDEX = None
MAX_PEPTIDE_LENGTH = None
MIN_RT_BOUND = None
MAX_RT_BOUND = None
VOCAB_SIZE = None

# --- Custom Keras Layers (MUST be defined for model loading) ---
@tf.keras.utils.register_keras_serializable(package='TransformerBlock')
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.attention(inputs, inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
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
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
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
        config = super().get_config()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        })
        return config

# --- Preprocessing and Inverse Scaling Functions ---
def peptide_to_indices_local(sequence: str) -> np.ndarray:
    """
    Converts a peptide sequence string to a numpy array of integer indices,
    using the global AA_TO_INDEX mapping. Pads the sequence with <PAD>
    if it is shorter than MAX_PEPTIDE_LENGTH. Unknown amino acids are
    mapped to '<UNK>'.
    """
    if AA_TO_INDEX is None or MAX_PEPTIDE_LENGTH is None:
        raise RuntimeError("Model parameters not loaded. Please ensure Flask app initializes correctly.")

    indices = [AA_TO_INDEX.get(aa, AA_TO_INDEX['<UNK>']) for aa in sequence]

    if len(indices) > MAX_PEPTIDE_LENGTH:
        indices = indices[:MAX_PEPTIDE_LENGTH]

    padding_length = MAX_PEPTIDE_LENGTH - len(indices)
    indices.extend([AA_TO_INDEX['<PAD>']] * padding_length)
    return np.array(indices, dtype=np.int32)

def inverse_min_max_scale(scaled_x: float, min_val: float, max_val: float) -> float:
    """
    Inverses the min-max scaling to convert a scaled value back to its original range.
    """
    return scaled_x * (max_val - min_val) + min_val

# --- Model Loading ---
def load_model_and_metadata():
    """
    Loads the Keras model and associated metadata (scaling params, vocab)
    into global variables. This function runs once on app startup.
    """
    global MODEL, AA_TO_INDEX, MAX_PEPTIDE_LENGTH, MIN_RT_BOUND, MAX_RT_BOUND, VOCAB_SIZE

    # Find the latest training log directory
    log_dirs = [d for d in os.listdir('logs') if os.path.isdir(os.path.join('logs', d)) and d.startswith('training_')]
    log_dirs.sort(reverse=True) # Sort to get the most recent first

    if not log_dirs:
        logger.error("No training log directories found in 'logs/'. Please train a model first.")
        raise FileNotFoundError("No trained model found.")

    latest_log_dir = os.path.join('logs', log_dirs[0])
    latest_timestamp = log_dirs[0].replace('training_', '') # Extract timestamp for model path

    # Load training configuration
    config_path = os.path.join(latest_log_dir, 'training_config.json')
    if not os.path.exists(config_path):
        logger.error(f"Training config file not found: {config_path}")
        raise FileNotFoundError(f"Training config file not found: {config_path}")

    with open(config_path, 'r') as f:
        training_config = json.load(f)

    AA_TO_INDEX = {aa: i for i, aa in enumerate(training_config['vocab'])}
    MAX_PEPTIDE_LENGTH = training_config['max_peptide_length']
    VOCAB_SIZE = training_config['vocab_size']

    # Load RT scaling parameters
    rt_scaling_path = os.path.join(latest_log_dir, 'rt_scaling_parameters.json')
    if not os.path.exists(rt_scaling_path):
        logger.error(f"RT scaling parameters file not found: {rt_scaling_path}")
        raise FileNotFoundError(f"RT scaling parameters file not found: {rt_scaling_path}")

    with open(rt_scaling_path, 'r') as f:
        rt_scaling_params = json.load(f)

    MIN_RT_BOUND = rt_scaling_params['min_rt_bound']
    MAX_RT_BOUND = rt_scaling_params['max_rt_bound']

    # Construct model path
    model_path = os.path.join('models', f'final_model_{latest_timestamp}.keras')
    if not os.path.exists(model_path):
        # Fallback to 'best_model.keras' if final_model_{TIMESTAMP}.keras not found
        model_path = os.path.join('models', f'model_{latest_timestamp}', 'best_model.keras')
        if not os.path.exists(model_path):
            logger.error(f"Best model not found at expected path: {model_path}")
            raise FileNotFoundError(f"Trained Keras model not found at: {model_path}")

    logger.info(f"Attempting to load model from: {model_path}")
    # Load the Keras model, providing custom objects
    try:
        MODEL = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'TransformerBlock': TransformerBlock,
                'TokenAndPositionEmbedding': TokenAndPositionEmbedding
            },
            compile=False # No need to recompile for inference if already compiled during training
        )
        MODEL.compile(loss='huber', metrics=['mae', 'mse']) # Recompile for potential metric access, but not strictly needed for raw prediction
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load Keras model: {e}")

# Load model and metadata when the Flask app starts
with app.app_context():
    try:
        load_model_and_metadata()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Application startup failed: {e}")
        # In a production environment, you might want to exit or disable routes
        # For this example, we'll let it continue but predictions will fail.

# --- API Endpoint for Single Prediction ---
@app.route('/predict', methods=['POST'])
def predict_rt():
    """
    API endpoint to predict retention time for a given peptide sequence.
    Expects a JSON payload with a 'peptide' key.
    Example:
    curl -X POST -H "Content-Type: application/json" \
         -d '{"peptide": "SAMPLEPEPTIDE"}' \
         http://127.0.0.1:5000/predict
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Server might be initializing or failed to load model."}), 500

    data = request.get_json()
    if not data or 'peptide' not in data:
        return jsonify({"error": "Invalid request. Please provide a JSON payload with a 'peptide' key."}), 400

    peptide_sequence = data['peptide']
    if not isinstance(peptide_sequence, str) or not peptide_sequence.strip():
        return jsonify({"error": "Invalid peptide sequence. Must be a non-empty string."}), 400

    try:
        # Preprocess the peptide sequence
        peptide_indices = peptide_to_indices_local(peptide_sequence.strip())

        # The model expects a batch of inputs, so reshape to (1, MAX_PEPTIDE_LENGTH)
        peptide_input_batch = np.expand_dims(peptide_indices, axis=0)

        # Make prediction
        scaled_prediction = MODEL.predict(peptide_input_batch, verbose=0).flatten()[0]

        # Inverse scale the prediction
        predicted_rt_original = inverse_min_max_scale(scaled_prediction, MIN_RT_BOUND, MAX_RT_BOUND)

        response = {
            "peptide_sequence": peptide_sequence.strip(),
            "predicted_retention_time_minutes": float(predicted_rt_original)
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during prediction for peptide '{peptide_sequence}': {e}")
        return jsonify({"error": f"Prediction failed due to an internal error: {str(e)}"}), 500

# --- API Endpoint for Batch Prediction ---
@app.route('/predict_batch', methods=['POST'])
def predict_batch_rt():
    """
    API endpoint to predict retention times for a list of peptide sequences.
    Expects a JSON payload with a 'peptides' key containing a list of strings.
    Example:
    curl -X POST -H "Content-Type: application/json" \
         -d '{"peptides": ["PEPTIDE1", "PEPTIDE2", "SHORT", "LONGERPEPTIDE"]}' \
         http://127.0.0.1:5000/predict_batch
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Server might be initializing or failed to load model."}), 500

    data = request.get_json()
    if not data or 'peptides' not in data:
        return jsonify({"error": "Invalid request. Please provide a JSON payload with a 'peptides' key."}), 400

    peptide_sequences = data['peptides']
    if not isinstance(peptide_sequences, list) or not all(isinstance(p, str) for p in peptide_sequences):
        return jsonify({"error": "Invalid 'peptides' input. Must be a list of strings."}), 400
    if not peptide_sequences:
        return jsonify({"error": "The 'peptides' list cannot be empty."}), 400

    predictions = []
    peptide_inputs_for_model = []
    original_sequences = []

    for pep_seq in peptide_sequences:
        if not pep_seq.strip():
            # Handle empty/whitespace strings in the list
            predictions.append({
                "peptide_sequence": pep_seq,
                "predicted_retention_time_minutes": None,
                "error": "Empty or whitespace peptide sequence"
            })
            continue

        try:
            peptide_indices = peptide_to_indices_local(pep_seq.strip())
            peptide_inputs_for_model.append(peptide_indices)
            original_sequences.append(pep_seq.strip())
        except Exception as e:
            # If preprocessing fails for a specific peptide, log and add error
            logger.error(f"Error preprocessing peptide '{pep_seq}': {e}")
            predictions.append({
                "peptide_sequence": pep_seq,
                "predicted_retention_time_minutes": None,
                "error": f"Preprocessing failed: {str(e)}"
            })

    if not peptide_inputs_for_model:
        return jsonify({"error": "No valid peptides provided for prediction after preprocessing."}), 400

    try:
        # Predict all valid peptides in one batch
        batch_predictions_scaled = MODEL.predict(np.array(peptide_inputs_for_model), verbose=0).flatten()

        # Inverse scale all predictions
        batch_predictions_original = inverse_min_max_scale(batch_predictions_scaled, MIN_RT_BOUND, MAX_RT_BOUND)

        # Map predictions back to original sequences
        idx = 0
        for i, pep_seq in enumerate(peptide_sequences):
            if not pep_seq.strip(): # Skip already handled invalid peptides
                continue
            # Find the corresponding original sequence in the list of valid ones
            # This logic assumes the order of original_sequences aligns with batch_predictions_original.
            # It's important that original_sequences only contains peptides that were successfully preprocessed.
            if idx < len(original_sequences) and original_sequences[idx] == pep_seq.strip():
                predictions.append({
                    "peptide_sequence": original_sequences[idx],
                    "predicted_retention_time_minutes": float(batch_predictions_original[idx])
                })
                idx += 1
            else:
                # Fallback for unexpected mapping issues (should be rare if preprocessing is robust)
                predictions.append({
                    "peptide_sequence": pep_seq,
                    "predicted_retention_time_minutes": None,
                    "error": "Internal mapping error for prediction"
                })


        return jsonify({"predictions": predictions}), 200

    except Exception as e:
        logger.error(f"Error during batch prediction: {e}", exc_info=True)
        return jsonify({"error": f"Batch prediction failed due to an internal error: {str(e)}"}), 500

# --- API Endpoint for Single Aligned Prediction ---
@app.route('/align_predict', methods=['POST'])
def align_predict_rt():
    """
    API endpoint to predict retention time for a target peptide, aligned
    using polynomial regression on reference peptides.

    Expects a JSON payload with:
    - 'target_peptide': A string representing the target peptide sequence.
    - 'reference_peptides': A list of strings for reference peptide sequences.
    - 'reference_rts': A list of floats representing known retention times
                       for the reference peptides, corresponding to
                       'reference_peptides'.

    Example:
    curl -X POST -H "Content-Type: application/json" \
         -d '{
               "target_peptide": "TARGETPEPTIDE",
               "reference_peptides": ["REF1PEPTIDE", "REF2PEPTIDE", "REF3PEPTIDE", "REF4PEPTIDE"],
               "reference_rts": [10.5, 20.3, 30.1, 40.8]
             }' \
         http://127.0.0.1:5000/align_predict
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Server might be initializing or failed to load model."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. JSON payload missing."}), 400

    # Validate input parameters
    target_peptide = data.get('target_peptide')
    reference_peptides = data.get('reference_peptides')
    reference_rts = data.get('reference_rts')

    if not (isinstance(target_peptide, str) and target_peptide.strip()):
        return jsonify({"error": "Missing or invalid 'target_peptide'. Must be a non-empty string."}), 400
    if not (isinstance(reference_peptides, list) and all(isinstance(p, str) for p in reference_peptides)):
        return jsonify({"error": "Missing or invalid 'reference_peptides'. Must be a list of strings."}), 400
    if not (isinstance(reference_rts, list) and all(isinstance(r, (int, float)) for r in reference_rts)):
        return jsonify({"error": "Missing or invalid 'reference_rts'. Must be a list of numbers."}), 400
    if len(reference_peptides) != len(reference_rts):
        return jsonify({"error": "'reference_peptides' and 'reference_rts' must have the same length."}), 400

    # For polynomial regression of degree 3, we need at least 4 data points (degree + 1)
    POLYNOMIAL_DEGREE = 3
    if len(reference_peptides) < (POLYNOMIAL_DEGREE + 1):
        return jsonify({
            "error": f"Insufficient reference points for polynomial regression of degree {POLYNOMIAL_DEGREE}. "
                     f"At least {POLYNOMIAL_DEGREE + 1} reference pairs are required, but got {len(reference_peptides)}."
        }), 400

    try:
        # 1. Predict retention times for reference peptides using the Transformer model
        # Prepare batch input for reference peptides to make a single model.predict call
        ref_peptide_inputs = []
        valid_ref_peptides_for_batch_pred = [] # Store valid peptides for later alignment with their actual RTs
        actual_ref_rts_aligned_with_preds = []

        for i, ref_pep in enumerate(reference_peptides):
            if not ref_pep.strip(): # Skip empty/whitespace peptides
                logger.warning(f"Skipping empty reference peptide in align_predict: '{ref_pep}'")
                continue
            try:
                ref_indices = peptide_to_indices_local(ref_pep.strip())
                ref_peptide_inputs.append(ref_indices)
                valid_ref_peptides_for_batch_pred.append(ref_pep.strip())
                actual_ref_rts_aligned_with_preds.append(reference_rts[i])
            except Exception as e:
                logger.error(f"Error preprocessing reference peptide '{ref_pep}': {e}")
                # Optionally, you could return an error for this specific peptide
                # or just log and skip it, reducing the number of reference points.
                continue

        if not ref_peptide_inputs:
            return jsonify({"error": "No valid reference peptides provided for prediction."}), 400

        # Make one prediction call for all valid reference peptides
        scaled_ref_predictions = MODEL.predict(np.array(ref_peptide_inputs), verbose=0).flatten()

        # Convert to original scale for polynomial regression input
        predicted_ref_rt_orig = np.array([
            inverse_min_max_scale(p, MIN_RT_BOUND, MAX_RT_BOUND) for p in scaled_ref_predictions
        ]).reshape(-1, 1) # Reshape for sklearn

        actual_ref_rt_orig = np.array(actual_ref_rts_aligned_with_preds).reshape(-1, 1)

        # Re-check for sufficient reference points after filtering
        if len(actual_ref_rt_orig) < (POLYNOMIAL_DEGREE + 1):
             return jsonify({
                 "error": f"Insufficient valid reference points after preprocessing for polynomial regression of degree {POLYNOMIAL_DEGREE}. "
                          f"At least {POLYNOMIAL_DEGREE + 1} reference pairs are required, but only {len(actual_ref_rt_orig)} valid pairs found."
             }), 400


        # 2. Perform 3rd-degree polynomial regression
        # Input (X) for polynomial regression: predicted_ref_rt_orig
        # Target (y) for polynomial regression: actual_ref_rt_orig
        polynomial_regression_model = make_pipeline(PolynomialFeatures(degree=POLYNOMIAL_DEGREE), LinearRegression())
        polynomial_regression_model.fit(predicted_ref_rt_orig, actual_ref_rt_orig)

        # 3. Predict retention time for the target peptide using the Transformer model
        target_indices = peptide_to_indices_local(target_peptide.strip())
        target_input_batch = np.expand_dims(target_indices, axis=0)
        scaled_target_pred = MODEL.predict(target_input_batch, verbose=0).flatten()[0]
        predicted_target_rt_orig_unaligned = inverse_min_max_scale(scaled_target_pred, MIN_RT_BOUND, MAX_RT_BOUND)

        # 4. Align the target peptide's retention time using the polynomial regression model
        aligned_target_rt = polynomial_regression_model.predict(
            np.array([[predicted_target_rt_orig_unaligned]])
        ).flatten()[0]

        response = {
            "target_peptide_sequence": target_peptide.strip(),
            "predicted_retention_time_unaligned_minutes": float(predicted_target_rt_orig_unaligned),
            "predicted_retention_time_aligned_minutes": float(aligned_target_rt),
            "reference_predictions_unaligned_minutes": predicted_ref_rt_orig.flatten().tolist(),
            "reference_actual_rts_minutes": actual_ref_rts_aligned_with_preds # Use the filtered list
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during aligned prediction for target peptide '{target_peptide}': {e}", exc_info=True)
        return jsonify({"error": f"Aligned prediction failed due to an internal error: {str(e)}"}), 500

# --- NEW API Endpoint for Batch Aligned Prediction ---
@app.route('/align_predict_batch', methods=['POST'])
def align_predict_batch_rt():
    """
    API endpoint to predict and align retention times for multiple target peptides.

    Expects a JSON payload with:
    - 'target_peptides': A list of strings representing the target peptide sequences.
    - 'reference_peptides': A list of strings for reference peptide sequences.
    - 'reference_rts': A list of floats representing known retention times
                       for the reference peptides, corresponding to
                       'reference_peptides'.

    Example:
    curl -X POST -H "Content-Type: application/json" \
         -d '{
               "target_peptides": ["TARGETPEPTIDE1", "TARGETPEPTIDE2"],
               "reference_peptides": ["REF1PEPTIDE", "REF2PEPTIDE", "REF3PEPTIDE", "REF4PEPTIDE"],
               "reference_rts": [10.5, 20.3, 30.1, 40.8]
             }' \
         http://127.0.0.1:5000/align_predict_batch
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Server might be initializing or failed to load model."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request. JSON payload missing."}), 400

    # Validate input parameters
    target_peptides = data.get('target_peptides')
    reference_peptides = data.get('reference_peptides')
    reference_rts = data.get('reference_rts')

    if not (isinstance(target_peptides, list) and all(isinstance(p, str) for p in target_peptides) and target_peptides):
        return jsonify({"error": "Missing or invalid 'target_peptides'. Must be a non-empty list of strings."}), 400
    if not (isinstance(reference_peptides, list) and all(isinstance(p, str) for p in reference_peptides)):
        return jsonify({"error": "Missing or invalid 'reference_peptides'. Must be a list of strings."}), 400
    if not (isinstance(reference_rts, list) and all(isinstance(r, (int, float)) for r in reference_rts)):
        return jsonify({"error": "Missing or invalid 'reference_rts'. Must be a list of numbers."}), 400
    if len(reference_peptides) != len(reference_rts):
        return jsonify({"error": "'reference_peptides' and 'reference_rts' must have the same length."}), 400

    POLYNOMIAL_DEGREE = 3
    # Check for sufficient reference points for polynomial regression
    if len(reference_peptides) < (POLYNOMIAL_DEGREE + 1):
        return jsonify({
            "error": f"Insufficient reference points for polynomial regression of degree {POLYNOMIAL_DEGREE}. "
                     f"At least {POLYNOMIAL_DEGREE + 1} reference pairs are required, but got {len(reference_peptides)}."
        }), 400

    try:
        # 1. Predict retention times for reference peptides (in batch)
        ref_peptide_inputs = []
        valid_ref_peptides_for_batch_pred = []
        actual_ref_rts_for_alignment = []

        for i, ref_pep in enumerate(reference_peptides):
            if not ref_pep.strip():
                logger.warning(f"Skipping empty reference peptide in align_predict_batch: '{ref_pep}'")
                continue
            try:
                ref_indices = peptide_to_indices_local(ref_pep.strip())
                ref_peptide_inputs.append(ref_indices)
                valid_ref_peptides_for_batch_pred.append(ref_pep.strip())
                actual_ref_rts_for_alignment.append(reference_rts[i])
            except Exception as e:
                logger.error(f"Error preprocessing reference peptide '{ref_pep}' for alignment: {e}")
                continue # Skip this reference peptide but continue with others

        if not ref_peptide_inputs:
            return jsonify({"error": "No valid reference peptides provided for alignment model training."}), 400

        scaled_ref_predictions = MODEL.predict(np.array(ref_peptide_inputs), verbose=0).flatten()
        predicted_ref_rt_orig = np.array([
            inverse_min_max_scale(p, MIN_RT_BOUND, MAX_RT_BOUND) for p in scaled_ref_predictions
        ]).reshape(-1, 1)

        actual_ref_rt_orig = np.array(actual_ref_rts_for_alignment).reshape(-1, 1)

        if len(actual_ref_rt_orig) < (POLYNOMIAL_DEGREE + 1):
             return jsonify({
                 "error": f"Insufficient valid reference points after preprocessing for polynomial regression of degree {POLYNOMIAL_DEGREE}. "
                          f"At least {POLYNOMIAL_DEGREE + 1} reference pairs are required, but only {len(actual_ref_rt_orig)} valid pairs found."
             }), 400

        # 2. Perform 3rd-degree polynomial regression
        polynomial_regression_model = make_pipeline(PolynomialFeatures(degree=POLYNOMIAL_DEGREE), LinearRegression())
        polynomial_regression_model.fit(predicted_ref_rt_orig, actual_ref_rt_orig)

        # 3. Predict retention times for target peptides (in batch)
        target_peptide_inputs = []
        original_target_sequences = []
        for target_pep in target_peptides:
            if not target_pep.strip():
                logger.warning(f"Skipping empty target peptide in align_predict_batch: '{target_pep}'")
                continue
            try:
                target_indices = peptide_to_indices_local(target_pep.strip())
                target_peptide_inputs.append(target_indices)
                original_target_sequences.append(target_pep.strip())
            except Exception as e:
                logger.error(f"Error preprocessing target peptide '{target_pep}' for prediction: {e}")
                # For batch, we'll handle this in the final results list
                continue

        if not target_peptide_inputs:
            return jsonify({"error": "No valid target peptides provided for prediction."}), 400

        scaled_target_predictions = MODEL.predict(np.array(target_peptide_inputs), verbose=0).flatten()
        predicted_target_rt_orig_unaligned = np.array([
            inverse_min_max_scale(p, MIN_RT_BOUND, MAX_RT_BOUND) for p in scaled_target_predictions
        ])

        # 4. Align the target peptides' retention times (in batch)
        # Reshape to (num_samples, 1) for prediction with sklearn pipeline
        aligned_target_rt_batch = polynomial_regression_model.predict(
            predicted_target_rt_orig_unaligned.reshape(-1, 1)
        ).flatten()

        results = []
        valid_pred_idx = 0
        for i, pep_seq in enumerate(target_peptides):
            if not pep_seq.strip():
                results.append({
                    "peptide_sequence": pep_seq,
                    "predicted_retention_time_unaligned_minutes": None,
                    "predicted_retention_time_aligned_minutes": None,
                    "error": "Empty or whitespace peptide sequence"
                })
                continue
            if valid_pred_idx < len(original_target_sequences) and original_target_sequences[valid_pred_idx] == pep_seq.strip():
                results.append({
                    "peptide_sequence": original_target_sequences[valid_pred_idx],
                    "predicted_retention_time_unaligned_minutes": float(predicted_target_rt_orig_unaligned[valid_pred_idx]),
                    "predicted_retention_time_aligned_minutes": float(aligned_target_rt_batch[valid_pred_idx])
                })
                valid_pred_idx += 1
            else:
                results.append({
                    "peptide_sequence": pep_seq,
                    "predicted_retention_time_unaligned_minutes": None,
                    "predicted_retention_time_aligned_minutes": None,
                    "error": "Prediction skipped due to preprocessing error or internal mapping issue"
                })


        response = {
            "target_predictions": results,
            "reference_predictions_unaligned_minutes": predicted_ref_rt_orig.flatten().tolist(),
            "reference_actual_rts_minutes": actual_ref_rts_for_alignment
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during batch aligned prediction: {e}", exc_info=True)
        return jsonify({"error": f"Batch aligned prediction failed due to an internal error: {str(e)}"}), 500

# --- Health Check Endpoint (Optional but Recommended) ---
@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    status = "ready" if MODEL is not None else "initializing"
    return jsonify({"status": status, "model_loaded": MODEL is not None}), 200

if __name__ == '__main__':
    logger.info("Starting Flask application. Model loading in progress...")
    # The load_model_and_metadata() call is outside __name__ == '__main__' to ensure it runs
    # when Flask's development server reloads or in a production WSGI environment.
    # We added the `with app.app_context()` block to ensure it runs properly.
    app.run(debug=True, host='0.0.0.0', port=5000)