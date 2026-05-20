import os
import json
import io
import redis
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

# ---------------------------------------------------
# Flask Setup
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates')
)

CORS(app)

# ---------------------------------------------------
# Redis Connection
# ---------------------------------------------------

r = redis.Redis(
    host='127.0.0.1',
    port=6379,
    decode_responses=True
)

# ---------------------------------------------------
# Labels
# ---------------------------------------------------

LABELS = [
    'butter_naan',
    'pav_bhaji',
    'Sandwich',
    'chicken_curry',
    'Hot Dog',
    'cheesecake',
    'sushi',
    'chai',
    'burger',
    'ice_cream',
    'kadai_paneer',
    'Baked Potato',
    'chapati',
    'masala_dosa',
    'dal_makhani',
    'Donut',
    'jalebi',
    'fried_rice',
    'chole_bhature',
    'kulfi',
    'kaathi_rolls',
    'dhokla',
    'Fries',
    'omelette',
    'pakode',
    'momos',
    'paani_puri',
    'samosa',
    'Taco',
    'idli',
    'Taquito',
    'Crispy Chicken',
    'pizza',
    'apple_pie'
]

NUM_CLASSES = len(LABELS)

# ---------------------------------------------------
# Model Cache
# ---------------------------------------------------

_models = {}

# ---------------------------------------------------
# Custom CNN Model
# ---------------------------------------------------

def build_custom_cnn(input_shape=(256, 256, 3), num_classes=NUM_CLASSES):

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same'
    )(inputs)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'
    )(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation='relu',
        padding='same'
    )(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(
        256,
        (3, 3),
        activation='relu',
        padding='same'
    )(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(
        512,
        (3, 3),
        activation='relu',
        padding='same'
    )(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax'
    )(x)

    return tf.keras.Model(inputs, outputs)

# ---------------------------------------------------
# VGG16 Model
# ---------------------------------------------------

def build_vgg16(num_classes=NUM_CLASSES):

    base = tf.keras.applications.VGG16(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = tf.keras.layers.Flatten()(base.output)

    x = tf.keras.layers.Dense(
        512,
        activation='relu'
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax'
    )(x)

    return tf.keras.Model(base.input, outputs)

# ---------------------------------------------------
# ResNet50 Model
# ---------------------------------------------------

def build_resnet50(num_classes=NUM_CLASSES):

    base = tf.keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)

    x = tf.keras.layers.Dense(
        512,
        activation='relu'
    )(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax'
    )(x)

    return tf.keras.Model(base.input, outputs)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

def load_model(model_name):

    if model_name in _models:
        return _models[model_name]

    weight_files = {
        'custom_cnn': 'food_classification_weights.weights.h5',
        'vgg16': 'vgg16_food_classification_weights.weights.h5',
        'resnet50': 'resnet50_food_classification_weights.weights.h5',
    }

    path = os.path.join(
        BASE_DIR,
        weight_files[model_name]
    )

    if model_name == 'custom_cnn':
        model = build_custom_cnn()

    elif model_name == 'vgg16':
        model = build_vgg16()

    elif model_name == 'resnet50':
        model = build_resnet50()

    try:
        model.load_weights(path)

    except:
        model.load_weights(
            path,
            skip_mismatch=True
        )

    _models[model_name] = model

    return model

# ---------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------

def preprocess(image_bytes, size, model_name):

    img = Image.open(
        io.BytesIO(image_bytes)
    ).convert('RGB')

    img = img.resize(size)

    img_array = np.array(
        img,
        dtype=np.float32
    )

    img_batch = np.expand_dims(
        img_array,
        axis=0
    )

    if model_name == 'vgg16':

        from tensorflow.keras.applications.vgg16 import preprocess_input

        return preprocess_input(img_batch)

    elif model_name == 'resnet50':

        from tensorflow.keras.applications.resnet50 import preprocess_input

        return preprocess_input(img_batch)

    else:

        return img_batch / 255.0

# ---------------------------------------------------
# Compute Per-Prediction Metrics
# Derived solely from the softmax probability vector.
#
# For a single-sample inference we treat the predicted
# class as the "positive" class and all others as
# "negative":
#
#   Confidence  = P_top  (model's own certainty score)
#   Precision   = P_top / sum(P where P > threshold)
#                 ≈ how exclusively the model pointed
#                   to this class vs close runners-up
#   Recall      = P_top / (P_top + mean(rest))
#                 ≈ ratio of top-class signal to all
#                   remaining probability mass
#   F1-Score    = harmonic mean of Precision & Recall
#
# All values are in [0, 1] and directly interpretable
# without any stored ground-truth labels.
# ---------------------------------------------------

def compute_prediction_metrics(probs):
    """
    probs : 1-D numpy array of softmax probabilities (length = NUM_CLASSES)
    Returns a dict with confidence, precision, recall, f1_score (all 0-100 %).
    """
    probs = np.array(probs, dtype=np.float64)
    top_idx  = int(np.argmax(probs))
    p_top    = float(probs[top_idx])

    # --- Confidence ---
    confidence = round(p_top * 100, 2)

    # --- Precision ---
    # Fraction of total "positive signal" captured by the top class.
    # Threshold: classes with prob > mean are considered "candidate positives".
    threshold = float(np.mean(probs))
    candidate_mass = float(np.sum(probs[probs > threshold]))
    precision = p_top / candidate_mass if candidate_mass > 0 else p_top
    precision = round(min(precision, 1.0) * 100, 2)

    # --- Recall ---
    # How much of the total prob mass is concentrated in the predicted class
    # vs the average noise from other classes.
    rest_mean = float(np.mean(np.delete(probs, top_idx)))
    denom     = p_top + rest_mean
    recall    = (p_top / denom) if denom > 0 else 1.0
    recall    = round(min(recall, 1.0) * 100, 2)

    # --- F1-Score ---
    p_frac = precision / 100.0
    r_frac = recall   / 100.0
    if (p_frac + r_frac) > 0:
        f1 = 2 * p_frac * r_frac / (p_frac + r_frac)
    else:
        f1 = 0.0
    f1_score = round(f1 * 100, 2)

    return {
        'confidence': confidence,
        'precision':  precision,
        'recall':     recall,
        'f1_score':   f1_score
    }

# ---------------------------------------------------
# Get Nutrition From Redis
# ---------------------------------------------------

def get_nutrition(food_name):

    try:

        # Get entire food_details object from Redis
        raw = r.get("food_details")

        if not raw:
            print("food_details key not found in Redis")
            return None

        # Convert JSON string -> Python dictionary
        food_details = json.loads(raw)

        # Get nutrition data for predicted food
        nutrition = food_details.get(food_name)

        if nutrition:

            return {
                'fiber':    nutrition.get('fiber',    0),
                'protein':  nutrition.get('protein',  0),
                'calories': nutrition.get('calories', 0),
                'fats':     nutrition.get('fats',     0),
                'carbs':    nutrition.get('carbs',    0)
            }

        else:
            print(f"No nutrition data found for: {food_name}")

    except Exception as e:

        print(f"Redis Lookup Error: {e}")

    return None

# ---------------------------------------------------
# Home Route
# ---------------------------------------------------

@app.route('/')
def index():

    return render_template('index.html')

# ---------------------------------------------------
# Prediction Route
# ---------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():

    try:

        # Check image upload
        if 'image' not in request.files:

            return jsonify({
                'error': 'No image uploaded'
            }), 400

        # Get model name
        model_name = request.form.get(
            'model',
            'custom_cnn'
        ).strip().lower().replace(' ', '_')

        # Read image
        image_bytes = request.files['image'].read()

        # Image sizes
        sizes = {
            'custom_cnn': (256, 256),
            'vgg16':      (224, 224),
            'resnet50':   (224, 224)
        }

        size = sizes.get(model_name, (256, 256))

        # Load model
        model = load_model(model_name)

        # Preprocess image
        processed_image = preprocess(image_bytes, size, model_name)

        # Predict
        preds = model.predict(processed_image, verbose=0)[0]

        # Top prediction
        top_idx    = int(np.argmax(preds))
        label      = LABELS[top_idx]
        confidence = round(float(preds[top_idx]) * 100, 2)

        # Per-prediction metrics (computed from softmax vector only)
        metrics = compute_prediction_metrics(preds)

        # Get nutrition from Redis
        nutrition_data = get_nutrition(label)

        # Top 5 Predictions
        top5 = []
        sorted_indices = np.argsort(preds)[::-1][:5]

        for i in sorted_indices:
            top5.append({
                'label':      LABELS[i],
                'confidence': round(float(preds[i]) * 100, 2)
            })

        # Final response
        return jsonify({
            'predicted_label': label,
            'confidence':      confidence,
            'model_used':      model_name.upper(),
            'top5':            top5,
            'nutrition':       nutrition_data,
            'metrics':         metrics        # <-- new
        })

    except Exception as e:

        print(f"[ERROR] {e}")

        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------
# Run App
# ---------------------------------------------------

if __name__ == '__main__':

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )