import os
import json
import warnings

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings("ignore")

# ============================================================
# CODEALPHA - TASK 2
# SPEECH EMOTION RECOGNITION
# MFCC + DELTA + DELTA-DELTA + CNN
# ============================================================

DATA_DIR = "data/Audio_Speech_Actors_01-24_16k"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "speech_emotion_cnn.keras"
)

CLASS_PATH = os.path.join(
    MODEL_DIR,
    "emotion_classes.json"
)

RANDOM_STATE = 42

SAMPLE_RATE = 16000

N_MFCC = 40

MAX_FRAMES = 174

EPOCHS = 60

BATCH_SIZE = 32


os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# EMOTION MAPPING
# ============================================================

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Normalize audio amplitude
    audio = librosa.util.normalize(audio)

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=256
    )

    # Delta
    delta = librosa.feature.delta(
        mfcc
    )

    # Delta-delta
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Combine as 3 CNN channels
    features = np.stack(
        [
            mfcc,
            delta,
            delta2
        ],
        axis=-1
    )

    # Fixed number of time frames
    current_frames = features.shape[1]

    if current_frames < MAX_FRAMES:

        padding = (
            MAX_FRAMES -
            current_frames
        )

        features = np.pad(
            features,
            (
                (0, 0),
                (0, padding),
                (0, 0)
            ),
            mode="constant"
        )

    else:

        features = features[
            :,
            :MAX_FRAMES,
            :
        ]

    return features.astype(
        np.float32
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    X = []
    y = []
    actors = []

    print("\nScanning RAVDESS dataset...")

    audio_files = []

    for root, _, filenames in os.walk(
        DATA_DIR
    ):

        for filename in filenames:

            if filename.lower().endswith(".wav"):

                audio_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    audio_files.sort()

    print(
        f"Audio files found: {len(audio_files)}"
    )

    if len(audio_files) == 0:

        raise FileNotFoundError(
            "\nNo .wav files found.\n"
            "Expected dataset location:\n"
            f"{DATA_DIR}"
        )

    for index, file_path in enumerate(
        audio_files
    ):

        filename = os.path.basename(
            file_path
        )

        parts = filename.split("-")

        if len(parts) != 7:
            continue

        emotion_code = parts[2]

        actor_id = int(
            parts[6].split(".")[0]
        )

        if emotion_code not in EMOTION_MAP:
            continue

        emotion = EMOTION_MAP[
            emotion_code
        ]

        try:

            features = extract_features(
                file_path
            )

            X.append(features)

            y.append(emotion)

            actors.append(actor_id)

        except Exception as error:

            print(
                f"Error processing "
                f"{filename}: {error}"
            )

        if (index + 1) % 100 == 0:

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(audio_files)}"
            )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(y)

    actors = np.array(
        actors
    )

    return X, y, actors


# ============================================================
# BUILD CNN
# ============================================================

def build_model(
    input_shape,
    number_of_classes
):

    inputs = layers.Input(
        shape=input_shape
    )

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(
        0.20
    )(x)


    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(
        0.25
    )(x)


    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(
        0.30
    )(x)


    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.40
    )(x)

    outputs = layers.Dense(
        number_of_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(y):

    unique_classes, counts = np.unique(
        y,
        return_counts=True
    )

    total = len(y)

    number_of_classes = len(
        unique_classes
    )

    class_weights = {}

    for class_id, count in zip(
        unique_classes,
        counts
    ):

        class_weights[int(class_id)] = (
            total /
            (
                number_of_classes *
                count
            )
        )

    return class_weights


# ============================================================
# TRAINING PLOTS
# ============================================================

def plot_training_history(
    history
):

    history_data = history.history


    # Accuracy
    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        history_data["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history_data["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(
        "CNN Training and Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "training_accuracy.png"
        ),
        dpi=300
    )

    plt.close()


    # Loss
    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        history_data["loss"],
        label="Training Loss"
    )

    plt.plot(
        history_data["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "CNN Training and Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "training_loss.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names
):

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(10, 8)
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        "Speech Emotion Recognition - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Emotion"
    )

    plt.ylabel(
        "Actual Emotion"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# EMOTION DISTRIBUTION
# ============================================================

def plot_class_distribution(
    y
):

    counts = pd.Series(
        y
    ).value_counts()
    
    plt.figure(
        figsize=(10, 5)
    )

    counts.plot(
        kind="bar"
    )

    plt.title(
        "RAVDESS Emotion Distribution"
    )

    plt.xlabel(
        "Emotion"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "emotion_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "       CODEALPHA - TASK 2"
    )

    print(
        "       SPEECH EMOTION RECOGNITION"
    )

    print(
        "       MFCC + DELTA + DELTA-DELTA + CNN"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    X, y, actors = load_dataset()


    print(
        "\nFeature matrix shape:"
    )

    print(
        X.shape
    )


    print(
        "\nEmotion distribution:"
    )

    print(
        pd.Series(y).value_counts()
    )


    plot_class_distribution(
        y
    )


    # --------------------------------------------------------
    # ENCODE CLASSES
    # --------------------------------------------------------

    class_names = sorted(
        np.unique(y)
    )

    class_to_index = {
        emotion: index
        for index, emotion
        in enumerate(class_names)
    }

    y_encoded = np.array([

        class_to_index[
            emotion
        ]

        for emotion in y

    ])


    with open(
        CLASS_PATH,
        "w"
    ) as file:

        json.dump(
            class_names,
            file,
            indent=4
        )


    # --------------------------------------------------------
    # ACTOR-BASED SPLIT
    # --------------------------------------------------------

    train_mask = actors <= 18

    validation_mask = (
        (actors >= 19) &
        (actors <= 21)
    )

    test_mask = actors >= 22


    X_train = X[
        train_mask
    ]

    y_train = y_encoded[
        train_mask
    ]


    X_validation = X[
        validation_mask
    ]

    y_validation = y_encoded[
        validation_mask
    ]


    X_test = X[
        test_mask
    ]

    y_test = y_encoded[
        test_mask
    ]


    print(
        "\nDataset split:"
    )

    print(
        f"Training   : {len(X_train)}"
    )

    print(
        f"Validation : {len(X_validation)}"
    )

    print(
        f"Testing    : {len(X_test)}"
    )


    # --------------------------------------------------------
    # GLOBAL FEATURE NORMALIZATION
    # --------------------------------------------------------
    #
    # Calculate statistics ONLY from training data.
    # This prevents information leakage.
    # --------------------------------------------------------

    mean = np.mean(
        X_train,
        axis=(0, 1, 2),
        keepdims=True
    )

    std = np.std(
        X_train,
        axis=(0, 1, 2),
        keepdims=True
    )

    std = np.maximum(
        std,
        1e-6
    )


    X_train = (
        X_train - mean
    ) / std

    X_validation = (
        X_validation - mean
    ) / std

    X_test = (
        X_test - mean
    ) / std


    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        y_train
    )

    print(
        "\nClass weights:"
    )

    print(
        class_weights
    )


    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print(
        "\nBuilding improved CNN..."
    )


    model = build_model(

        input_shape=X_train.shape[1:],

        number_of_classes=len(
            class_names
        )
    )


    model.summary()


    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    early_stopping = callbacks.EarlyStopping(

        monitor="val_accuracy",

        patience=10,

        mode="max",

        restore_best_weights=True,

        verbose=1
    )


    reduce_lr = callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=4,

        min_lr=0.00001,

        verbose=1
    )


    checkpoint = callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\nTraining improved CNN..."
    )


    history = model.fit(

        X_train,

        y_train,

        validation_data=(
            X_validation,
            y_validation
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        callbacks=[
            early_stopping,
            reduce_lr,
            checkpoint
        ],

        shuffle=True,

        verbose=1
    )


    # --------------------------------------------------------
    # TRAINING PLOTS
    # --------------------------------------------------------

    plot_training_history(
        history
    )


    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print(
        "\nEvaluating test set..."
    )


    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )


    probabilities = model.predict(
        X_test,
        verbose=0
    )


    predictions = np.argmax(
        probabilities,
        axis=1
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-Score : {f1:.4f}"
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(

        y_test,

        predictions,

        target_names=class_names,

        zero_division=0
    )


    print(
        "\nClassification Report:"
    )

    print(
        report
    )


    with open(
        os.path.join(
            OUTPUT_DIR,
            "classification_report.txt"
        ),
        "w"
    ) as file:

        file.write(
            "Speech Emotion Recognition\n\n"
        )

        file.write(
            "MFCC + Delta + Delta-Delta + CNN\n\n"
        )

        file.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        file.write(
            f"Precision: {precision:.4f}\n"
        )

        file.write(
            f"Recall   : {recall:.4f}\n"
        )

        file.write(
            f"F1-Score : {f1:.4f}\n\n"
        )

        file.write(
            report
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    plot_confusion_matrix(

        y_test,

        predictions,

        class_names
    )


    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ],

        "Score": [
            accuracy,
            precision,
            recall,
            f1
        ]
    })


    metrics.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "model_metrics.csv"
        ),

        index=False
    )


    # --------------------------------------------------------
    # SAVE NORMALIZATION VALUES
    # --------------------------------------------------------

    np.savez(

        os.path.join(
            MODEL_DIR,
            "feature_normalization.npz"
        ),

        mean=mean,

        std=std
    )


    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EMOTION RECOGNITION PROJECT COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nBest model saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nOutputs saved to:"
    )

    print(
        OUTPUT_DIR
    )

if __name__ == "__main__":
    main()