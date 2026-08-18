import os
import sys
import json

import numpy as np
import librosa
import tensorflow as tf

# ============================================================
# SPEECH EMOTION PREDICTION
# MFCC + DELTA + DELTA-DELTA + CNN
# ============================================================

MODEL_PATH = "models/speech_emotion_cnn.keras"

CLASS_PATH = "models/emotion_classes.json"

NORMALIZATION_PATH = (
    "models/feature_normalization.npz"
)

SAMPLE_RATE = 16000

N_MFCC = 40

MAX_FRAMES = 174

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Normalize audio
    audio = librosa.util.normalize(
        audio
    )

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=256
    )

    # First-order derivative
    delta = librosa.feature.delta(
        mfcc
    )

    # Second-order derivative
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Create 3-channel feature representation
    features = np.stack(
        [
            mfcc,
            delta,
            delta2
        ],
        axis=-1
    )

    # Pad or truncate to fixed number of frames
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
# PREDICT EMOTION
# ============================================================

def predict_emotion(audio_path):

    # --------------------------------------------------------
    # Check audio file
    # --------------------------------------------------------

    if not os.path.exists(
        audio_path
    ):

        raise FileNotFoundError(
            f"Audio file not found:\n"
            f"{audio_path}"
        )

    if not audio_path.lower().endswith(
        ".wav"
    ):

        raise ValueError(
            "Please provide a WAV audio file."
        )


    # --------------------------------------------------------
    # Check required model files
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}\n\n"
            "Run 'python main.py' first."
        )

    if not os.path.exists(
        CLASS_PATH
    ):

        raise FileNotFoundError(
            f"Class file not found:\n"
            f"{CLASS_PATH}"
        )

    if not os.path.exists(
        NORMALIZATION_PATH
    ):

        raise FileNotFoundError(
            f"Normalization file not found:\n"
            f"{NORMALIZATION_PATH}\n\n"
            "Run 'python main.py' again to "
            "generate this file."
        )


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading trained model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    # --------------------------------------------------------
    # Load emotion classes
    # --------------------------------------------------------

    with open(
        CLASS_PATH,
        "r"
    ) as file:

        class_names = json.load(
            file
        )


    # --------------------------------------------------------
    # Load training normalization values
    # --------------------------------------------------------

    normalization = np.load(
        NORMALIZATION_PATH
    )

    mean = normalization[
        "mean"
    ]

    std = normalization[
        "std"
    ]


    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    features = extract_features(
        audio_path
    )


    # --------------------------------------------------------
    # Apply same normalization used during training
    # --------------------------------------------------------

    features = (
        features - mean
    ) / std


    # --------------------------------------------------------
    # Add ONLY the batch dimension
    #
    # Model expects:
    # (batch, 40, 174, 3)
    #
    # features before this:
    # (40, 174, 3)
    #
    # features after this:
    # (1, 40, 174, 3)
    # --------------------------------------------------------

    
    # --------------------------------------------------------
    # Prepare model input
    # Model expects:
    # (batch, 40, 174, 3)
    # --------------------------------------------------------

    features = np.asarray(
        features,
        dtype=np.float32
    )

    # Remove any accidental extra dimensions
    features = np.squeeze(
        features
    )

    # Feature extraction must produce:
    # (40, 174, 3)

    if features.shape != (
        N_MFCC,
        MAX_FRAMES,
        3
    ):

        raise ValueError(
            f"Unexpected feature shape: "
            f"{features.shape}. "
            f"Expected: "
            f"({N_MFCC}, {MAX_FRAMES}, 3)"
        )

    # Add exactly ONE batch dimension
    features = np.expand_dims(
        features,
        axis=0
    )

    print(
        f"Model input shape: {features.shape}"
    )

    # Prediction
    probabilities = model.predict(
        features,
        verbose=0
    )[0]


    predicted_index = int(
        np.argmax(
            probabilities
        )
    )

    predicted_emotion = (
        class_names[
            predicted_index
        ]
    )

    confidence = (
        probabilities[
            predicted_index
        ] * 100
    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "       SPEECH EMOTION PREDICTION"
    )

    print(
        "=" * 60
    )

    print(
        f"\nAudio File : "
        f"{os.path.basename(audio_path)}"
    )

    print(
        f"Emotion    : "
        f"{predicted_emotion.upper()}"
    )

    print(
        f"Confidence : "
        f"{confidence:.2f}%"
    )

    print(
        "\nEmotion probabilities:"
    )


    for emotion, probability in zip(
        class_names,
        probabilities
    ):

        print(
            f"{emotion:<12}"
            f"{probability * 100:>7.2f}%"
        )


    print(
        "=" * 60
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python src/predict.py sample.wav"
        )

        print(
            "\nExample:"
        )

        print(
            "python src/predict.py sample.wav"
        )

        sys.exit(1)


    audio_path = sys.argv[1]

    predict_emotion(
        audio_path
    )
