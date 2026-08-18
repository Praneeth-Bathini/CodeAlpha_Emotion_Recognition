# Speech Emotion Recognition using MFCC and CNN

---

## Project Overview

This project develops a **Speech Emotion Recognition (SER)** system using Machine Learning and Deep Learning techniques.

The system analyzes speech recordings and predicts the emotion expressed in the speech.

The project uses the **RAVDESS Speech dataset** and extracts three types of audio features:

- MFCC (Mel-Frequency Cepstral Coefficients)
- Delta MFCC
- Delta-Delta MFCC

These features are combined and provided to a **Convolutional Neural Network (CNN)** for emotion classification.

The model recognizes eight different emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

The project also includes a prediction system that can classify the emotion of a new `.wav` audio file.

---

## Objective

The main objective of this project is to develop a deep learning model that can automatically recognize human emotions from speech audio.

The project demonstrates:

- Audio preprocessing
- Speech feature extraction
- MFCC analysis
- Delta and Delta-Delta feature extraction
- CNN-based classification
- Actor-independent model evaluation
- Emotion prediction from new audio

---

## Dataset

The project uses the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** Speech dataset.

The dataset contains:

- **1,440 speech audio files**
- **24 professional actors**
- **8 emotion categories**
- WAV audio format

### Dataset Source

Official RAVDESS dataset:

https://zenodo.org/records/11063852

The dataset is used according to its applicable license and attribution requirements.

---

## Emotion Classes

The emotions are encoded in the RAVDESS filenames.

| Code | Emotion   |
|------|-----------|
| 01   | Neutral   |
| 02   | Calm      |
| 03   | Happy     |
| 04   | Sad       |
| 05   | Angry     |
| 06   | Fearful   |
| 07   | Disgust   |
| 08   | Surprised |

---

## Features Used

### MFCC

**Mel-Frequency Cepstral Coefficients (MFCCs)** are widely used features for representing important characteristics of speech signals.

Configuration:

- Sampling rate: 16,000 Hz
- MFCC coefficients: 40
- FFT size: 512
- Hop length: 256

### Delta Features

Delta features represent the first-order changes in the MFCC features over time.

### Delta-Delta Features

Delta-Delta features represent the second-order changes in MFCC features.

### Final Feature Representation

The three feature types are combined as three CNN input channels:

```text
Channel 1 -> MFCC
Channel 2 -> Delta
Channel 3 -> Delta-Delta
```

Final feature shape:

```text
40 x 174 x 3
```

---

## Machine Learning Workflow

```text
RAVDESS Speech Dataset
          |
          v
    Audio Loading
          |
          v
  Audio Normalization
          |
          v
   MFCC Extraction
          |
     +----+----+
     |         |
     v         v
   Delta    Delta-Delta
     |         |
     +----+----+
          |
          v
  3-Channel Features
          |
          v
 Feature Normalization
          |
          v
Actor-Based Data Split
          |
          v
      CNN Model
          |
          v
    Model Training
          |
    +-----+-----+-----+
    |           |     |
    v           v     v
Early Stop  Class   Learning
            Weights  Rate Reduction
    |           |     |
    +-----+-----+-----+
          |
          v
 Best Model Selection
          |
          v
  Model Evaluation
          |
          v
Classification Report
          |
          v
 Confusion Matrix
          |
          v
 New Audio Prediction
```

---

## Actor-Based Dataset Splitting

Instead of randomly splitting individual audio files, the dataset is divided according to actor IDs.

This helps reduce speaker/actor data leakage because the actors used for testing are not included in the training data.

```text
Actors 01-18 -> Training
Actors 19-21 -> Validation
Actors 22-24 -> Testing
```

### Dataset Split

| Dataset    | Samples |
|------------|---------|
| Training   | 1,080   |
| Validation | 180     |
| Testing    | 180     |
| Total      | 1,440   |

---

## CNN Architecture

The project uses a Convolutional Neural Network designed for the three-channel MFCC feature representation.

The architecture contains:

- Conv2D layers
- Batch Normalization
- Max Pooling
- Dropout
- Global Average Pooling
- Dense layer
- Softmax output layer

The final output layer contains **8 neurons**, corresponding to the eight emotion classes.

The model contains approximately **1.8 million parameters**.

---

## Training Techniques

### Class Weighting

Class weights are calculated from the training data and provided to the CNN during training.

### Early Stopping

Training automatically stops when validation performance stops improving for several consecutive epochs.

### Learning Rate Reduction

The learning rate is reduced when validation loss stops improving.

### Model Checkpointing

The model with the best validation accuracy is automatically saved.

---

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

---

## Final Test Results

The final model was evaluated on the actor-independent test set containing **180 recordings**.

| Metric    | Score      |
|-----------|------------|
| Accuracy  | **31.11%** |
| Precision | **64.58%** |
| Recall    | **31.11%** |
| F1-Score  | **27.90%** |

### Classification Report

| Emotion   | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Angry     | 0.00      | 0.00   | 0.00     |
| Calm      | 0.25      | 0.04   | 0.07     |
| Disgust   | 0.24      | 0.08   | 0.12     |
| Fearful   | 0.33      | 0.08   | 0.13     |
| Happy     | 0.27      | 0.79   | 0.40     |
| Neutral   | 0.50      | 0.83   | 0.63     |
| Sad       | 0.00      | 0.00   | 0.00     |
| Surprised | 1.00      | 0.67   | 0.80     |

The model performs better on emotions such as **Happy, Neutral, and Surprised**, while emotions such as **Angry and Sad** remain more difficult to classify.

---

## New Audio Prediction

The trained model was tested using a separate WAV audio file.

Example:

```text
Audio File : sample.wav
Emotion    : HAPPY
Confidence : 48.33%
```

The model also produces probabilities for all eight emotions:

```text
angry          1.63%
calm           9.21%
disgust        3.88%
fearful        7.38%
happy         48.33%
neutral       22.37%
sad            3.91%
surprised      3.29%
```

---

## Project Structure

```text
CodeAlpha_Emotion_Recognition/
|
+-- data/
|   +-- Audio_Speech_Actors_01-24_16k/
|       +-- Actor_01/
|       +-- Actor_02/
|       +-- Actor_03/
|       +-- ...
|       +-- Actor_24/
|
+-- models/
|   +-- speech_emotion_cnn.keras
|   +-- emotion_classes.json
|   +-- feature_normalization.npz
|
+-- outputs/
|   +-- emotion_distribution.png
|   +-- training_accuracy.png
|   +-- training_loss.png
|   +-- confusion_matrix.png
|   +-- classification_report.txt
|   +-- model_metrics.csv
|
+-- src/
|   +-- predict.py
|
+-- main.py
+-- requirements.txt
+-- README.md
```

---

## Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras
- Convolutional Neural Network (CNN)

### Audio Processing

- Librosa
- MFCC
- Delta
- Delta-Delta

### Data Processing

- NumPy
- Pandas

### Machine Learning

- Scikit-learn

### Visualization

- Matplotlib
- Seaborn

---

## Installation

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Download the RAVDESS Speech dataset from the official source.

Extract the dataset into the `data` directory.

The final structure should be:

```text
data/
└── Audio_Speech_Actors_01-24_16k/
    ├── Actor_01/
    ├── Actor_02/
    ├── Actor_03/
    ├── ...
    └── Actor_24/
```

Each actor folder should contain WAV audio files.

---

## Train the Model

From the project root, run:

```bash
python main.py
```

The program will:

1. Scan the RAVDESS dataset.
2. Find the WAV audio files.
3. Extract MFCC features.
4. Extract Delta features.
5. Extract Delta-Delta features.
6. Create the three-channel feature representation.
7. Normalize the features.
8. Split the data using actor IDs.
9. Calculate class weights.
10. Build the CNN.
11. Train the model.
12. Save the best model.
13. Generate training graphs.
14. Generate the confusion matrix.
15. Generate the classification report.
16. Save model metrics.

---

## Predict Emotion from New Audio

After training, place a WAV file in the project root.

For example:

```text
sample.wav
```

Run:

```bash
python src/predict.py sample.wav
```

The prediction program displays:

- Audio filename
- Predicted emotion
- Confidence
- Probability of every emotion

Example:

```text
============================================================
       SPEECH EMOTION PREDICTION
============================================================

Audio File : sample.wav
Emotion    : HAPPY
Confidence : 48.33%

Emotion probabilities:
angry          1.63%
calm           9.21%
disgust        3.88%
fearful        7.38%
happy         48.33%
neutral       22.37%
sad            3.91%
surprised      3.29%
============================================================
```

---

## Generated Outputs

### Emotion Distribution

```text
outputs/emotion_distribution.png
```

Displays the distribution of the eight emotion classes.

### Training Accuracy

```text
outputs/training_accuracy.png
```

Shows training and validation accuracy across epochs.

### Training Loss

```text
outputs/training_loss.png
```

Shows training and validation loss across epochs.

### Confusion Matrix

```text
outputs/confusion_matrix.png
```

Shows actual versus predicted emotion classes.

### Classification Report

```text
outputs/classification_report.txt
```

Contains detailed precision, recall and F1-score for each emotion.

### Model Metrics

```text
outputs/model_metrics.csv
```

Contains the final Accuracy, Precision, Recall and F1-Score.

---

## Saved Model Files

The trained CNN is saved as:

```text
models/speech_emotion_cnn.keras
```

Emotion class names are stored in:

```text
models/emotion_classes.json
```

Training feature normalization values are stored in:

```text
models/feature_normalization.npz
```

These files are required by the prediction script.

---

## Testing

The project was successfully tested with a WAV audio file.

The prediction pipeline successfully:

- Loaded the trained CNN
- Loaded the emotion classes
- Loaded feature normalization parameters
- Extracted MFCC features
- Extracted Delta features
- Extracted Delta-Delta features
- Created the correct CNN input shape
- Generated an emotion prediction
- Generated probabilities for all emotion classes

The tested model input shape was:

```text
(1, 40, 174, 3)
```

Example prediction:

```text
Emotion    : HAPPY
Confidence : 48.33%
```

---

## Key Learning Outcomes

This project provided practical experience with:

- Speech signal processing
- Audio preprocessing
- MFCC feature extraction
- Delta feature extraction
- Delta-Delta feature extraction
- Deep learning
- CNN architecture
- Audio classification
- Actor-independent dataset splitting
- Class weighting
- Feature normalization
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Confusion matrix analysis
- Classification reports
- Real-world audio prediction

---

## Limitations

The model achieved **31.11% accuracy** on the actor-independent test set.

Speech emotion recognition is challenging because emotions can vary based on:

- Speaker characteristics
- Voice pitch
- Speaking speed
- Intensity
- Pronunciation
- Recording conditions
- Similarity between certain emotions

The model performs better for some emotions than others, while Angry and Sad emotions remain difficult for the current model.

The current implementation focuses on demonstrating an end-to-end speech emotion recognition pipeline rather than achieving state-of-the-art performance.

---

## Future Improvements

Possible improvements include:

- Using larger speech emotion datasets
- Data augmentation
- Pitch and chroma features
- Mel-spectrogram features
- CNN-LSTM architectures
- Bidirectional LSTM
- Transformer-based speech models
- Transfer learning
- Pretrained speech representation models
- Hyperparameter optimization
- Advanced audio augmentation techniques

---

## Conclusion

This project implements an end-to-end **Speech Emotion Recognition system using MFCC, Delta, Delta-Delta features and a Convolutional Neural Network**.

The system successfully processes the RAVDESS speech dataset, extracts audio features, trains a CNN using actor-independent data splitting, evaluates the model using multiple classification metrics, generates visualizations, saves the trained model, and predicts emotions from new WAV audio files.

The final model achieved:

```text
Accuracy  : 31.11%
Precision : 64.58%
Recall    : 31.11%
F1-Score  : 27.90%
```

The project demonstrates the complete workflow from **raw speech audio to emotion prediction**.

---

## Internship

This project was developed as part of the:

**CodeAlpha Machine Learning Internship**

### Task 2: Emotion Recognition from Speech
