# Resnet50-LSTM Deepfake Video Detector

This repository contains a **deep learning solution** designed to identify synthetically generated videos (deepfakes). The system employs a highly specialized **hybrid Resnet50-LSTM architecture** to achieve robust and reliable detection.

---

## Architecture and Logic

The model is structured to analyze both visual inconsistencies and temporal errors, the two main failure points for deepfake generation:

1.  **Feature Extraction (Spatial Domain):** A **pre-trained ResNet50** convolutional neural network serves as the initial feature extractor. It analyzes each video frame to identify **visual artifacts**, unnatural textures, and blending inconsistencies left by the forgery process (e.g., poor blending around facial features).
2.  **Sequence Modeling (Temporal Domain):** The output vectors from the ResNet50 (one per frame) are fed into a **Long Short-Term Memory (LSTM) layer**. The LSTM analyzes the temporal sequence to detect non-human inconsistencies in movement, such as unnatural blinking patterns, head pose glitches, or subtle rhythmic disruptions over time.
3.  **Classification Head:** A dense network makes the final binary classification decision (REAL or FAKE).

### Architecture Diagram

![](Image/Deepfake_System Architecture.png)

---

## Performance Summary

The final, optimized model successfully overcame training instability and class imbalance issues (where the model previously favored the "Fake" class) by implementing aggressive **Class Weighting** and **Early Stopping** strategies.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Final Validation Accuracy** | **90.00%** | Overall correct classification rate on unseen data. |
| **Fake Detection Rate (Recall)** | **94.5%** | The model successfully identified 94.5% of all forged videos (critical for security). |
| **Real Video Reliability (TP Rate)** | **93.3%** | The model correctly identified 93.3% of all genuine videos (critical for avoiding false alarms). |

---

## Optimization Strategy

The final high performance was achieved using the following custom configuration:

* **Optimizer:** Adam (Low Learning Rate: $5 \times 10^{-6}$ for stable Fine-Tuning).
* **Regularization:** High Dropout ($60\%$) and **Image Data Augmentation** (Flip, Rotate, Zoom) to prevent overfitting.
* **Class Weights:** Custom-tuned weights (**Real: 1.15, Fake: 1.05**) were used to balance the model, specifically encouraging the confident identification of **Real** videos (boosting the True Positive rate).
* **Stability:** **Early Stopping** with a patience of 10 epochs prevented resource waste by halting training once the model stabilized.
