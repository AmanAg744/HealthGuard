Comparative Analysis of LSTM and TCN Models for Anomaly Detection in Healthcare Time Series Data
=================================================================================================

### 📌 Project Overview

This project explores the **comparative performance of Long Short-Term Memory (LSTM)** and **Temporal Convolutional Network (TCN)** models for anomaly detection in **healthcare time-series data**, with a particular focus on **chronic disease monitoring** (e.g., continuous glucose monitoring).

We further design and evaluate a **hybrid LSTM--TCN model** to leverage the complementary strengths of recurrent and convolutional architectures for robust anomaly detection.

* * * * *

🚀 Key Features
---------------

-   **Data Preprocessing Pipeline**

    -   Handles missing values with bidirectional LSTM imputation.

    -   Applies noise reduction via Wavelet Transform.

    -   Aligns asynchronous multi-modal signals with Dynamic Time Warping (DTW).

-   **Model Architectures Implemented**

    -   🧠 **LSTM** with attention gates for long-term dependency modeling.

    -   ⚡ **TCN** with dilated causal convolutions for efficient parallel sequence learning.

    -   🔗 **Hybrid LSTM--TCN** fusion with attention-based feature integration.

-   **Evaluation Framework**

    -   Stratified 5-fold validation.

    -   Class-imbalance handling with focal loss + MAE.

    -   Metrics: Precision, Recall, F1 Score, ROC-AUC, PR-AUC, and confusion matrices.

    -   Robustness testing with synthetic augmentation (noise & sensor dropout).

-   **Visualization Tools**

    -   Patient-level glucose trends.

    -   Daily glucose patterns & anomaly labeling.

    -   Model comparison with PR/ROC curves, radar charts, and confusion matrices.

* * * * *

📊 Results Summary
------------------

| Model | Precision | Recall | F1 Score | ROC-AUC |
| --- | --- | --- | --- | --- |
| **LSTM** | 0.94 | 0.95 | 0.945 | 0.99 |
| **TCN** | 0.81 | 0.99 | 0.89 | 0.99 |
| **Hybrid (LSTM+TCN)** | 0.89 | 0.97 | 0.93 | 0.99 |

✅ The **hybrid model** demonstrated the best balance between **precision and recall**, making it more suitable for real-world clinical applications where both false positives and false negatives are critical.

* * * * *

🗂️ Project Structure
---------------------

`├── dl-proj-bloodgluc.ipynb     # Jupyter Notebook with full implementation
├── DL_mini_proj.pdf            # Code + pipeline documentation
├── DL_mini_Report.pdf          # Detailed research paper/report
├── A-Comparative-Analysis-...  # Project presentation slides
└── README.md                   # Project overview (this file)`

* * * * *

⚙️ Installation & Setup
-----------------------

### 1️⃣ Clone Repository

`git clone https://github.com/your-username/healthcare-anomaly-detection.git
cd healthcare-anomaly-detection`

### 2️⃣ Install Dependencies

`pip install -r requirements.txt`

Main libraries used:

-   `tensorflow` / `keras`

-   `scikit-learn`

-   `pandas`, `numpy`

-   `matplotlib`, `seaborn`

-   `keras-tcn`

### 3️⃣ Run Jupyter Notebook

`jupyter notebook dl-proj-bloodgluc.ipynb`

* * * * *

📂 Dataset
----------

-   **MIMIC-IV** (critical care dataset with 65,000+ ICU patients).

-   **PhysioNet Challenge 2012** dataset.

-   **Synthetic augmentation** (Gaussian noise, sensor dropout) to simulate real-world monitoring conditions.

⚠️ Dataset files are not included in this repo due to size and access restrictions. Follow instructions in the notebook to load datasets.

* * * * *

🔮 Future Work
--------------

-   Real-time deployment on edge devices with resource constraints.

-   Enhanced interpretability with saliency maps & attention visualization.

-   Incorporation of multimodal signals (ECG, respiration, BP) for holistic monitoring.

-   Clinical validation in real-world healthcare settings.

* * * * *

👨‍💻 Authors
-------------

-   Aman Agarwal

    *(Department of Computer Science, MIT Manipal)*
