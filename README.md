# 🫁 LungsNet: Smart Radiologist Assistant

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **"Opening the Black Box of Medical AI"**

**LungsNet** is a high-performance **Computer-Aided Diagnosis (CAD)** system designed to assist medical professionals in detecting **Pneumonia** from Chest X-Ray images. By integrating state-of-the-art Deep Learning architectures with Explainable AI (XAI), LungsNet not only predicts diseases with high accuracy but also provides visual transparency into its decision-making process.

---

## 🌟 Key Features

*   **⚡ High Accuracy**: Powered by **DenseNet121** with Transfer Learning, achieving **~94.23% accuracy** on test data.
*   **🔍 Explainable AI (XAI)**: Integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** technology to visualize the "Region of Interest" (ROI), highlighting the infected areas used for diagnosis.
*   **🏥 Medical-Grade Recall**: Optimized for high sensitivity (Recall) to minimize False Negatives, ensuring fewer missed diagnoses.
*   **💻 Interactive Dashboard**: User-friendly web application built with **Streamlit** for real-time demonstration and testing.

---

## 🧠 Methodology

### 1. Model Architecture
We utilize **DenseNet121**, an architecture known for its parameter efficiency and dense connectivity.
*   **Input**: 224x224x3 Chest X-Ray images.
*   **Backbone**: DenseNet121 (Pre-trained on ImageNet).
*   **Head**: Custom Global Average Pooling + Dropout (0.5) + Sigmoid Activation.

### 2. Dataset
*   **Source**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
*   **Preprocessing**: Resizing, Normalization, and extensive Data Augmentation (Rotation, Zoom, Shift).
*   **Handling Imbalance**: Applied oversampling to the minority 'Normal' class to ensure balanced training.

---

## 📊 Performance Results

The model was evaluated on a held-out test set of 624 images.

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | **94.23%** | High overall correctness |
| **Precision** | **93.5%** | Reliable positive predictions |
| **Recall** | **98.0%** | **Critical for Medical AI** (Low False Negative) |
| **F1-Score** | **95.7%** | Balanced performance |

> *The high Recall score indicates that the model is extremely effective at flagging potential pneumonia cases, which is the primary requirement for a screening tool.*

---

## 🚀 Installation & Usage

### Prerequisites
*   Python 3.8+
*   Git

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/HisyamAlammar/LungsNet.git
    cd LungsNet
    ```

2.  **Create a Virtual Environment**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Web Application**
    ```bash
    streamlit run app/app.py
    ```

---

## 📂 Project Structure

```
LungsNet/
├── app/                    # Streamlit Application source code
├── data/                   # Raw and processed datasets
├── models/                 # Saved .h5 model files
├── notebooks/              # Jupyter Notebooks for experiments
│   ├── 01_Data_Acquisition.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Evaluation.ipynb
├── scripts/                # Utility scripts for data prep
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📚 References

1.  *G. Huang et al., "Densely Connected Convolutional Networks," CVPR 2017.*
2.  *R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks," ICCV 2017.*
3.  *Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," Cell 2018.*

---

<p align="center">
  Made with ❤️ by <b>Hisyam Alammar</b>
</p>
