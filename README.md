# LungsNet: Smart Radiologist Assistant

**LungsNet** is a Deep Learning project designed to assist medical professionals in detecting Pneumonia and COVID-19 from Chest X-Ray images. It features an Explainable AI (Grad-CAM) module to visualize the decision-making process of the model.

## Features
- **Deep Learning Model**: Uses DenseNet121/ResNet50 (Transfer Learning) for high accuracy.
- **Explainable AI**: Grad-CAM visualization to highlight infected regions.
- **Interactive Web App**: Built with Streamlit for easy demonstration.

## Project Structure
- `data/`: Contains raw and processed X-Ray images.
- `notebooks/`: Jupyter notebooks for data analysis and model training.
- `app/`: Source code for the Streamlit web application.
- `models/`: Trained model files (.h5).

## Setup Instructions

1.  **Create Virtual Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    streamlit run app/app.py
    ```

## Dataset
This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.
