# Brain Tumor Classification 🧠🔬

This project is a deep learning-based image classification model designed to detect and classify brain tumors using MRI scans. It utilizes a Convolutional Neural Network (CNN) built with TensorFlow/Keras, trained on a publicly available dataset from Kaggle, and deployed through a user-friendly Streamlit app.

## 📁 Dataset

- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- The dataset includes four classes of brain conditions:
  - `Glioma`
  - `Meningioma`
  - `Pituitary`
  - `No Tumor`

## 🧠 Model

- **Type**: Deep Learning - CNN
- **Framework**: TensorFlow (Keras API)
- **Input**: MRI images
- **Output**: Predicted tumor class
- The model is saved as a `.keras` file for reuse and deployment.

## 📊 Performance

- The model was trained and evaluated using accuracy and loss metrics.
- *Achieved 98.1% validation accuracy*

## 🛠️ Tech Stack

- `Python`
- `TensorFlow / Keras`
- `NumPy`, `Matplotlib`, `Pandas` , `Seaborn` , `PIL`
- `Streamlit` (for UI deployment)

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/lixmed/brain-tumor.git
cd brain-tumor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Make sure your environment has Python 3.7+ and `tensorflow` installed.

### 3. Run the Streamlit App

```bash
python -m streamlit run app.py
```

> The `app.py` file loads the trained model and allows users to upload an MRI image to get instant predictions.

## 📌 Project Structure

```bash
brain-tumor/
├── app.py               # Streamlit app
├── model.keras          # Trained model
├── notebook.ipynb       # Model training and evaluation
└── README.md            # Project overview
```

Feel free to star ⭐ the repo or contribute!
