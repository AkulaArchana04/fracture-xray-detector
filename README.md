****PROJECT NAME:** **🦴 X-ray Fracture Detection AI**

A deep learning-powered web app built with TensorFlow and Streamlit to detect bone fractures from X-ray images. Upload an image, and the AI predicts whether the bone is fractured or normal — along with confidence visualization.


## 🚀 Features

- ✅ Upload bone X-ray images (JPEG, PNG)
- 🧠 Predicts **Fractured** or **Normal** using a trained deep learning model
- 📊 Confidence score with dynamic progress bar
- 🎨 Clean UI with custom CSS for modern aesthetics
- ⚡ Real-time inference in the browser using Streamlit


## 🧠 Model Info

The model used is a Convolutional Neural Network (CNN) trained on a labeled dataset of bone X-ray images (Fractured / Normal). The input image is resized to **64x64** and normalized before inference.

> **Model File:** `fracture_xray_model.h5`


## 🛠 Tech Stack

- [TensorFlow/Keras](https://www.tensorflow.org/) – For training and loading the model  
- [Streamlit](https://streamlit.io/) – To create the interactive UI  
- [NumPy](https://numpy.org/) – Image preprocessing  
- [Pillow](https://python-pillow.org/) – Image handling


## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AkulaArchana04/xray-fracture-detection.git
   cd xray-fracture-detection


**Place the trained model:**
 
 Ensure your fracture_xray_model.h5 file is in the root directory.

**Run the Streamlit app:**

 streamlit run app.py



   
