# CNN Feature Map Visualization

### 📌 Project Overview
This project demonstrates how a **Convolutional Neural Network (CNN)** learns and extracts hierarchical features from an image.  
A 5-block CNN was built from scratch using **TensorFlow/Keras**, and the **feature maps from each convolutional block** were visualized to understand how the network perceives different levels of abstraction.

---

### 🧰 Technologies Used
- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**

---

### ⚙️ Experiment Workflow

#### **Step 1 – Imports**
Loaded the required libraries for deep learning, image preprocessing, and visualization.

#### **Step 2 – Load & Preprocess Image**
- Input image: `golden_retriever.jpg`
- Resized to **128×128**
- Normalized pixel values to [0, 1]
- Expanded dimensions to form a batch of shape `(1, 128, 128, 3)`

#### **Step 3 – Build CNN Architecture**
The network consists of **5 convolutional blocks**, each containing:
- `Conv2D` layer with ReLU activation  
- `MaxPooling2D` layer  

| Block | Conv Filters | Output Shape (H×W×C) |
|-------|---------------|----------------------|
| Block 1 | 16 | (64, 64, 16) |
| Block 2 | 32 | (32, 32, 32) |
| Block 3 | 64 | (16, 16, 64) |
| Block 4 | 128 | (8, 8, 128) |
| Block 5 | 256 | (4, 4, 256) |

---

### 🔍 **Step 4 – Feature Map Extraction**
The model outputs the feature maps after each block’s pooling layer.  
A forward pass was performed to get intermediate outputs:
```python
feature_maps = model.predict(x)
