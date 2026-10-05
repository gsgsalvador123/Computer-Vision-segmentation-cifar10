# Visión por Computador y Deep Learning: Segmentación de Imágenes y Clasificación en CIFAR-10

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-brightgreen.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Proyecto práctico de visión por computador centrado en **segmentación no supervisada y cuantización de color** mediante técnicas de clustering, junto con el diseño y optimización de **arquitecturas de deep learning supervisadas** para la clasificación de imágenes en el benchmark **CIFAR-10**.

---

## 📌 Executive Summary

This project investigates core computer vision methodologies divided into two fundamental sections:
1. **Unsupervised Image Segmentation & Color Quantization:**
   - Benchmarked **K-Means clustering** and **Expectation-Maximization (EM / Gaussian Mixture Models)** for color quantization and image compression.
   - Evaluated the geometric impact of **covariance matrix structures** (Spherical, Diagonal, and Full/Generic) on cluster distributions.
   - Explored perceptual color spaces (**RGB**, **YCrCb**, and **CIELAB**), demonstrating how separating luminance ($Y / L^*$) from chrominance preserves fine structural details while drastically compressing color palettes.

2. **Deep Learning Classification on CIFAR-10:**
   - Designed and evaluated four neural network architectures from linear baselines to deep convolutional models.
   - Diagnosed severe overfitting in unregularized deep networks (21% generalization gap).
   - Engineered an optimization pipeline featuring **Batch Normalization**, **progressive Dropout (0.2 $\rightarrow$ 0.5)**, **Data Augmentation**, and adaptive learning rate decay (**`ReduceLROnPlateau`**), improving test accuracy from **38.5%** to **85.3%**.

---

## 📊 Benchmark Results (CIFAR-10)

| Model Architecture | Train Acc | Test Acc | Train Loss | Test Loss | Generalization Gap | Key Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Linear Baseline (ffNN Base)** | 41.0% | 38.5% - 39.0% | 1.70 | 1.77 | ~2% | Single `Dense(10)` layer; lacks capacity to model spatial hierarchies. |
| **Deep Dense ffNN (BN + Dropout)** | 69.6% | 52.5% | 0.86 | 1.33 | 17.1% | 3 hidden layers (1024-512-256); hits spatial limitation of flattened pixels. |
| **Unregularized CNN (VGG-style)** | 97.6% | 76.3% | 0.21 | 1.38 | **21.3%** | 3 `Conv-Conv-Pool` blocks; heavy memorization of training data. |
| **Regularized CNN (BN + Progressive Dropout)** | 88.9% | **85.3%** | 0.32 | **0.46** | **3.6%** | Batch Normalization + layer-dependent dropout (0.2 to 0.5); peak test accuracy. |
| **Fully Augmented CNN + Adaptive LR** | 81.5% | **81.8%** | 0.53 | 0.53 | **0.0%** | `ImageDataGenerator` + `ReduceLROnPlateau`; **completely eliminates overfitting**. |

---

## 🔬 Key Methodologies & Findings

### 1. Color Quantization & Segmentation (K-Means vs. EM)
* **K-Means Clustering:**
  - Evaluated on grayscale (1D) and RGB (3D) color spaces.
  - Demonstrated indexed color compression: reducing from 24 bits/pixel (True Color) to 2, 3, and 4 bits/pixel yields theoretical compression factors up to **$12\times$**.
* **Expectation-Maximization (EM) & Gaussian Mixtures:**
  - Unlike hard assignment in K-Means (Euclidean distance), EM performs soft probabilistic assignment using the **Mahalanobis distance**.
  - Compared covariance constraints:
    - **Spherical ($\sigma^2 I$):** Equivalent to K-Means; circular clusters.
    - **Diagonal:** Axis-aligned elliptical clusters.
    - **Full (Generic):** Rotated ellipsoids that capture inter-channel correlation between chrominance bands ($Cr$ and $Cb$).
* **Color Spaces (RGB vs. YCrCb vs. CIELAB):**
  - In **RGB**, clustering degrades edges and shading because intensity is entangled across all channels.
  - In **YCrCb**, leaving luminance ($Y$) unquantized while clustering chrominance ($Cr, Cb$) maintains sharp structural fidelity with as few as 3–4 color clusters.
  - **CIELAB ($L^*a^*b^*$)** provides perceptual uniformity: Euclidean distance ($\Delta E$) aligns with human visual perception, making it optimal for visual color quantization.

---

### 2. Deep Learning Pipeline & Overfitting Mitigation
* **Spatial Representations Matter:** Transitioning from dense networks (flattened $32\times32\times3$ inputs) to convolutional feature extractors boosted test accuracy from $52.5\%$ to $76.3\%$.
* **The Regularization Breakthrough:**
  - Adding **Batch Normalization** stabilized internal activations and gradient flow.
  - Applying **progressive Dropout** (higher rates in deeper layers: 0.2 $\rightarrow$ 0.3 $\rightarrow$ 0.4 $\rightarrow$ 0.5) tackled the excess capacity of layers with 128+ filters, raising test accuracy to **85.3%** and shrinking validation loss from 1.38 down to 0.46.
* **Data Augmentation & Generalization:**
  - Real-time transformations (rotation, horizontal flip, width/height shifts, zoom) presented novel variations each epoch.
  - **Zero Overfitting:** Closed the generalization gap completely ($81.5\%$ train vs. $81.8\%$ validation). While converging more slowly within fixed epoch budgets, it produced the most robust model for real-world domain shifts.

---

## 📁 Repository Structure

```text
Computer-Vision-segmentation-cifar10/
├── computer_vision_segmentation_cifar10.ipynb   # Main Jupyter Notebook (code, experiments, plots & discussion)
├── requirements.txt                              # Project dependencies
├── .gitignore                                    # Git ignore rules
└── README.md                                     # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gsgsalvador123/Computer-Vision-segmentation-cifar10.git
   cd Computer-Vision-segmentation-cifar10
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch the Jupyter Notebook:**
   ```bash
   jupyter notebook computer_vision_segmentation_cifar10.ipynb
   ```

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python
- **Computer Vision:** OpenCV (`cv2`)
- **Deep Learning:** TensorFlow & Keras
- **Scientific Computing:** NumPy, SciPy
- **Data Visualization:** Matplotlib

---

## 👤 Author

**Salvador Gonzalo Guevara**  
- GitHub: [@gsgsalvador123](https://github.com/gsgsalvador123)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
