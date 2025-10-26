# Eigenfaces-Based Face Recognition (PCA Implementation)

## Overview
This project implements the **Eigenfaces method** for face recognition, based on **Principal Component Analysis (PCA)**. The algorithm is developed entirely **from scratch in Python**, without using high-level PCA or recognition utilities.  
The project demonstrates how varying the number of eigenfaces (M) affects reconstruction quality, classification accuracy, and noise robustness.

---

## Features
- Implementation of PCA without using `sklearn.decomposition.PCA`
- Mean face and eigenface computation
- Face reconstruction with varying numbers of eigenfaces
- Recognition accuracy evaluation with confusion matrices
- Robustness tests under Gaussian and Salt-and-Pepper noise

---

## Project Structure
```
.
├── eigenfaces.py              # Main implementation script
├── BLG202_PROJECTREPORT.pdf   # Project report (IEEE format)
├── Numerical_Methods_Project.pdf  # Original assignment guideline
├── /output/
│   ├── mean_face_all.png
│   ├── mean_face_subset.png
│   ├── eigenfaces/
│   ├── reconstructed/
│   ├── recognition/
│   └── noise/
└── README.md
```

---

## Methodology

### 1. Data Preprocessing
- Uses the [ORL Face Dataset (AT&T Database of Faces)](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces)
- Each image resized and vectorized into a matrix form  
- Mean face subtracted to normalize the dataset  

### 2. PCA and Eigenfaces
- Covariance matrix computed as \( C = XX^T \)
- Eigenvalues and eigenvectors extracted and projected to image space
- Top M eigenfaces selected and visualized

### 3. Face Reconstruction
- Each face reconstructed using M eigenfaces  
- Mean Squared Error (MSE) used to evaluate reconstruction quality  

### 4. Face Classification
- Faces projected into eigenspace  
- Classified using nearest mean method per individual  

### 5. Robustness Testing
- Gaussian and Salt-and-Pepper noise applied to test stability  
- Recognition accuracy plotted as a function of noise level  

---

## Experimental Results
- **Optimal eigenface count (M):** 200  
- **Highest classification accuracy:** 98.75%  
- **Minimum M for MSE < 500:** 50  
- **Noise robustness:** Stable under Gaussian noise, more sensitive to Salt-and-Pepper distortions  

---

## Installation & Usage

### Requirements
```bash
numpy
scipy
matplotlib
pandas
scikit-learn
```

### Run
```bash
python eigenfaces.py --data_path path/to/ORL_faces --output_path path/to/output_directory
```

### Example Output Directory
```
/output/
│-- mean_face_all.png
│-- eigenvalues.png
│-- cumulative_variance.png
│-- eigenfaces/
│-- reconstructed/
│-- recognition/
│-- noise/
```

---

## Reference
- M. Turk and A. Pentland, *"Eigenfaces for Recognition"*, *Journal of Cognitive Neuroscience*, vol. 3, no. 1, pp. 71–86, 1991.

---

## Author
**Berker Eriş**  
Artificial Intelligence and Data Engineering Department  
Istanbul Technical University  
berkereris1@gmail.com
