# Noise-Resilient Lip-to-Speech Recognition

## Overview
This project implements a **visual-only lip-reading system** that converts **silent lip movement videos into spoken words**.  
It is designed to operate reliably in **noisy environments** and to assist individuals with **hearing or speech impairments**.

The model leverages **deep learning–based visual speech recognition**, combining:
- **3D Convolutional Neural Networks (3D-CNNs)** for spatio-temporal feature extraction  
- **Temporal Convolutional Networks (TCN) / Transformer-based models** for sequence modeling

---

## Key Features
- **Word-level lip reading** from silent video
- **Audio-independent** (noise-resilient)
- **3D-CNN** for spatial and temporal feature extraction
- **Transformer / TCN** for temporal sequence modeling
- Evaluated on **large-scale public datasets (LRW, LRW-1000)**

---

## Technologies Used
- **Language:** Python  
- **Deep Learning:** TensorFlow / Keras  
- **Computer Vision:** OpenCV, dlib  
- **Models:** 3D CNN, TCN, Transformer Encoder  
- **Optimizer:** Adam with learning-rate scheduling  

---

## Datasets
- **LRW (Lip Reading in the Wild)** – 500 English word classes  
- **LRW-1000** – 1000 Mandarin word/phrase classes  

Both datasets are **publicly available** and widely used in **lip-reading research**.

---

## Methodology
- Face detection and alignment  
- Lip region (ROI) extraction and preprocessing  
- Spatio-temporal feature extraction using **3D-CNN**  
- Temporal modeling using **TCN / Transformer**  
- **Word-level classification**

---

## Results
- **Word Recognition Accuracy:** **62.1%**
- Demonstrates strong performance in **audio-free and noisy environments**
- Confirms the **feasibility of visual-only speech recognition**

---

## Limitations
- Sensitive to **extreme lighting** and **lip occlusion** (e.g., masks)
- Struggles with **accent and speaking-style variations**
- Limited to **word-level recognition** (no sentence generation)

---

## Future Scope
- Sentence-level lip reading  
- Real-time inference optimization  
- Multilingual support  
- Integration with **assistive communication systems**

---

## Authors
**Bhaaskar Krishna**, Nikhil S. Tengli, Harsh Chauhan,  
G. L. Meghana, Shreeshail S. Y  

**REVA University**, Bangalore
