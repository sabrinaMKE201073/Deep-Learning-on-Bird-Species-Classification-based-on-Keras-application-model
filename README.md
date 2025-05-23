# Bird-Species-Classification-Training-System
Inception V3, ResNet 152V2, DenseNet 121, and DenseNet 169

---
# ABSTRACT

  For the past few years, the rapid extinction of many bird species due to climate change
and habitat destruction caused by human activities is a growing concern. Accurately tracking the
distribution of these species and understanding the factors that contribute to a region's
biodiversity is essential for creating effective conservation plans. However, the task of
identifying bird species from images is challenging and requires a significant amount of time
and effort. Deep learning-based methods are superior at detecting and localizing birds in images
because they can handle situations when the birds usually portray of various forms, sizes, and
backgrounds. This project proposes an implemented training and validation model system for
bird species classification based on performance of the four Keras deep learning CNN model on
two different datasets based on the evaluation in term of accuracy, loss function, and confusion
matrix. The proposed model is trained using the two set of datasets of bird images which consist
of 1,494 and 2,353 data training sets for 15 rare bird species which can be found in Malaysia and
worldwide. The process of training the model was carried out using ResNet152V2, InceptionV3,
DenseNet121 and DenseNet169 models. Several repeated tests were conducted repeatedly using
Phyton on Google Collab with Tesla T4 GPU. Results showed that DenseNet169 had the best
performance for both datasets, with 95% accuracy for dataset 2 and 85% training and 86%
validation accuracy for dataset 1. This system will be beneficial for ornithologists and
researchers in tracking rare bird species for ecosystem preservation.

---

# DATASET

<p align="center">
  <img src="https://github.com/sabrinaMKE201073/Bird-Species-Classification-Training-System/assets/95947484/c003c44a-463f-4d83-a8c8-724a0686a1aa">
</p>

<p align="center">
  <img src="https://github.com/sabrinaMKE201073/Bird-Species-Classification-Training-System/assets/95947484/38e95281-f2cd-4dc7-adf6-f8d273b0b082">
</p>


Dataset 1: https://drive.google.com/drive/folders/17dRfXkEYmQ4WkVnHqtaygVW3c6E-dfRX?usp=sharing

Dataset 2
https://drive.google.com/drive/folders/1so_7L1fhqBd_i8KM9oZWS1UgYFJ9OYbd?usp=sharing

---

# METHOD

<p align="center">
  <img src="https://github.com/sabrinaMKE201073/Bird-Species-Classification-Training-System/assets/95947484/3c4e4d96-1515-48dc-81b2-f96e3016aa77">
</p>

---

# A PART OF PROCESS ON TRAINING THE DATA

<p align="center">
  <img src="https://github.com/sabrinaMKE201073/Deep-Learning-on-Bird-Species-Classification-based-on-Keras-application-model/assets/95947484/e3e80dc5-7664-461a-91e1-6086ebf5e2c2">
</p>

<p align="center">
  <img src="https://github.com/sabrinaMKE201073/Deep-Learning-on-Bird-Species-Classification-based-on-Keras-application-model/assets/95947484/2f1e3237-628a-4776-8294-c29316b81234">
</p>

---

# RESULTS(DATA VISUALIZATION)

1) Plots of training accuracies & loss using DenseNet169 on dataset 1 and 2
<p align="center">
  <img src="https://github.com/sabrinaMKE201073/Deep-Learning-on-Bird-Species-Classification-based-on-Keras-application-model/assets/95947484/72e0260b-f1c6-4947-8833-eec832f4ff84">
</p>

2) Confusion matrix of different type Keras Moodel on dataset 1
<p align="center">
  <img src="https://github.com/sabrinaMKE201073/Deep-Learning-on-Bird-Species-Classification-based-on-Keras-application-model/assets/95947484/ec2a75df-9190-42aa-a4ed-bf1393349f6d">
</p>

3) Confusion matrix of different type Keras Moodel on dataset 2
<p align="center">
  <img src="https://github.com/sabrinaMKE201073/Deep-Learning-on-Bird-Species-Classification-based-on-Keras-application-model/assets/95947484/18344564-414a-4f14-a76e-1a7713d17392">
</p>

4) Performance Analysis based on Accuracies
<p align="center">
  <img src="https://github.com/sabrinaMKE201073/Bird-Species-Classification-Training-System/assets/95947484/df04f001-4262-4259-95c8-54419bd82fa3">
</p>
   
5) Performance Analysis based on training loss during iteration and Execution time
<p align="center">
  <img src="https://github.com/sabrinaMKE201073/Bird-Species-Classification-Training-System/assets/95947484/5166caa6-8ab0-4a14-a093-73df25018f1b">
</p>



---

🔹 IEEE Paper – Main Contributor & Original First Author
Title: Deep Learning Analysis Using CNN for Bird Species Classification System
Conference: ICICyTA 2023 Publication: IEEE Xplore

Note: I wrote and submitted this paper as the original first author and was the sole contributor to the research and writing for my master project. Without my consent, my lecturer assigned a master's student to present it, which unintentionally led to a change in the author order. I did not approve this change, but the paper remains entirely my work.

Link: https://ieeexplore.ieee.org/document/10428729
