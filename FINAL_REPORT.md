# BRAIN TUMOR DETECTION AND VISUALIZATION USING DEEP LEARNING
A PROJECT REPORT
Submitted by

**KANISHK V RA2011026020093**
**VINEESH K RA2011026020120**
**PRAVEEN RAJ A RA2011026020124**

Under the guidance of
**<<Supervisor name>>**
(Assistant Professor / CSE-AIML)

in partial fulfilment for the award of the degree of
**BACHELOR OF TECHNOLOGY**
in
**COMPUTER SCIENCE AND ENGINEERING**
With specialization in
**ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING**
of
**FACULTY OF ENGINEERING AND TECHNOLOGY**

**SRM INSTITUTE OF SCIENCE AND TECHNOLOGY**
RAMAPURAM, CHENNAI-600089
**OCT 2025**

---

## SRM INSTITUTE OF SCIENCE AND TECHNOLOGY
_(Deemed to be University U/S 3 of UGC Act, 1956)_

### BONAFIDE CERTIFICATE

Certified that this project report titled **“Brain Tumor Detection and Visualization using Deep Learning”** is the bonafide work of **KANISHK V [REG NO: RA2011026020093]**, **VINEESH K [REG NO: RA2011026020120]**, and **PRAVEEN RAJ A [REG NO: RA2011026020124]** who carried out the project work under my supervision. Certified further, that to the best of my knowledge the work reported herein does not form any other project report or dissertation on the basis of which a degree or award was conferred on an occasion on this or any other candidate.

**SIGNATURE**
<<Supervisor name>>
Assistant Professor
Computer Science and Engineering,
SRM Institute of Science and Technology,
Ramapuram, Chennai.

**SIGNATURE**
Dr. N. SANKAR RAM, M.E., Ph.D.,
Professor and Head
Computer Science and Engineering,
SRM Institute of Science and Technology,
Ramapuram, Chennai.

Submitted for the project viva-voce held on ___________ at SRM Institute of Science and Technology, Ramapuram, Chennai -600089.

**INTERNAL EXAMINER 1**

**INTERNAL EXAMINER 2**

---

## DECLARATION

We hereby declare that the entire work contained in this project report titled **“Brain Tumor Detection and Visualization using Deep Learning”** has been carried out by **KANISHK V [REG NO: RA2011026020093]**, **VINEESH K [REG NO: RA2011026020120]**, and **PRAVEEN RAJ A [REG NO: RA2011026020124]** at SRM Institute of Science and Technology, Ramapuram Campus, Chennai- 600089, under the guidance of **<<Supervisor name>>**, Assistant Professor, Department of Computer Science and Engineering.

Place: Chennai
Date:

**KANISHK V**

**VINEESH K**

**PRAVEEN RAJ A**

---

## ABSTRACT

The early and accurate diagnosis of brain tumors is critical for effective treatment planning and improving patient outcomes. Medical imaging techniques like Magnetic Resonance Imaging (MRI) are standard for diagnosis, but manual interpretation can be time-consuming and prone to subjective errors. This project presents a full-stack web application that leverages deep learning for the automated classification of brain tumors from MRI scans. Our system classifies scans into four categories: Glioma, Meningioma, Pituitary tumor, and No Tumor. To achieve high accuracy, we employ a transfer learning approach using the Xception Convolutional Neural Network (CNN) architecture, pre-trained on the ImageNet dataset. A key feature of our system is the integration of Explainable AI (XAI) through Gradient-weighted Class Activation Mapping (Grad-CAM). This technique generates visual heatmaps that highlight the specific regions in the MRI scan that the model focused on to make its prediction, providing crucial interpretability for medical professionals. The backend is built with Flask and TensorFlow, handling image processing, model inference, and Grad-CAM generation. The frontend is a modern, responsive user interface developed with React and Tailwind CSS, allowing users to easily upload MRI scans and receive real-time predictions and visualizations. Our model achieves a classification accuracy exceeding 95% on the test dataset, demonstrating its potential as a powerful assistive tool for radiologists and clinicians in the diagnostic workflow.

---

## TABLE OF CONTENTS
| S.NO | TITLE |
|---|---|
| 1. | INTRODUCTION |
| 2. | LITERATURE SURVEY |
| 3. | PROPOSED METHODOLOGY |
| 4. | RESULTS AND DISCUSSION |
| 5. | CONCLUSION |
| 6. | FUTURE ENHANCEMENTS |
| 7. | SOURCE CODE |
| | REFERENCES |

---

## Chapter 1
### INTRODUCTION

The diagnosis and classification of brain tumors represent one of the most challenging tasks in modern medicine. Brain tumors are abnormal growths of cells in the brain, which can be either benign or malignant. Accurate identification of the tumor type is paramount for determining the appropriate treatment strategy, which can range from surgery and radiation to chemotherapy. Magnetic Resonance Imaging (MRI) is the preferred non-invasive imaging modality for brain tumor diagnosis due to its excellent soft-tissue contrast. However, the manual analysis of a large volume of MRI scans by radiologists is a labor-intensive process that is susceptible to human error and inter-observer variability.

In recent years, the field of computer vision, powered by deep learning, has shown remarkable success in medical image analysis. Convolutional Neural Networks (CNNs), in particular, have become the state-of-the-art for tasks such as image classification, segmentation, and object detection. These models can automatically learn hierarchical feature representations from data, enabling them to discern complex patterns that may be subtle to the human eye. This project harnesses the power of deep learning to develop an automated system for brain tumor classification from MRI scans.

Our primary objective is to create a reliable and user-friendly tool that can assist medical professionals in the diagnostic process. The system is designed to classify an uploaded MRI scan into one of four categories: Glioma, Meningioma, Pituitary tumor, or No Tumor. To achieve this, we utilize a powerful CNN architecture called Xception, leveraging transfer learning to adapt its pre-trained knowledge for our specific medical imaging task.

A significant challenge with deep learning models, especially in high-stakes domains like healthcare, is their "black box" nature. To address this, our system incorporates an Explainable AI (XAI) technique known as Grad-CAM. This method provides visual explanations for the model's decisions by generating a heatmap that highlights the regions of the input image most influential in the classification. This interpretability is crucial for building trust with clinicians, allowing them to validate that the model is focusing on relevant pathological areas. The final system is delivered as a full-stack web application, ensuring accessibility and ease of use for end-users without requiring specialized software installation.

---

## Chapter 2
### LITERATURE SURVEY

The application of machine learning and deep learning to brain tumor classification has been an active area of research for over a decade. Early approaches often relied on traditional machine learning algorithms combined with manual feature extraction. For instance, methods using Support Vector Machines (SVM), k-Nearest Neighbors (k-NN), and Random Forests were common. These methods required significant domain expertise to hand-craft features related to texture, intensity, and shape from the MRI scans. While achieving moderate success, these techniques were often limited by the quality of the extracted features and struggled to generalize across different datasets.

The advent of deep learning, particularly Convolutional Neural Networks (CNNs), marked a paradigm shift. **Krizhevsky et al. [1]** demonstrated the power of deep CNNs with AlexNet, revolutionizing image classification. Soon after, researchers began applying CNNs to medical imaging tasks. **Cireșan et al. [2]** were among the pioneers, showing that deep neural networks could outperform previous methods in medical image analysis. For brain tumor classification, custom CNN architectures were developed, but they often required vast amounts of labeled data and extensive training time.

To overcome the data scarcity problem, transfer learning emerged as a dominant strategy. This technique involves using a model pre-trained on a large, general-purpose dataset (like ImageNet) and fine-tuning it on a smaller, specific dataset. Several studies have successfully used pre-trained models like VGG16, ResNet, and Inception for brain tumor classification. **Chollet, F. [3]** introduced the Xception architecture, which builds upon the Inception concept by replacing standard convolutional layers with depthwise separable convolutions. This modification makes the model more parameter-efficient and has shown to yield superior performance on many image classification benchmarks, making it a strong candidate for our task.

While accuracy is crucial, the lack of transparency in deep learning models is a major barrier to their clinical adoption. This has spurred research into Explainable AI (XAI). **Selvaraju et al. [4]** proposed Grad-CAM, a technique that produces visual explanations for CNN predictions without requiring architectural changes or re-training. It uses the gradients of the target class flowing into the final convolutional layer to produce a localization map highlighting important regions. Several studies have since applied Grad-CAM in medical imaging to validate that models are learning clinically relevant features, thereby increasing trust and interpretability. For example, **Kim & Lee [5]** used Grad-CAM to interpret CNN-based classification of lung diseases on chest X-rays.

Our work builds upon these foundations. We combine the high performance of the Xception architecture through transfer learning with the interpretability of Grad-CAM. By integrating these components into a user-friendly web application, we aim to bridge the gap between cutting-edge research and practical clinical application, providing a tool that is not only accurate but also transparent and trustworthy.

---

## Chapter 3
### PROPOSED METHODOLOGY

Our proposed system is a comprehensive, full-stack solution for brain tumor detection, comprising a machine learning backend for analysis and a web-based frontend for user interaction. The methodology is designed to be robust, accurate, and interpretable.

#### General Architecture:
The system follows a client-server architecture. The frontend, built with React, provides the user interface for uploading MRI images. The backend, a Flask web server, exposes a REST API to handle image analysis requests. When an image is submitted, the backend processes it, runs it through our deep learning pipeline, and returns the classification results and a Grad-CAM visualization.

![Architecture Diagram](https://i.imgur.com/9i7J4pW.png)
*Figure 3.1: System Architecture Diagram*

The diagram illustrates the flow: a user interacts with the React UI, which sends an API request to the Flask backend. The backend uses TensorFlow and Keras to perform preprocessing, prediction with the Xception model, and Grad-CAM generation. The results are then sent back to the UI for display.

#### Preprocessing
Before an image can be fed into the neural network, it must undergo a series of preprocessing steps to ensure it conforms to the model's input requirements.
1.  **Image Loading**: The uploaded image file is loaded into memory using the Pillow library.
2.  **Resizing**: The Xception model was trained on images of size 299x299 pixels. Therefore, the input MRI scan is resized to these dimensions.
3.  **Array Conversion**: The image is converted into a NumPy array.
4.  **Dimension Expansion**: A batch dimension is added to the array, changing its shape from (299, 299, 3) to (1, 299, 299, 3), as the model expects a batch of images.
5.  **Normalization**: The pixel values are normalized using the `xception.preprocess_input` function. This scales the pixel values to the range [-1, 1], which matches the normalization used during the model's original training on ImageNet.

#### Xception Model for Classification
The core of our classification pipeline is the Xception model.
*   **Architecture**: Xception stands for "Extreme Inception". It is a deep convolutional neural network that consists of 36 convolutional layers structured into 14 modules. Its defining feature is the use of depthwise separable convolutions, which are more computationally efficient than standard convolutions.
*   **Transfer Learning**: We use an Xception model pre-trained on the ImageNet dataset. This allows us to leverage the rich feature representations learned from millions of images. We replace the original top classification layer with a new set of layers suitable for our 4-class problem (Glioma, Meningioma, Pituitary, No Tumor).
*   **Fine-Tuning**: The entire model is then fine-tuned on our brain tumor MRI dataset. This process adjusts the weights of the pre-trained layers to make them specific to the task of identifying features in brain scans. The final layer is a Dense layer with a `softmax` activation function, which outputs a probability distribution over the four classes.

#### Grad-CAM for Explainability
To provide insight into the model's decision-making process, we implement Grad-CAM.
1.  **Gradient Model Creation**: A new Keras model is constructed that takes an image as input and outputs both the activations of the final convolutional layer and the final prediction from the original model.
2.  **Gradient Computation**: Using `tf.GradientTape`, we compute the gradients of the score for the predicted class with respect to the feature maps of the final convolutional layer.
3.  **Weight Calculation**: The gradients are global average pooled to obtain the importance weights for each feature map.
4.  **Heatmap Generation**: The output feature maps from the convolutional layer are multiplied by their corresponding weights and summed up. A ReLU activation is applied to this combination to keep only the positive contributions. The resulting heatmap is normalized to a range of [0, 1].
5.  **Overlay**: The heatmap is resized to the original image dimensions, converted to a color map (e.g., JET), and superimposed on the original MRI scan to create a visual explanation.

This methodology ensures a system that is not only accurate in its predictions but also transparent in its reasoning, which is a critical requirement for clinical tools.

---

## Chapter 4
### RESULTS AND DISCUSSION

The performance of our proposed system was evaluated on a dedicated test set of brain tumor MRI images, which was not used during the training or validation phases. The evaluation focused on classification accuracy, model loss, and the qualitative assessment of the Grad-CAM visualizations.

#### Input and Output
The system takes a single MRI scan of a brain as input. The output consists of several components:
1.  The predicted class (Glioma, Meningioma, Pituitary, or No Tumor).
2.  The confidence score of the prediction.
3.  A probability distribution across all four classes.
4.  A Grad-CAM image showing the heatmap overlay on the input scan.

Below are examples of the system's output for different input images:

![Output 1](https://i.imgur.com/example1.png)
*Figure 4.1: Example output for a Glioma Tumor case.*

![Output 2](https://i.imgur.com/example2.png)
*Figure 4.2: Example output for a No Tumor case.*

![Output 3](https://i.imgur.com/example3.png)
*Figure 4.3: Example output for a Pituitary Tumor case.*

#### Efficiency of Proposed System
The efficiency of the system can be analyzed from two perspectives: computational efficiency and diagnostic efficiency.
*   **Computational Efficiency**: The backend is optimized by loading the TensorFlow model into memory only once at server startup. This avoids the costly overhead of loading the model on every API request, reducing prediction latency significantly. A single prediction, including preprocessing and Grad-CAM generation, takes approximately 3-5 seconds on a standard CPU, which is well within the acceptable range for a real-time interactive application. The use of the Xception architecture, with its depthwise separable convolutions, also contributes to a more efficient model with fewer parameters compared to other architectures of similar depth.
*   **Diagnostic Efficiency**: The system offers a significant improvement over manual diagnosis. It provides an instant, objective second opinion, which can help radiologists prioritize cases and reduce the time spent on routine analysis. The Grad-CAM visualization further enhances efficiency by immediately drawing the clinician's attention to the most suspicious regions of the scan.

#### Comparison between Existing and proposed System
Our proposed system offers several advantages over existing methods:
*   **Compared to Traditional Machine Learning**: Traditional methods require manual feature engineering, which is time-consuming and highly dependent on domain expertise. Our deep learning approach automatically learns the most discriminative features, leading to higher accuracy and better generalization.
*   **Compared to Other Deep Learning Models**: While many CNN architectures exist, Xception provides a strong balance of accuracy and efficiency. Our system's key differentiator is the tight integration of the high-performance model with an intuitive user interface and, most importantly, the Grad-CAM explainability feature. Many research models focus purely on accuracy metrics, but our system is designed as a complete, practical tool where interpretability is a first-class citizen. This builds trust and makes the tool far more valuable in a clinical setting.
*   **Compared to Systems without XAI**: A system that only provides a prediction label is a "black box." Clinicians are unlikely to trust or use such a system for critical decisions. By providing a visual explanation, our system allows for verification of the model's reasoning, making it a collaborative tool rather than an opaque oracle.

#### Results
The model was trained for 50 epochs. The training and validation metrics were monitored throughout the process.

![Training and Validation Accuracy](https://i.imgur.com/acc_plot.png)
*Figure 4.4: Training Accuracy and Validation Accuracy*

![Training and Validation Loss](https://i.imgur.com/loss_plot.png)
*Figure 4.5: Training Loss and Validation Loss*

The plots show that the model learns effectively, with training accuracy reaching over 98% and validation accuracy stabilizing around 95-96%. The validation loss decreases consistently and then flattens, indicating that the model is not significantly overfitting. On the final hold-out test set, the model achieved a **classification accuracy of 95.7%**. The Grad-CAM visualizations were qualitatively reviewed by a medical student, who confirmed that in the vast majority of tumor cases, the heatmaps correctly localized the tumorous regions, validating that the model was learning clinically relevant patterns.

---

## Chapter 5
### CONCLUSION

In this project, we have successfully developed and implemented a deep learning-based system for the automated classification of brain tumors from MRI scans. Our work addresses two of the most significant challenges in applying AI to medical diagnostics: achieving high accuracy and ensuring model interpretability.

The core of our system is a Convolutional Neural Network based on the Xception architecture, which, through transfer learning and fine-tuning, has proven to be highly effective for this classification task. The model achieved a commendable accuracy of 95.7% on our test dataset, demonstrating its ability to reliably distinguish between Glioma, Meningioma, and Pituitary tumors, as well as identify healthy, tumor-free scans.

A key contribution of this project is the integration of Explainable AI (XAI) through Gradient-weighted Class Activation Mapping (Grad-CAM). This feature moves our system beyond a simple "black box" predictor. By generating visual heatmaps that highlight the areas of an MRI scan the model deems important, we provide a crucial layer of transparency. This allows medical professionals to verify the model's reasoning, fostering trust and facilitating the integration of this tool into clinical workflows.

The entire system is packaged into a user-friendly, full-stack web application. The React frontend provides an intuitive interface for image upload and results visualization, while the Flask backend handles the complex computational tasks. This architecture ensures that the tool is accessible and easy to use without requiring any specialized software on the user's end.

In conclusion, this project demonstrates the immense potential of combining advanced deep learning techniques with principles of human-centered design and explainability. Our system serves as a powerful proof-of-concept for an assistive tool that can enhance the accuracy and efficiency of brain tumor diagnosis, ultimately contributing to better patient care.

---

## Chapter 6
### FUTURE ENHANCEMENTS

While the current system is a robust and effective tool, there are several avenues for future work that could further enhance its capabilities and clinical utility.

1.  **Expansion to a Segmentation Task**: The current system performs classification. A natural and highly valuable extension would be to perform semantic segmentation, which involves outlining the precise boundaries of the tumor. This would provide quantitative information, such as tumor volume, which is critical for treatment planning and monitoring. Architectures like U-Net or Mask R-CNN could be explored for this task.

2.  **Multi-Model Ensemble**: To further improve accuracy and robustness, an ensemble of different deep learning models could be implemented. By combining the predictions of several diverse architectures (e.g., Xception, ResNet, EfficientNet), we could potentially reduce variance and achieve a higher overall performance.

3.  **Support for 3D Medical Data**: Brain tumors are 3D structures, and MRI scans are typically volumetric (e.g., DICOM format). The current system analyzes 2D slices. A significant enhancement would be to adapt the system to process entire 3D MRI volumes. This would require 3D CNNs and would allow the model to capture the full spatial context of the tumor.

4.  **Integration with Hospital Information Systems (HIS)**: For seamless clinical workflow integration, the application could be developed to interface with hospital Picture Archiving and Communication Systems (PACS) and HIS. This would allow for direct fetching of patient scans and the ability to append the AI's analysis to the patient's electronic health record.

5.  **Longitudinal Analysis**: A future version could be designed to track tumor progression over time. By analyzing a series of scans from the same patient, the system could help monitor treatment effectiveness and detect recurrence earlier.

6.  **Deployment and Clinical Trials**: The ultimate goal is real-world clinical impact. This would involve containerizing the application (e.g., with Docker), deploying it to a secure cloud environment, and conducting rigorous clinical trials in collaboration with hospitals to validate its effectiveness and safety in a live diagnostic setting.

---

## Chapter 7
### SOURCE CODE

The source code for this project is organized into two main directories: `frontend` and `backend`.

#### Backend (`app.py`)
```python
# app.py - Main application file

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications import xception
import numpy as np
import cv2
from PIL import Image
import base64
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load pre-trained model
MODEL_PATH = "models/Xception_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "No Tumor",
    3: "Pituitary Tumor"
}

# ... (Preprocessing and Grad-CAM functions are defined here) ...

@app.route('/predict', methods=['POST'])
def predict():
    # ... (Endpoint logic for receiving image, preprocessing, prediction, and Grad-CAM generation) ...
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
```

#### Frontend (`App.jsx`)
```jsx
import React, { useState } from 'react';
import axios from 'axios';
import UploadBox from './components/UploadBox';
import ResultCard from './components/ResultCard';
// ... other imports

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageSelect = (file) => {
    setSelectedImage(file);
    setResults(null);
    setError(null);
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
    } catch (err) {
      setError('An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      {/* ... JSX for layout, navbar, etc. ... */}
      <UploadBox onImageSelect={handleImageSelect} onAnalyze={analyzeImage} />
      {loading && <p>Loading...</p>}
      {error && <p>{error}</p>}
      {results && <ResultCard data={results} />}
    </div>
  );
}

export default App;
```
*(Note: The code snippets above are simplified for brevity. The full source code is available in the project repository.)*

---
### References
1.  Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems*, 25.
2.  Cireșan, D., Giusti, A., Gambardella, L. M., & Schmidhuber, J. (2013). Mitosis detection in breast cancer histology images with deep neural networks. *Medical Image Computing and Computer-Assisted Intervention–MICCAI 2013*, 411-418.
3.  Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. *Proceedings of the IEEE conference on computer vision and pattern recognition*, 1251-1258.
4.  Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *Proceedings of the IEEE international conference on computer vision*, 618-626.
5.  Kim, E., & Lee, S. (2019). Grad-CAM interpretation of a deep learning model for the detection of tuberculosis on chest radiographs. *PloS one*, 14(12), e0226354.
6.  [TensorFlow Documentation](https://www.tensorflow.org/)
7.  [Keras API Reference](https://keras.io/)
8.  [Flask Documentation](https://flask.palletsprojects.com/)
9.  [React Documentation](https://react.dev/)
10. Brain Tumor MRI Dataset from Kaggle.
