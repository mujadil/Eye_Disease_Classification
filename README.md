# Eye_Disease_Classification
 
EYE DISEASE CLASSIFICATION
Project Report

 
 
##Abstract
This project focuses on the application of digital image processing techniques to classify eye diseases using a Convolutional Neural Network (CNN). The dataset consists of four classes: cataract, diabetic retinopathy, glaucoma, and normal eyes. The developed model achieves significant accuracy in identifying the diseases and provides a valuable diagnostic aid in ophthalmology.
________________________________________
##Introduction
Digital image processing plays a vital role in medical imaging, especially in the early diagnosis of diseases. This project leverages image classification techniques to distinguish between different eye diseases from retinal images. By using deep learning, specifically CNNs, the project automates disease detection, improving diagnostic speed and accuracy.
________________________________________
##Literature Review
Deep learning has been extensively explored for medical imaging and disease classification. Gulshan et al. (2016) demonstrated the effectiveness of deep learning for detecting diabetic retinopathy, showing high accuracy and specificity in retinal fundus photographs. Similarly, Kermany et al. (2018) highlighted how deep neural networks can diagnose a range of diseases, including ocular conditions, from medical images with expert-level precision. Pratt et al. (2016) applied convolutional neural networks specifically for diabetic retinopathy detection, providing a foundational approach to tackling retinal image-based diagnosis. Furthermore, Li et al. (2019) explored deep learning for detecting glaucomatous optic neuropathy, showcasing its potential in glaucoma screening. Finally, Ronneberger et al. (2015) introduced the U-Net architecture, a pivotal development in biomedical image segmentation, which has since influenced retinal image analysis and disease classification methods.
________________________________________
##Objective
To develop a model capable of accurately classifying eye diseases using retinal images.
________________________________________
##Dataset
The dataset used is the "Eye Diseases Classification" dataset from Kaggle. It contains 4,217 images divided into four classes:
•	Normal: 1,074 images
•	Cataract: 1,038 images
•	Glaucoma: 1,007 images
•	Diabetic Retinopathy: 1,098 images
The dataset was split into training (80%) and validation (20%) sets for model development.
________________________________________
##Methodology
Preprocessing
•	Image resizing: All images were resized to 224x224 pixels.
•	Normalization: Pixel values were normalized to a range of [0, 1].
•	Data batching: The dataset was divided into batches of 32 images for efficient processing.
Model Architecture
A CNN was implemented using TensorFlow and Keras with the following architecture:
1.	Rescaling Layer: Normalizes pixel values.
2.	Convolutional Layers: 
o	32, 64, and 128 filters with ReLU activation.
o	MaxPooling layers to reduce spatial dimensions.
3.	Fully Connected Layers: 
o	A dense layer with 128 neurons.
o	Output layer with softmax activation for classification.
Training
•	Optimizer: Adam
•	Loss Function: Sparse Categorical Crossentropy
•	Metrics: Accuracy
•	Epochs: 10
________________________________________
##Results
The model achieved the following performance:
•	Training Accuracy: 89.6%
•	Validation Accuracy: 82.7%
•	Validation Loss: 0.478
Epoch-wise Performance
Epoch	Training Accuracy	Validation Accuracy
1	52.9%	72.5%
2	74.2%	76.0%
3	78.0%	79.7%
4	80.2%	80.0%
5	81.8%	80.7%
10	89.6%	82.7%
________________________________________
##Visualizations
 
Sample predictions:
•	Predicted Class: Normal
•	Actual Class: Normal
•	Confidence: 94.3%
Images were visualized with predictions and actual labels using matplotlib to ensure the model's interpretability.
________________________________________
##Conclusion
This project demonstrates the potential of CNNs in medical imaging for automated disease classification. With further improvements, such as data augmentation and hyperparameter tuning, the model can achieve higher accuracy and robustness.


________________________________________
##References
1.	Kaggle Dataset: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification
2.	Gulshan, V., Peng, L., Coram, M., et al. (2016). "Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs." JAMA, 316(22), 2402-2410. https://doi.org/10.1001/jama.2016.17216
3.	Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). "Identifying medical diagnoses and treatable diseases by image-based deep learning." Cell, 172(5), 1122-1131. https://doi.org/10.1016/j.cell.2018.02.010
4.	Pratt, H., Coenen, F., Broadbent, D. M., et al. (2016). "Convolutional neural networks for diabetic retinopathy." Procedia Computer Science, 90, 200-205. https://doi.org/10.1016/j.procs.2016.07.014
5.	Li, Z., He, Y., Keel, S., et al. (2019). "Efficacy of a deep learning system for detecting glaucomatous optic neuropathy based on color fundus photographs." Ophthalmology, 125(8), 1199-1206. https://doi.org/10.1016/j.ophtha.2018.10.032
6.	Ronneberger, O., Fischer, P., Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention, 234-241. https://arxiv.org/abs/1505.04597

________________________________________
##Appendix
Code Snippet: Prediction Visualization
for images, labels in val_ds.take(1):
    predictions = model.predict(images)
    i = random.randint(0, len(images) - 1)
    plt.figure(figsize=(5, 5))
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(f"Predicted: {class_names[tf.argmax(predictions[i])]} | Actual: {class_names[labels[i]]}")
    plt.axis("off")
    plt.show()

