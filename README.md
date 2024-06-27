# Vehicle detection and Classification from images

*Created by Albert Dańko and Michał Jurzak for a Fundamentals of Artificial Intelligence AGH course.*

## Data

The problem of object identification and classification is reduced to a regression problem using YOLO techniques. The YOLO algorithm attempts to reformulate object detection into a single regression problem, including image pixels to class probabilities and bounding box coordinates. Therefore, the algorithm only needs to look at the image once to predict and locate target objects in the images.

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/ded3cd97-3793-49a8-8957-a6a1d9628f5e)

The data consists of images likely from the Indian region, labeled with a .txt file containing a class number and the coordinates of the bounding box containing the object. The dataset found on Kaggle: [Road Vehicle Images Dataset](https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset) provides 3004 images and has 21 different classes.

For the established dataset, 10% of the data is designated for validation (testing), which will later be changed. Initial analysis of the set showed varying image sizes. The most popular resolutions include 640x360, 360x640, and 480x640. The first two resolutions account for almost 66% of the training set and over 99% of the validation set (incorrectly defined set).

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/7f1b8bb1-ccfc-4029-a6b9-de96ee23d3d6)

Checking the color distribution in the images suggests a normal distribution. This is confirmed by the Kolmogorov-Smirnov normalization test, which is applied in numerous cases such as this where the p-value is 0. The second image (below) suggests that the data is significantly less normal, but the same test also suggests that the p-value is 0.

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/644ccd8d-2f22-40bc-aac9-8aa08ebcdbf3)

The significance threshold for p is 0.05, which suggests normality.
Class distribution:

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/77353047-145d-42a0-87ac-d45bcceb702e)
![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/4f4db491-e53c-416b-816f-d5458f2e54c5)

Unfortunately, these sets are unbalanced, which is also evident in the histograms. The validation set lacks many classes that are present in the training set.

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/0eeea858-847b-4ebf-a621-6eff661c167b)

The above chart shows the class content in both sets. We interpret it as follows: 100% of the "human hauler" class instances are in the training set, which automatically means 0% in the test set.

There are many unique image height and width values, but the channel value is one: 3 (RGB). The only data we have at the moment is image data, which is treated quantitatively. Reviewing these images, it can be seen that they are probably all from one country, presumably Asian.

The color distribution in the image is balanced and very close to normal distribution as shown by the Shapiro-Wilk test (p-value 10^(-35)) and the Kolmogorov-Smirnov test, so normalization is not necessary.

The problematic issue is the unbalanced classes. Cars are by far the most common, followed by rickshaws, buses, and tricycles (motorized rickshaws). Then gradually, the classes become less significant.

The four largest classes – cars, buses, and rickshaws (including motorized ones) – occupy over 625% of the set.
The nine largest classes account for over 90% of all classes. At later stages, this fact will need to be taken into account.

## Data Processing
The amount of missing records and the way to solve this problem, normalization/standardization of input data, transforms performed on the data (depending on the problem, e.g., for time series, the average of the last 10 records, for image recognition, processing through graphic filters). Method of dividing into training and testing set.

Primarily, it is necessary to resample the sets – training, validation, and add the final test set. The convention suggests not combining the test and validation sets.
The new sets were determined by random selection and are as follows:

Training – 2102 images 70%
Validation – 301 images 10%
Test – 601 images 20%

A brief summary will be replaced by a histogram checking the percentage share in the data sets:

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/2a95caf8-a9ed-43bb-a10b-d0df33834290)

This share is similar and is much better than before.

The current dataset also includes a data_1.yaml file, which is a configuration file created for the YOLO model. It will later be used for classification with slight modification.

## Description of Applied Neural Networks

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/226c494f-11e6-484c-a65a-28fd2f386b7c)

The YOLOv1 architecture initially used only convolutional networks, while YOLOv8 has a much more complex structure, as shown in the diagram below:

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/99232089-5c67-41f7-be32-eaa1f782018b)

YOLOv8 strongly resembles YOLOv5, which was also created by the company Ultralytics, but there is no publicly available original scientific publication.

The YOLOv8 architecture is based on a typical object detection model structure consisting of three main components:

1. **Backbone**  
  This initial stage is responsible for extracting features from the input image. YOLOv8 uses a convolutional neural network (CNN) for this purpose, similar to many other object detection models.

2. **Neck**
  The neck section refines and combines the extracted features from the backbone. In this stage, YOLOv8 uses a combination of two techniques:
  - **Feature Pyramid Network (FPN)**: This module gradually reduces the spatial resolution of the image while increasing the number of feature channels. It creates feature maps suitable for detecting objects at various scales and resolutions.
  - **Path Aggregation Network (PAN)**: This module enhances feature representation by aggregating features from different levels of the backbone using skip connections. This allows the network to effectively capture features at multiple scales, which is crucial for accurately detecting objects of various sizes and shapes.
3. **Head**
  The final stage, the head, takes the processed features from the neck and predicts bounding boxes, object confidence scores, and class probabilities for objects present in the image. Unlike YOLOv5, which uses three output heads, YOLOv8 has one output head responsible for all these predictions. It is worth noting that YOLOv8 uses an anchor-free detection mechanism. This means it directly predicts the center of the object instead of relying on predefined anchors, leading to faster processing during post-processing.

In summary, the YOLOv8 architecture uses a powerful combination of FPN and PAN in the neck for better object detection at multiple scales. Additionally, the use of a single output head and an anchor-free detection mechanism streamlines the prediction process.

In the following considerations, we used different versions of the YOLOv8 and YOLOv5 networks:
- pretrained (we have available pre-trained weights)
- yaml version, i.e., "from scratch" (we do not have pre-prepared weights, they are initialized randomly).

In each case, due to hardware limitations, we used 40 or 35 epochs and the default batch size of 64 images. Reducing this parameter also probably allows increasing accuracy but significantly lengthens the training process. YOLO also automatically performs both resizing images (default to 640x640) and augmentation.

## Results discussion and Conclusions

The following metrics and confusion matrix are returned by YOLO. The first 3 charts on the left are the cost over time, respectively box_loss, cls_loss, and dfl_loss. Box_loss corresponds to the accuracy of the bounding box, cls_loss corresponds to the accuracy of image classification, and dfl_loss reduces the cost associated with class imbalance. The last ones are metrics: Precision indicates how many classes were correctly classified as correct, recall indicates the fraction of classes that were correctly classified out of all. mAP50 (mean average precision) indicates IoU (Intersection over Union) for a threshold of 0.50. This metric corresponds to the accuracy for "simple" classifications. mAP50-90 indicates the same, only the accuracy threshold is between 0.5 and 0.9. This last metric is a good view of how the model handles different detection difficulties.

### Results for YOLOv8n in the pretrained version:

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/d90ff601-f9d4-4d8a-a141-5de8f2408f73)

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/9c63e5d7-98e9-4eef-890b-28a2b7fbbe37)

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/43ac7fbe-0efa-49e9-87c4-c695192f05d3)
![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/d37cb9b0-ab09-48ab-98c4-9fc1fe31dc88)

The above results suggest that the network is still undertrained, but due to time constraints, we decided that it would be most reasonable to compare all models with the same hyperparameters. Ultimately, the results are going in the right direction at the level (mAP50 equal to 0.35 and mAP50-90 around 0.25).

### Results for YOLOv8n from scratch:

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/6f6b8152-e304-4ffe-a49d-1498986a1696)

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/32959b68-a5a5-44e1-90c0-a6a19abb13ea)

It is clearly visible that the model without prior weights performs significantly worse (at the level of 0.15 mAP50 and 0.08 mAP50-90), but the metrics are still increasing, suggesting that the network is even more undertrained than the previous one. For comparison, we will also check how the previous version (YOLOv5n) performs under the same conditions.

### Results for YOLOv5n in the pretrained version:

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/0f4392e3-1dd9-4a9b-9ae2-6da554ce148d)

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/c9d3d0a9-b79f-4be0-ad87-d68f8a9f18cb)

These results are similar to those for the YOLOv8n.pt version, but it is evident that the model is better at learning the detection of the rectangle itself, which can be seen in the mAP50 and mAP50-90 metrics. The newer YOLO version in the nano version slowed down in these metrics, which may cause difficulty in later training, while the 5 nano version should easily increase the accuracy of covering the object with a rectangle.

### Results for YOLOv5n from scratch:

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/f1246be1-d694-45aa-8c81-57fe1f402967)

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/c528f908-f3ec-4612-90b6-735709e1daa2)

Again, as with YOLOv8n from scratch, the performance here is significantly worse. However, the newer YOLO version proves to be slightly more effective after 35 epochs.

### Results for YOLOv8s pretrained:

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/f53ae635-e1fc-41df-83ff-18a4e1315d43)

![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/3e9f9959-4d34-41a5-835f-110cd0145f1b)

The YOLO v8 small architecture was used here (more complex than the nano version). Although it clearly gives significantly better results, it is evident that the network is significantly undertrained. The mAP50 metric equals 0.60, which is probably acceptable for practical use. The last test was to examine the small architecture without pre-prepared weights.

### Results for comparison for YOLOv8s:

**Actual**  
![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/1fea0115-d069-41e4-be36-0d07c9abde04)

**Predicted**  
![image](https://github.com/michaljurzak1/Vehicle-Images-Detection/assets/65761848/3cd946bb-89cb-46f6-8263-8327c43dc412)

## Final conclusions:

- The selection of 35 epochs was probably insufficient to obtain correct results for practical use. However, it allowed a comparative analysis of various models.
- The selection of the nano model for practical use is inappropriate due to the difficulty of appropriate learning of the detection model. However, for comparison purposes, it was the optimal choice.
- The YOLOv8 version works better than the YOLOv5 version.
- Comparing the nano and small versions of the models shows that the newer version, despite being in an early development stage, is more effective, suggesting it is a better choice for long-term investment in object detection models.
- Obtaining pre-trained weights significantly improves the results, especially for short training sessions.
