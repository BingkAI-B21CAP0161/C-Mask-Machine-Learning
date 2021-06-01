# C-Mask Machine Learning
Machine Learning research Jupyter Notebooks and saved models for C-Mask app.

For Back-End project using the trained models, see [here for classification](https://github.com/BingkAI-B21CAP0161/Back-End-CMask) and [here for object detection](https://github.com/BingkAI-B21CAP0161/Back-End-ObjectDetection).

For Front-End project of C-Mask, see [here](https://github.com/BingkAI-B21CAP0161/CMask).

## Overview
C-Mask is an app that can detect whether people are wearing masks correctly. There are two measures, first by [single face classification](./Masked_Face_Classification) and another one by [faces object detection](./Masked_Face_Object_Detection). App is created for Android devices, and the inferences are done in cloud back-end.

Face classification is done by using custom `CNN` model. The output is binary label prediction. Faces object detection is done by using `Mask R-CNN` architecture. The output of inferences is a list of coordinates for each bounding box and labels for each bounding box. Researches are executed using `Jupyter Notebooks` and `Google Colaboratory`.

## Project Stucture
We separate the directories for __Classification__ and __Object Detection__ researches. For each directory, there would be `Preprocessing`, `Modelling`, and `Saved_Models` subdirectories.
1. `Preprocessing` includes notebooks for extracting, preprocessing, and arranging datasets.
1. `Modelling` includes notebooks for building, training, and evaluating machine learning models.
1. `Saved_Models` includes saved model files to be used in back-end app.

## Research History
### Dataset History
For __classification__, we first used [MaskedFace-Net dataset](https://github.com/cabani/MaskedFace-Net) which consists of 2 classes, `CMFD` (Correctly Masked) and `IMFD` (Incorrectly Masked). The images are faces of people patched with generated medical masks, __not real masks__. The IMFD images can be further splited into three classes, `uncovered chin`, `uncovered nose`, and `uncovered nose and chin`. We also combine it with [Flickr Faces HQ dataset](https://github.com/NVlabs/ffhq-dataset) for `no mask` faces. Thus we had 5 labels for this combined dataset which can be found [here](https://drive.google.com/drive/folders/1e6dsErnWnZ-ZMsk5HqGw6PUS-YgaUZXZ?usp=sharing).

After facing overfitting, due to the dataset only contains single variant of mask and all of them are generated by computer, we switch to new dataset. [Face mask dataset by omkargurav](https://www.kaggle.com/omkargurav/face-mask-dataset) contains more diverse masks and also different image shapes. This dataset improves prediction on new images, especially with diverse face masks. We further tried [the third dataset by prithwirajmitra](https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset) and combine it with the second dataset. The combined dataset can be found [here](https://drive.google.com/drive/folders/1NvGlWbR7O0nZnI1CJXWcHZ3b_P3YExXj?usp=sharing).

For __object detection__, we use [face mask detection dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) with XML annotations for the bounding boxes. This dataset has 3 labels, `with mask`, `incorrectly worn mask`, and `without mask`.

### Model History
For __classification__, we first tried `transfer learning` using MobileNet. We tried `MobileNetV2` and `MobileNetV3Large` with various scenarios. All of them overfits on the training and fails to predict new images correctly. The most frequent problem we found is the training and validation `accuracy` and `loss` are high on training, but when we evaluate afterward the model predicts all images as the same label, even on training and validation datasets.

We then tried creating custom `CNN` model in hope having a better performance. It did much better than previous MobileNets, as it didn't fall into overfitting. But as the first dataset only contains images with uniform generated masks, it failed to predict random mask faces. When we use 2nd and 3rd datasets, we gain improvement on predictions on new images.

For __object detection__, we tried `transfer learning` using [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN) and use [face mask detection dataset](https://www.kaggle.com/andrewmvd/face-mask-detection). This `Mask R-CNN` architecture library already provides functions to create `Dataset` object, create `Config` for the model, as well as training and evaluating the model. We refer [a tutorial by Jason Brownlee](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/) for learning to use the library.

### Latest Datasets and Models used
The latest model used for C-Mask __classification__ is`model_cnn_scenario_6`, trained using combined 2nd and 3rd datasets on custom `CNN` model for 50 epochs. The latest model used for C-Mask __object detection__ is `face_mask_detection_config_50_epoch_128_steps`, trained using [face mask detection dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) on `Mask R-CNN` model for 50 epochs.

## Getting Started
### Prerequisites
We need to have `Python 3` and `Jupyter` on the environment where you'd like to run these notebooks. We used `Google Colaboratory` for researching on all these notebooks. We also need to install external packages that are imported inside each notebook, including, but not limited to `tensorflow`, `numpy`, `matplotlib`, and others.

### Run notebooks
Run the notebook on prepared environment. Each notebook can be run independently.

## Notes
- It is still unknown why MobileNets yield good results on training, but fail on evaluation, and even fail on predicting training set.
- Per May 26th 2021, the object detection notebook using Mask R-CNN cannot be run on `Google Colab`. It seems like there were sudden changes in some libraries on Colab. The codes work well on our back-end GCP VM, though.

## References
- Cabani, A., Hammoudi, K., Benhabiles, H., &amp; Melkemi, M. (2020). MaskedFace-Net – A dataset of correctly/incorrectly masked face images in the context of COVID-19. Smart Health, 19, 100144. https://doi.org/10.1016/j.smhl.2020.100144
- He, K., Gkioxari, G., Dollár, P., &amp; Girshick, R. (2018, January 24). Mask R-CNN. arXiv.org. https://arxiv.org/abs/1703.06870
- Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., &amp; Adam, H. (2017, April 17). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv.org. https://arxiv.org/abs/1704.04861
- Karras, T., Laine, S., &amp; Aila, T. (2019, February 6). A Style-Based Generator Architecture for Generative Adversarial Networks. arXiv.org. https://arxiv.org/abs/1812.04948
- Singh, S., Ahuja, U., Kumar, M., Kumar, K., &amp; Sachdeva, M. (2021). Face mask detection using YOLOv3 and faster R-CNN models: COVID-19 environment. Multimedia Tools and Applications, 80(13), 19753–19768. https://doi.org/10.1007/s11042-021-10711-8

- [https://d2l.ai/chapter_computer-vision/rcnn.html](https://d2l.ai/chapter_computer-vision/rcnn.html)
- [https://linkinpark213.com/2019/03/17/rcnns/](https://linkinpark213.com/2019/03/17/rcnns/)
- [https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/)
- [https://towardsdatascience.com/transfer-learning-using-mobilenet-and-keras-c75daf7ff299](https://towardsdatascience.com/transfer-learning-using-mobilenet-and-keras-c75daf7ff299)

## Licenses

## Contributors
- [Kc-codetalker](https://github.com/Kc-codetalker)
- [mohfaisal25](https://github.com/mohfaisal25)