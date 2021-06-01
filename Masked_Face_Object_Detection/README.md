# Faces Object Detection

## Overview
Latest version of our faces object detection is done by using `Mask R-CNN` architecture. `Mask R-CNN` is one of the latest addition to the `R-CNN` family, extending `Faster R-CNN`, one of the __fastest__ R-CNN models. The output of inferences is a list of coordinates for each bounding box and labels for each bounding box. It receives `RGB` (3 channels) images and predict the labels and coordinates.

## Directories Stucture
There are `Modelling`, and `Saved_Models` subdirectories.
1. `Modelling` includes notebooks for building, training, and evaluating machine learning models.
    1. [`Example_Mask_R-CNN_Object_Detection_Kangaroo.ipynb`](./Modelling/Example_Mask_R-CNN_Object_Detection_Kangaroo.ipynb) includes codes from [a tutorial by Jason Brownlee](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/) to try object detection using [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN). We used [Kangaroo dataset for object detection](https://github.com/experiencor/kangaroo) here as used in aforementioned tutorial.
    1. [`Mask_R_CNN_Object_Detection.ipynb`](./Modelling/Mask_R_CNN_Object_Detection.ipynb) includes codes from previous notebook, but with modifications for [face mask detection dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) and some refactoring to achieve `clean code`.
1. `Saved_Models` includes saved model files to be used in back-end app.

## Research History
### Dataset History
We use [face mask detection dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) with XML annotations for the bounding boxes. This dataset has 3 labels, `with mask`, `incorrectly worn mask`, and `without mask`.

### Model History
We tried `transfer learning` using [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN) and use [face mask detection dataset](https://www.kaggle.com/andrewmvd/face-mask-detection). This `Mask R-CNN` architecture library already provides functions to create `Dataset` object, create `Config` for the model, as well as training and evaluating the model. We refer [a tutorial by Jason Brownlee](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/) for learning to use the library.

### Latest Datasets and Models used
The latest model used is `face_mask_detection_config_50_epoch_128_steps`, trained using [face mask detection dataset](https://www.kaggle.com/andrewmvd/face-mask-detection) on `Mask R-CNN` model for 50 epochs.

## Flow of Program
Training
1. Create `Dataset` class.
1. Open image dataset as `Dataset` object.
1. Create the model and load pre-trained weights (`transfer learning`).
1. Train the model using the `Dataset` object.
1. Evaluate the model using library's built-in function.

Inference
1. Open test image as `NumPy` array.
1. Load the model (if not created yet, or if not continuing directly from training).
1. Execute detect on the model using the test image.
1. Post-process the bounding boxes coordinates and labels.

## Getting Started
### Prerequisites
We need to have `Python 3` and `Jupyter` on the environment where you'd like to run these notebooks. We used `Google Colaboratory` for researching on all these notebooks. We also need to install external packages that are imported inside each notebook, including, but not limited to `tensorflow`, `numpy`, `matplotlib`, and `Mask R-CNN by Matterport`.

### Run notebooks
Run the notebook on prepared environment. Each notebook can be run independently.

## Notes
- Per May 26th 2021, the object detection notebook using Mask R-CNN cannot be run on `Google Colab`. It seems like there were sudden changes in some libraries on Colab. The codes work well on our back-end GCP VM, though.

## References
- He, K., Gkioxari, G., Dollár, P., &amp; Girshick, R. (2018, January 24). Mask R-CNN. arXiv.org. https://arxiv.org/abs/1703.06870
- Singh, S., Ahuja, U., Kumar, M., Kumar, K., &amp; Sachdeva, M. (2021). Face mask detection using YOLOv3 and faster R-CNN models: COVID-19 environment. Multimedia Tools and Applications, 80(13), 19753–19768. https://doi.org/10.1007/s11042-021-10711-8

- [https://d2l.ai/chapter_computer-vision/rcnn.html](https://d2l.ai/chapter_computer-vision/rcnn.html)
- [https://linkinpark213.com/2019/03/17/rcnns/](https://linkinpark213.com/2019/03/17/rcnns/)
- [https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/)

## Licenses

## Contributors
- [Kc-codetalker](https://github.com/Kc-codetalker)