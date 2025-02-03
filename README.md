# Convolution Neural Network

CNN is used for computer vision. In this case the network will determine what is on a 150x150 image. There will be a total of 6 classes (buildings, forest, glacier, mountain, sea and street). The dataset is available on https://www.kaggle.com/datasets/puneet6060/intel-image-classification.

## Topology

Topology of a CNN can be separated into 2 parts. First one is a convolution part with convolution layers and after that there is a classification part with fully connected neural network.

### Convolution layers

Convolution is a linear operation which uses a kernel. This kernel is applied onto the input image resulting into a feature map. The kernel is moving pixel by pixel (if the step is set to 1) and its new value is calculated with affection on its neighbors. This way is kernel applied onto the whole image resulting into new one. The size of the new image depends on several factors. These are size of the kernel, moving step and if there was added padding. Generally speaking the output size will be smaller or the same, but cannot be the same without padding.

The purpose of these kernels is to capture some patterns in the image. Lets say we have a kernel that is designed to capture horizontal lines. In the output image the places where are horizontal lines will be most active. Each convolution layer uses multiple kernels which results into multiple feature maps. These maps are input for the following layer, which finds patterns in the already existing ones (it is looking for more complex shapes). The feature maps from the last convolution layer serve as input into fully connected neural network.

What technique is also typically used in CNN is max pooling. This operation is applied on a feature map before passing it to the next layer. What max pooling does is simple. It has specified size of pixels and from this area it takes the maximum values. The main benefit of this techniques is that it reduces the size of the feature maps. Smaller features maps means more effective calculation with them.

### Fully connected neural network

The goal of this network is to make a classification based on taken feature maps. The network is trained to recognize classes (objects) based on their features. Type of this network could be the typical supervised model with error back propagation learning algorithm.

## Implementation

accuracy: 85.4%

## Notes

- With ImageFolder, labels are assigned based on the subfolder from which the image was 
- Softmax is applied inside the CrossEntropyLoss