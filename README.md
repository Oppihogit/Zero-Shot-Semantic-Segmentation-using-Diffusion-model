# Zero-Shot-Semantic-Segmentation-using-Diffusion-model
## Abstract
Our method uniquely uses diffusion models for visual-semantic data generation in unseen classes, bypassing pre-trained backbones. It directly produces segmentation masks with a basic semantic classifier and can enhance data for semantic segmentation datasets.

## Our main contributions are as follows:
1) We introduce diffusion models into the zero-shot
semantic segmentation domain for the first time,
realizing the synthesis of visual-semantic feature
data for unseen classes.
2) Our approach achieves approaching state-of-the-art
segmentation performance using a streamlined
architecture without relying on a pre-trained
backbone.
3) Based on the original architecture, we propose a
novel and efficient data augmentation method for
semantic segmentation dataset.

## METHODOLOGY
![alt text](overview.png)
Our proposed method shown in the Figure 2, does not rely on
pre-trained backbones. we stack seen class visual information with their pixel-level semantic expressions as training
data and train a DDPM model to generate corresponding
visual-semantic data. During the sampling process, we use
images with unseen classes as the condition, guiding the
trained generator to generate matching semantic embedding
data. Subsequently, we constructed a semantic classifier to
achieve pixel-level classification of unseen and seen class
objects within the image realizing semantic segmentation.
The above technical details were elaborated in the following
subsections


