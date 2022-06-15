---
title: "ARN - Laboratory 05" 

author: 
- Anthony Coke
- Mehdi Salhi
- Guilain Mbayo
date : \today
titlepage: true
logo: figures/logo.png
toc: true
toc-own-page: true
...

# Introduction

Our application is made in the context of the course ARN (Apprentissage par
réseau de neurone) At the HEIG-VD.
Its goal is to classify different types of nuts.
In order to achieve this goal, we decided to take several pictures of cashew
nuts, hazelnuts, and pecans with various backgrounds, various specimens and various angles. 
Depending on the result, we will then eventually add pictures from the web.

We will use MobileNet V2, which is a pre-trained convolutionnal network. We will
then add some layers and train them to our specific application.

Our application could be used for various purpose. For example, allow users to
identify different nuts, for allergy purpose. This could also be used for
industrial application when sorting nuts or detecting allergens in food.

# The problem

The problem is to be able to identify and differentiate between 3 sorts of nuts:
pecan, hazelnuts and cashews, using a camera on a portable device.

# Data preparation

We took photos of the different nuts on various background (color, texture) with
various angles and zoom.

##TODO: data augmentation, crop, zoom, format, ...
Our whole images dataset is rescaled and resized in order to always give the
same dimensions as input for our model.
In order to have slightly different images at each iteration, we applied some
data augmentations to our training set. It include RandomFlip, RandomZoom,
RandomRotation and RandomContrast. We chose to use value between -0.2 and 0.3
and not higher because we saw that the resultant pictures were too deformed, and
thus could maybe mislead our model.

# Model creation

##TODO: parameters, ...



# Results

##TODO: screenshots, confusion matrix, etc


# Conclusion

We tried several configurations for our model (1 layer, 2 layers, 3 layers, 256
neurons, 8 neurons, dropout, etc...) and most of the time, we managed to get
high accuracies and fine confusion matrix. But even so, we had problems once our
model loaded on the app. Indeed, the app detected each classes with great
confidence, but not for the right nut. 
We believe that our model is too influenced by the background due to a to
different images set between our 3 classes.
TODO: ARN IS FUN
