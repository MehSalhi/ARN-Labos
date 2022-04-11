---
title: "ARN - Laboratory 03"

author: 
- Anthony Coke
- Guilain Mbayo
- Mehdi Salhi
date : \today
titlepage: true
logo: figures/logo.png
toc: true
toc-own-page: true
...
 
# Introduction

# Man vs Woman
The purpose of this experiment was to train our model to be able to
differentiate between men and women.

## Number of observations
Our dataset consist of audio samples of vowels pronounced by men and women.
For this part, we have treated 36 values for each class (men and women).
We then computed 13 MFCCs for each sample.

![Boxplot of the MFCCs for men and women](figures/ARN-L3-MFCC-Men-Women.png)

## Features to train the model
We chose to use the tanh function for this part, as it was easy for two classes to put the value 1 for the first and -1 for the second. 
As tanh output value between minus one and plus one, we can get better learning curves than with sigmoïdal, which give an output between zero and one.

## Procedure for model selection
The very first step after getting the dataset is to normalize and label it.
Then, in order to choose the best parameters for our model, we tried at first the basis value of 0.001 for the learning rate, 0.5 for the momentum and 50 epochs. 
After observing the results, we adjusted our parameters. Those steps where repeated several times in order to narrow our results.
When we obtained a satisfying curve for the training and test sets, we generated the confusion matrix to verify that our datas were indeed well classified.

![Exploring the number of neurons](figures/ARN-L3-ExploringNeuron-Men-Women.png)

## Description of the final model and performance evaluation
Our final model used the tanh validation function, used a learning rate of 0.0009, a momentum of 0.9, 2 hidden neurons, one output neuron, an epoch number of 100 and a threshold at 0.0.
We got a MSE training of 0.059 and an MSE test of 0.178. 
Our confusion matrix was 

[34. 2.] 

[3. 33.]

We measured the performances of our model by using a 5-fold cross-validation.

![Final Model Evaluation](figures/ARN-L3-FinalModel-Men-Women.png){width=50%}

## Comments
We had a problem with data normalizations. At first, we normalized female and male dataset
separately, which produced a curious error. We needed to give output value between 0 and 1
instead of -1 and 1 for the tanh validation function in order to get acceptables MSE curves 
for both training and test sets. This problem has been fixed by merging both dataset before 
the normalization.

# Man vs Woman vs Children

The purpose of this experiment was to train our model to be able to
differentiate between men, women and kids.

## Number of observations
The dataset was composed of 180 values of 13 mfccs each. This represents all the 
male, female and kids voices. 

![MFCCs Men Women Kids](figures/ARN-L3-MFCC-Men-Women-Kids.png){width=80%}

## Features to train the model
This part required a different approach than the first one, as our goal was to classify
the data into three classes instead of two. We labeled those data with three distinct 
column taking the values (1,-1,-1), (-1,1,-1) or (-1,-1,1). With this, we could use the
activation function tanh in order to train and test our dataset.

![Exploring Number of Neurons](figures/ARN-L3-ExploringNeurons-Men-Women-Kids.png){width=80%}

## Procedure for model selection
The procedure that we used to select the model was the same as for the first part, except that we specified the last three column as classes labels to the "fit" function. 

![Final Model Test](figures/ARN-L3-FinalModel-Men-Women.png){width=80%}

## Description of the final model and Performance evaluation 

# Final experiment

## Number of observations
Our dataset was composed of 360 values of 13 mfccs each. We used all the natural voices 
values as well as all the synthetic voices. Our objective for this experiment was to 
classify values as either human or synthetic.

![MFCCs Men Women Kids Synthetic](figures/ARN-L3-MFCC-Men-Women-Kids-Synth.png){width=80%}

## Features to train the model
We chose again to use the tanh function for this part, as it was easy for two classes to put the value one for the first and minus one for the second.
As tanh output value between minus one and plus one, we can get better learning curves than with sigmoïdal, which give an output between zero and one.

## Procedure for model selection
As our goal was to separate two classes (synthetic or human), we chose to use the same
method as for the first part. 
Of course, the exploration of hyper-parameters was different as the dataset was bigger and 
composed of different values.

## Description of the final model
The final model is similar to the first one (man and woman only)

![Final Model Men Women Kids Synthetic](figures/ARN-L3-FinalModel-Men-Women-Kids-Synth.png){width=80%}

## Performance evaluation 
We came out with the following values for the evaluation of our final model:

- MSE training:  0.12012370859086838
- MSE test:  0.15383503249177605
- Confusion matrix:

    [176.   4.]

    [ 12. 168.]

Those results look pretty good, even if there is some little error, especially in the second class.

## Comments
This part was the easiest as we widely took advantage of our past experiences with the 
two first parts of this lab.
