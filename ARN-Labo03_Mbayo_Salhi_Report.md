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

# Report
## Man vs Woman

### Number of observations
For this part, we have treated 36 values for each classes (men and women).
Each of this values are separated into 13 mfccs.

### Features to train the model
We chose to use the tanh function for this part, as it was easy for two classes to put the value one for the first and minus one for the second. 
As tanh output value between minus one and plus one, we can get better learning curves than with sigmoïdal, which give an output between zero and one.

### Procedure for model selection
The very first step after getting the dataset is to normalize and labelize it.
Then, in order to choose the best parameters for our model, we tried at first the basis value of 0.001 for the learning rate, 0.5 for the momentum and 50 epochs. 
After observing the results, we adjusted our parameters. Those steps where repeated several times in order to narrow our results.
When we obtained a satisfying curve for the training and test sets, we generated the confusion matrix to verify that our datas were indeed well classified.

### Description of the final model
Our final model used the tanh validation function, used a learning rate of 0.0009, a momentum of 0.9, 2 hidden neurons, one output neuron, an epoch number of 100 and a treshold at 0.0.
We got a MSE training of 0.059 and an MSE test of 0.178. Our confusion matrix was [[34. 2.] [3. 33.]]


### Performance evaluation 


- Comments

## Man vs Woman vs Children

- Number of observations
- Features to train the model
- Procedure for model selection
- Description of the final model
- Performance evaluation 

## Final experiment

- Number of observations
- Features to train the model
- Procedure for model selection
- Description of the final model
- Performance evaluation 
- Comments
