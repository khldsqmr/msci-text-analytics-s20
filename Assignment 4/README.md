Report your classification accuracy results in a table with three different activation functions in the hidden layer(ReLU, sigmoid and tanh). What effect do activation functions have on your results? What effect does addition of L2-norm regularization have on the results? What effect does dropout have on the results?

---------------------------------------------------------------------------------------------------------------------------------------------------------------

<table>
<thead><tr><th>Activation Function</th><th>Train Accuracy</th><th>Test Accuracy</th><th>Train Accuracy</th><th>Test Accuracy</th></tr></thead><tbody>
 <tr><td>&nbsp;</td><td>(With L2-Regularization)</td><td>(With L2-Regularization)</td><td>(Without L2-Regularization)</td><td>(Without L2-Regularization)</td></tr>
 <tr><td><b>ReLU</b></td><td>81.66%</td><td>80.71%</td><td>85.84%</td><td>81.42%</td></tr>
 <tr><td><b>Sigmoid</b></td><td>81.27%</td><td>80.74%</td><td>86.03%</td><td>81.66%</td></tr>
 <tr><td><b>Tanh</b></td><td>81.84%</td><td>80.54%</td><td>84.35%</td><td>81.15%</td></tr>
</tbody></table>


Based on the above table, our model performs well with Sigmoid activation function with an accuracy of 80.74% on the test data. Although ReLU (Accuracy of 80.71% on test data) and Tanh (Accuracy of 80.54% on test data) also provide non-linear activation functions and perform well on sparse variables, Sigmoid function works well with simple model having binary categorical values. In our case, we have to predict reviews based on binary categorical variable y, where 0 represents negative review and 1 represents positive review. 

L2-norm regularization helps in reducing overfitting of the model on the training data. Basically its a penalty parameter/regulator which optimzies the weights in the neural network. As we observe in our model, train and test accuracies with L2-norm regularization is less than that of without L2-norm regularization, but there's a large difference between train and test accuracies in the model without L2-norm regularization. Thus, it can be inferred that the model performs well on the training data, but not on the test data. On the contrary, there's not much difference between train and test accuracies in the model with L2-norm regularization, and thus, the model performs comparatively well on the test data.  

In our model, for the value of Dropout at 0.3, the accuracy on test data is improved as compared to dropout values of 0.1, 0.5 and 0.7 where the accuracy tends to reduce. This is because the Dropout layer randomly chooses some weights to drop them out. Thus, the neurons dont become accustomed to the neighbouring neurons for learning. There is less dependency between subsequent neurons and they learn by themselves to combat the cases where the pairing is not always together. Thus, the neurons perform well when they encounter new data and thus, the model performs well without overfitting.

