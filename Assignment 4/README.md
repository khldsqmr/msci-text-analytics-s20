Report your classification accuracy results in a table with three different activation functions in the hidden layer(ReLU, sigmoid and tanh). What effect do activation functions have on your results? What effect does addition of L2-norm regularization have on the results? What effect does dropout have on the results?

simple model with binary cateforical values
our model is simple 
Help reduce overfitting
add penalty to network 

It randomly chooses some weights to drop them out. It doesn't become accustomed to neighbouring neurons for learning. There is less dependency and learn by itself. Pairing not always togerther. perform well when encountering new data. And try to performs well without overfitting

Activation functions:

ReLU, Accuracy = 
Sigmoid, Accuracy = 80.74%
Tanh, Accuracy = 

Using different non-linear activation functions (ReLU, Sigmoid and tanh), we observe 

  <table>
    <thead>
      <tr>
        <th>Activation</th>
	<th>Accuracy</th>
      </tr>
    </thead>
    <tbody>
        <tr>
            <td>ReLU</td>
            <td>80.71%</td>
        </tr>
        <tr>
            <td>Sigmoid</td>
            <td>80.75%</td>
        </tr>
	<tr>
            <td>Tanh</td>
            <td>80.54%</td>
        </tr>
    </tbody>
  </table>


<table>
<thead><tr><th>Activation Function</th><th>Accuracy</th><th>&nbsp;</th><th>&nbsp;</th><th>&nbsp;</th></tr></thead><tbody>
 <tr><td>&nbsp;</td><td>With L2-Regularizer</td><td>&nbsp;</td><td>Without L2-Regularizer</td><td>&nbsp;</td></tr>
 <tr><td>&nbsp;</td><td>Train </td><td>Test</td><td>Train</td><td>Test</td></tr>
 <tr><td>ReLU</td><td>81.66%</td><td>80.71%</td><td>&nbsp;</td><td>&nbsp;</td></tr>
 <tr><td>Sigmoid</td><td>81.27%</td><td>80.74%</td><td>&nbsp;</td><td>&nbsp;</td></tr>
 <tr><td>Tanh</td><td>81.84%</td><td>80.54%</td><td>&nbsp;</td><td></td></tr>
</tbody></table>
