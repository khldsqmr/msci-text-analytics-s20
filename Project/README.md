A conversational agent (chatbot) is a machine-learned dialogue model designed to simulate human conversations. In this study we will start by understanding what neural network is and how Recurrent Neural Network (RNN) and Sequence-to-Sequence (Seq2Seq) model are related to it. Then, we will talk about ways through which RNN and Seq2Seq can be improved such as by adding LSTMs and Attention mechanism. Finally, we will talk about a more developed model such as the Transformer model for chatbots. In this project, we will use the Cornell Movie-Dialogs Corpus to train our models. We will build Seq2Seq LSTM model, Seq2Seq LSTM model with Attention mechanism, and the Transformer model. Finally, we aim at comparing the performance of these dialogue models using the Bleu score and human judgement metrics to determine which model would be a better suggestion.

On the same set of sample of 100 data: 
For Seq2Seq model, Bleu score of 0.4368 was observed on train data, and 0.0561 on test data.
For Seq2Seq model with Attention mechanism, Bleu score of 0.2626 was observed on train data, and 0.0415 on test data.
For Seq2Seq Transformer model, Bleu score of 0.5296 was observed on train data, and 0.2790 on test data.

On a set of 100 sampled data, following Bleu Scores were observed for different epoch values:

|       | Transformer Model | Transformer Model | Attention Mechanism | Attention Mechanism | No Attention Mechanism | No Attention Mechanism |
|-------|-------------------|-------------------|---------------------|---------------------|------------------------|------------------------|
| Epoch | Train Bleu Score  | Test Bleu Score   | Train Bleu Score    | Test Bleu Score     | Train Bleu Score       | Test Bleu Score        |
| 10    | 0.085             | 0.073             | 0.05                | 0.063               | 0.077                  | 0.085                  |
| 20    | 0.245             | 0.078             | 0.039               | 0.047               | 0.065                  | 0.07                   |
| 30    | 0.384             | 0.074             | 0.036               | 0.059               | 0.056                  | 0.066                  |
| 40    | 0.454             | 0.08              | 0.055               | 0.063               | 0.057                  | 0.061                  |
| 50    | 0.553             | 0.074             | 0.102               | 0.047               | 0.051                  | 0.067                  |
| 60    | 0.52              | 0.076             | 0.25                | 0.047               | 0.054                  | 0.043                  |
| 70    | 0.558             | 0.082             | 0.303               | 0.049               | 0.051                  | 0.056                  |
| 80    | 0.522             | 0.082             | 0.324               | 0.052               | 0.054                  | 0.059                  |
| 90    | 0.549             | 0.07              | 0.329               | 0.052               | 0.057                  | 0.048                  |
| 100   | 0.493             | 0.081             | 0.327               | 0.042               | 0.045                  | 0.058                  |


In this study we trained and compared three different conversational dialogue generation models i.e., Seq2Seq LSTM model, Seq2Seq LSTM model with Attention mechanism, and the Transformer model, and performed model evaluation for comparison.
Based on our research, we expected the evaluation scores to improvise as more advanced models are used. However, this was not completely observed in our analysis. The Bleu scores for the Seq2Seq model with attention mechanism were approximately less than or equal to that of Seq2Seq without attention in some epochs. This can be explained by the fact that we trained the Seq2Seq model having attention with reduced vocabulary due to limited computational power. This is because we wanted to train all the three models on same set of data. As a result, the missing vocabularies were unknown to the model. This could be improved by setting the right hyper parameters for model optimization and training the model with increased vocabulary over more epochs. 
The responses and Bleu scores of the Transformer model, on the other hand, were significantly higher than the results of other two models. Conclusively, this model can improve conversational chatbot performance despite the restrictions.
