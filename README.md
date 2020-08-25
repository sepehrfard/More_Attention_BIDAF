# More_Attention_BIDAF
## Bidrectional Attention Flow with more attention on context comprehension
This model was the recreation of original paper [All you need is Attention](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) with improved performance using AdaBoost activation function and scheduled learning rate. Using [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset, model was utilized for Question Answering application. SQuAD 2.0 consited of answerable questions, and non answerable questions to benchmark the model on true undrestanding of given context. Non answerable questions perfomance is used to show weather model is simply returning probablity of some sentence existing in context and question or weather it is able to check weather question is unrelated to context and does not offer viable answer to the question using given context. 
### Model Architecture: 
![Image of Model](https://github.com/sepehrfard/More_Attention_BIDAF/blob/master/images/Model.png)
### Use Case
![Image of Use Case](https://github.com/sepehrfard/More_Attention_BIDAF/blob/master/images/usecase.png)