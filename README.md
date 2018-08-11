# Artificial-Intelligence-FOMC-Monetary-Policy-Analysis-for-EUR-USD-version-2

This is the second version of a project I've decided to do in Artificial Intelligence and Deep Learning. I've started getting quite a passion for the subject and its potential applications. This uses natural language processing (NLP) for binary classification from a multi-layer perceptron (MLP) i.e. standard neural network. It attempts to predict the EUR/USD appreciation/depreciation from the Fed's Monetary Policy Report release date up until the next report release to Congress.

With this second version of my project, I again set the optimizer to Adam. I've read some research that suggests at this point in time it is the best optimizer in dealing with gradient descent as it realizes both benefits of AdaGrad and RMSProp. I then proceeded to incorporate the Hyperas library to assist in optimizing my hyper-parameters. Those parameters chosen for optimization are listed below:

Best performing model chosen hyper-parameters:
{'Dense': 64, 'Dense_1': 16, 'Dropout_1': 0.3838088604298333, 'Dense_2': 8, 'Dropout': 0.9128294469805703}

With those hyper-parameters, I was able to get a 60% level of accuracy which does make it statistically significant but not ultimately ideal....in my opinion.

I also did some research with recurrent neural networks such as LSTM and GRU. This did achieve a similar level of accuracy, but they seemed to be more resource intensive than the standard. I also did some research and testing with a 1D convolutional neural network to see what output I could achieve. This did not lead me to higher levels of accuracy, but was certainly less resource intensive in general.
