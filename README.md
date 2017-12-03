## Precision-Recall curve with Keras

A blog post describing the work here can be found on my [Medium profile](https://medium.com/@akionakas/precision-recall-curve-with-keras-cd92647685e1).

[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) is a suite of visualizations for inspecting and understanding your TensorFlow models and runs. They recently [released](https://research.googleblog.com/2017/09/build-your-own-machine-learning.html) of a "consistent set of APIs that allows developers to add custom visualization plugins to TensorBoard". There are already [several plugins](https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins) available.

[Keras](https://keras.io/) "is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano". When using the TensorFlow backend, they typically support the [TensorBoard callback](https://keras.io/callbacks/#tensorboard), to take advantage for its visualizations.

Keras' TensorBoard callback, however, still do not support the plugins. I recently wanted to use the [Precision-Recall curve](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) [plugin](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/pr_curve/summary.py) (pr_curve) to see how my binary classification problem was doing. I ended up writing an extension of the callback supporting it. Although it is only a partial support (lacks usage of weights, for example), hopefully this will help anyone else in need of similar code, since I've found very little material about it around the web.

### Run the Example:
Assuming you have all the dependecies installed, run:

	python3 example.py
	tensorboard --logdir=./logs

The script will download a small dataset to run the example on real data. The data is from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), specifically the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).
