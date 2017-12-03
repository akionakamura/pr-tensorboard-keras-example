# Extends Keras' TensorBoard callback to include the Precision-Recall summary plugin.

import os
from urllib.request import urlretrieve

import pandas as pd

from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorboard.plugins.pr_curve import summary as pr_summary


class PRTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        # One extra argument to indicate whether or not to use the PR curve summary.
        self.pr_curve = kwargs.pop('pr_curve', True)
        super(PRTensorBoard, self).__init__(*args, **kwargs)

        global tf
        import tensorflow as tf

    def set_model(self, model):
        super(PRTensorBoard, self).set_model(model)

        if self.pr_curve:
            # Get the prediction and label tensor placeholders.
            predictions = self.model._feed_outputs[0]
            labels = tf.cast(self.model._feed_targets[0], tf.bool)
            # Create the PR summary OP.
            self.pr_summary = pr_summary.op(tag='pr_curve',
                                            predictions=predictions,
                                            labels=labels,
                                            display_name='Precision-Recall Curve')

    def on_epoch_end(self, epoch, logs=None):
        super(PRTensorBoard, self).on_epoch_end(epoch, logs)

        if self.pr_curve and self.validation_data:
            # Get the tensors again.
            tensors = self.model._feed_targets + self.model._feed_outputs
            # Predict the output.
            predictions = self.model.predict(self.validation_data[:-2])
            # Build the dictionary mapping the tensor to the data.
            val_data = [self.validation_data[-2], predictions]
            feed_dict = dict(zip(tensors, val_data))
            # Run and add summary.
            result = self.sess.run([self.pr_summary], feed_dict=feed_dict)
            self.writer.add_summary(result[0], epoch)
        self.writer.flush()


def build_model(n_features):
	input_layer = Input(shape=(n_features,))
	dense = Dense(16, activation='relu')(input_layer)
	pred = Dense(1, activation='sigmoid')(dense)

	model = Model(inputs=input_layer, outputs=pred)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

if __name__ == '__main__':
	current_dir = os.path.dirname(os.path.realpath(__file__))
	data_dir = os.path.join(current_dir, 'data')
	data_file = os.path.join(data_dir, 'wdbc.csv')
	log_dir = os.path.join(current_dir, 'logs')

	print('Working directory: %s' % current_dir)
	if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
	    os.mkdir(data_dir)

	if not os.path.exists(data_file) or not os.path.isfile(data_file):
	    print('Downloading data...')
	    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
	    urlretrieve(data_url, data_file)
	    print('Data saved under: %s' % data_file)

	if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
	    os.mkdir(log_dir)

	# Load data.
	df = pd.read_csv(data_file, header=None)

	# Encode benign (0) and malignant (1)
	label_encoder = preprocessing.LabelEncoder()
	label_encoder.fit(df[1])
	labels = label_encoder.transform(df[1])

	# Normalize features.
	features = df[df.columns[2:]]
	features_normalized = preprocessing.normalize(features)

	# Split train and test.
	X_train, X_test, y_train, y_test = train_test_split(features_normalized, labels, test_size=0.3, random_state=42)

	model = build_model(features_normalized.shape[1])

	max_epochs = 200
	callbacks = [ PRTensorBoard(log_dir=log_dir), EarlyStopping(monitor='val_loss', patience=3) ]
	history = model.fit(X_train, y_train, validation_data=[X_test, y_test], epochs=max_epochs, callbacks=callbacks)
