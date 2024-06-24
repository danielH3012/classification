#classification
from __future__ import absolute_import, division,print_function,unicode_literals
import pandas as pd
import tensorflow as tf

CSV_COLUMN_NAMES = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
SPECIES = ['Setosa','Versicolor','Virginica']
train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path,names = CSV_COLUMN_NAMES,header =0)
test = pd.read_csv(test_path,names = CSV_COLUMN_NAMES,header =0)

#pop species
train_y = train.pop('Species')
test_y = test.pop('Species')

#input function
def input_fn(features, labels, training = True, batch_size = 256):
  #convert input ke dataset
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

  if training:
    dataset = dataset.shuffle(1000).repeat()

  return dataset.batch(batch_size)

#feature column menjelaskan cara menggunakan input
my_feature_columns = []
for key in train.keys():
  my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#bikin model DNN dn 2 hidden layer dgn 30 dan 10 hidden nodes
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],
    n_classes=3)

#train model
classifier.train(
    input_fn =lambda: input_fn(train, train_y, training = True),
    steps=50000
    )


eval_result = classifier.evaluate(input_fn =lambda: input_fn(test, test_y, training = False))

print('test accuracy: {accuracy:0.3f}\n'.format(**eval_result))

#prediction
def input_fn(features, batch_size=256):
  #convert input ke dataset tanpa label
  return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength','SepalWidth','PetalLength','PetalWidth']
predict = {}

print("tulis numeric value seperti prompt (isi float ya kontol!!!)")
for feature in features:
  valid = True
  while valid:
    val = input(feature + " :")
    if not val.isdigit(): valid = False
  predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))

for pred_dict in predictions:
  class_id = pred_dict['class_ids'][0]
  probability = pred_dict['probabilities'][class_id]

  print('prediksi: "{}" ({:.1f}%)'.format(SPECIES[class_id], 100* probability))
