import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
TRAIN_CSV_PATH = './data/BostonHousing_subset.csv'
TEST_CSV_PATH = './data/boston_test_subset.csv'
PREDICT_CSV_PATH = './data/boston_predict_subset.csv'

# target variable to predict:
LABEL_PR = "medv"

def get_batch(file_path, batch_size, num_epochs=None, **args):
    with open(file_path) as file:
        num_rows = len(file.readlines())

    dataset = tf.data.experimental.make_csv_dataset(
        file_path, batch_size, label_name=LABEL_PR, num_epochs=num_epochs, header=True, **args)

    # repeat and shuffle and batch separately instead of the previous line
    # for clarity purposes
    # dataset = dataset.repeat(num_epochs)
    # dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    elem = iterator.get_next()
    return elem

def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


# Now to define the feature columns
tax = tf.feature_column.numeric_column('tax')
indus = tf.feature_column.numeric_column('indus')
crim = tf.feature_column.numeric_column('crim')

# Building the model
model_est = tf.estimator.LinearRegressor(feature_columns=[crim, indus, tax], model_dir='model_dir')

# Train it now
model_est.train(steps=2300, input_fn=lambda: get_batch(TRAIN_CSV_PATH, batch_size=256))

results = model_est.evaluate(steps=1000, input_fn=lambda: get_batch(TEST_CSV_PATH, batch_size=128))

for key in results:
    print("   {}, was: {}".format(key, results[key]))

to_pred = {
    'crim': [0.03359, 5.09017, 0.12650, 0.05515, 8.15174, 0.24522],
    'indus': [2.95, 18.10, 5.13, 2.18, 18.10, 9.90],
    'tax': [252, 666, 284, 222, 666, 304],
}


def test_get_inp():
    dataset = tf.data.Dataset.from_tensors(to_pred)
    return dataset

# Predict
for pred_results in model_est.predict(input_fn=test_get_inp):
    print(pred_results['predictions'][0])

# Now to export as SavedModel
print(tf.feature_column.make_parse_example_spec([crim, indus, tax]))
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(tf.feature_column.make_parse_example_spec([crim, indus, tax]))


# def serving_input_fn():
#     features_placeholders = {
#         'crim': tf.placeholder(tf.float32),
#         'indus': tf.placeholder(tf.float32),
#         'tax': tf.placeholder(tf.float32)
#     }
#
#     features = dict()
#
#     for key, tensor in features_placeholders.items():
#         features[key] = tf.expand_dims(tensor, -1)  # -1 counts from backwards
#
#     return tf.estimator.export.ServingInputReceiver(features, features_placeholders)

export_path = model_est.export_saved_model("saved_model", serving_input_fn)

# See:
#https://github.com/rstudio/tfdeploy/issues/3