import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TRAIN_CSV_PATH = './data/BostonHousing_subset.csv'
TEST_CSV_PATH = './data/boston_test_subset.csv'
PREDICT_CSV_PATH = './data/boston_predict_subset.csv'

learning_rate = 0.001
epochs = 20

n_samples = 0

# load training data
train_csv_file = open(TRAIN_CSV_PATH, 'r')
lines = train_csv_file.readlines()
n_samples = len(lines) - 1

train_crim = []
train_indus = []
train_tax = []
train_medv = []

for line in lines[1:]:
    train_crim.append(float(line.split(',')[0]))
    train_indus.append(float(line.split(',')[1]))
    train_tax.append(float(line.split(',')[2]))
    train_medv.append(float(line.split(',')[3]))

train_csv_file.close()

train_crim = np.asarray(train_crim)
train_indus = np.asarray(train_indus)
train_tax = np.asarray(train_tax)
train_medv = np.asarray(train_medv)

print(train_crim)

# finished loading training data
# start making model

crim = tf.placeholder(tf.float32, [])
indus = tf.placeholder(tf.float32, [])
tax = tf.placeholder(tf.float32, [])
medv = tf.placeholder(tf.float32, [])

W_crim = tf.Variable(np.random.randn(), name='weights_crim')
W_indus = tf.Variable(np.random.randn(), name='weights_indus')
W_tax = tf.Variable(np.random.randn(), name='weights_tax')
B = tf.Variable(np.random.randn(), name='bias')

pred = W_crim * crim + W_indus * indus + W_tax * tax + B

cost = tf.reduce_sum(((pred - medv) ** 2) / (2 * n_samples))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

builder = tf.saved_model.builder.SavedModelBuilder('./savedModel_new')

init = tf.global_variables_initializer()

with tf.Session() as sesh:
    sesh.run(init)

    for epoch in range(epochs):
        for (crim_val, indus_val, tax_val, medv_val) in zip(train_crim, train_indus, train_tax, train_medv):
            #print(crim_val, indus_val, tax_val, medv_val, 'k')
            sesh.run(optimizer, feed_dict={crim: crim_val, indus: indus_val, tax: tax_val, medv: medv_val})

        if not epoch % 100:
            #c = sesh.run(cost, feed_dict={crim: train_crim, indus: train_indus, tax: train_tax, medv: train_medv})
            w_crim = sesh.run(W_crim)
            w_indus = sesh.run(W_indus)
            w_tax = sesh.run(W_tax)
            b = sesh.run(B)
            print(f'epoch: {epoch:4d} w1={w_crim:.7f} w2={w_indus:.7f} w3={w_tax:.7f} b={b:.7f}')

    # converter = tf.lite.TFLiteConverter.from_session(input_tensors=[crim, indus, tax], output_tensors=[medv])
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)

    # tensor_info_crim = tf.saved_model.utils.build_tensor_info(crim)
    # tensor_info_indus = tf.saved_model.utils.build_tensor_info(indus)
    # tensor_info_tax = tf.saved_model.utils.build_tensor_info(tax)
    # tensor_info_medv = tf.saved_model.utils.build_tensor_info(medv)

    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={'crim': tensor_info_crim, 'indus': tensor_info_indus, 'tax': tensor_info_indus},
    #         outputs={'medv': tensor_info_medv},
    #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    #
    # builder.add_meta_graph_and_variables(
    #     sesh, [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #             prediction_signature
    #     },
    #     main_op=tf.tables_initializer(),
    #     strip_default_attrs=True)

#builder.save()
