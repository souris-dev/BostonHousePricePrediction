# Converts saved model into TFLite format

import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/1576262763')
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)