from tensorflow import keras
from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf
import tensorflow_transform as tft
import constants
import os

device = "gpu"

if device == "tpu":
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.experimental.TPUStrategy(resolver)
else:
  strategy = tf.distribute.MultiWorkerMirroredStrategy()

_NUMERICAL_FEATURES = constants.NUMERICAL_FEATURES
_LABEL_KEY = constants.LABEL_KEY
transformed_label_key = constants.t_name(_LABEL_KEY)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=None,
              batch_size=32) -> tf.data.Dataset:

    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_label_key
        )

    return dataset

def _get_tf_examples_serving_signature(model, tf_transform_output):
  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
  ])
  def serve_tf_examples_fn(serialized_tf_examples):
      raw_feature_spec = tf_transform_output.raw_feature_spec()
      raw_feature_spec.pop(_LABEL_KEY)
      raw_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
      transformed_features = tf_transform_output.transform_features_layer()(raw_features)
      outputs = model(transformed_features)
      return {'outputs': outputs}

  return serve_tf_examples_fn

def get_model():
    for key in _NUMERICAL_FEATURES:
        input_numeric = []
        input_numeric.append(tf.keras.Input(shape=(1,),
            name=constants.t_name(key),
            dtype=tf.float32
        ))

        inputs = input_numeric
        wide = tf.keras.layers.concatenate(inputs)

        x = tf.keras.layers.Flatten()(wide)
        x = tf.keras.layers.Dense(128, activation='relu')(x)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs, output)

        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics = ['mean_squared_error'])

        model.summary()

    return model

def run_fn(fn_args: FnArgs):
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
  train_dataset = _input_fn(fn_args.train_files[0], tf_transform_output, 10)
  eval_dataset = _input_fn(fn_args.eval_files[0], tf_transform_output, 10)

  model = get_model()

  log_dir = os.path.join(fn_args.model_run_dir, 'logs')
  os.makedirs(log_dir, exist_ok=True)
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch')

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  model.tft_layer = tf_transform_output.transform_features_layer()

  signatures = {
      'serving_default':
          _get_tf_examples_serving_signature(model, tf_transform_output)
  }

  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
