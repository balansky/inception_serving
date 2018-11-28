import os
import tensorflow as tf

from inception import inception_model
from tensorflow.python.ops import control_flow_ops


tf.app.flags.DEFINE_string('checkpoint_dir', 'models/v3',
                           """Directory where to read training checkpoints.""")
tf.app.flags.DEFINE_string('output_dir', 'models/servable_v3',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")
tf.app.flags.DEFINE_integer('image_size', 299,
                            """Needs to provide same value as in training.""")
FLAGS = tf.app.flags.FLAGS


def load_model_classes(synset_file, metadata_file):
    with open(synset_file) as f:
        synsets = f.read().splitlines()
    # Create synset->metadata mapping
    texts = {}
    with open(metadata_file) as f:
        for line in f.read().splitlines():
            parts = line.split('\t')
            assert len(parts) == 2
            texts[parts[0]] = parts[1]
    class_descriptions = ['unused background']
    for s in synsets:
        class_descriptions.append(texts[s])
    return class_descriptions


def preprocess_image(jpeg):
    # image = tf.image.decode_jpeg(jpeg, channels=3)
    # # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image.set_shape([299, 299, 3])
    # image = tf.to_float(image)
    # image = (image - 128.)/128.

    image = tf.decode_raw(jpeg, out_type=tf.float64)
    image = tf.reshape(image, [299, 299, 3])
    image = tf.to_float(image)
    #
    # image = tf.image.decode_jpeg(jpeg, channels=3)
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # image.set_shape([299, 299, 3])
    # image = tf.to_float(image)
    # image = (image - 128.)/128.
    return image


def parse_incoming_data(incoming_data):
    with tf.device('/cpu:0'):
        feature_configs = {
            'image/encoded': tf.FixedLenFeature(shape=[], dtype=tf.string)
        }
        tf_examples = tf.parse_example(incoming_data, feature_configs)
        jpegs = tf_examples['image/encoded']
        images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
    return jpegs, images


def lookup_top_classes(logits, class_tensor, num_tops):
    with tf.device('/cpu:0'):
        values, indices = tf.nn.top_k(logits, num_tops)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        classes = table.lookup(tf.to_int64(indices))
    return classes, values

def restore_check_point(tf_graph, checkpoint_dir):
    variable_averages = tf.train.ExponentialMovingAverage(0.9999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    sess = tf.Session(graph=tf_graph)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Successfully loaded model from %s at step=%s.' % (ckpt.model_checkpoint_path, global_step))
    else:
        raise Exception('No checkpoint file found at %s' % checkpoint_dir)
    return sess


def build_classification_signature(incoming_data, classes, class_values):
    classify_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
        incoming_data)
    classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(
        classes)
    scores_output_tensor_info = tf.saved_model.utils.build_tensor_info(class_values)

    classification_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                    classify_inputs_tensor_info
            },
            outputs={
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                    classes_output_tensor_info,
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                    scores_output_tensor_info
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
        ))
    return classification_signature


def build_prediction_signature(jpegs, embeddings, classes):
    classes_output_tensor_info = tf.saved_model.utils.build_tensor_info(
        classes)
    predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(jpegs)
    embedding_tensor_info = tf.saved_model.utils.build_tensor_info(embeddings)
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"images": predict_inputs_tensor_info},
            outputs={
                'classes': classes_output_tensor_info,
                'features': embedding_tensor_info
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        ))
    return prediction_signature



def main(unused_argv=None):
    with tf.Graph().as_default() as tf_graph:
        class_tensor = tf.constant(load_model_classes(SYNSET_FILE, METADATA_FILE))
        incoming_data = tf.placeholder(tf.string, name='incoming_data')
        jpegs, images = parse_incoming_data(incoming_data)
        logits, end_points = inception_model.inference(images, NUM_CLASSES + 1)
        inception_embeddings = end_points['prelogits']
        top_classes, class_values = lookup_top_classes(logits, class_tensor, NUM_TOP_CLASSES)
        sess =  restore_check_point(tf_graph, CHECKPOINT_DIR)
        output_path = os.path.join(tf.compat.as_bytes(FLAGS.output_dir), tf.compat.as_bytes(str(FLAGS.model_version)))
        print('Exporting trained model to', output_path)
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)
        classification_signature = build_classification_signature(incoming_data, top_classes, class_values)
        prediction_signature = build_prediction_signature(jpegs, inception_embeddings, top_classes)
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_images':
                    prediction_signature,
                tf.saved_model.signature_constants.
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    classification_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
        print('Successfully exported model to %s' % FLAGS.output_dir)


if __name__=="__main__":
    NUM_CLASSES = 1000
    NUM_TOP_CLASSES = 5
    CHECKPOINT_DIR = FLAGS.checkpoint_dir
    SYNSET_FILE = os.path.join(CHECKPOINT_DIR, 'imagenet_lsvrc_2015_synsets.txt')
    METADATA_FILE = os.path.join(CHECKPOINT_DIR, 'imagenet_metadata.txt')
    tf.app.run()