import tensorflow as tf
import os
from tensorflow.python.framework import graph_util


# def load_model_classes(synset_file, metadata_file):
#     with open(synset_file) as f:
#         synsets = f.read().splitlines()
#     # Create synset->metadata mapping
#     texts = {}
#     with open(metadata_file) as f:
#         for line in f.read().splitlines():
#             parts = line.split('\t')
#             assert len(parts) == 2
#             texts[parts[0]] = parts[1]
#     class_descriptions = ['unused background']
#     for s in synsets:
#         class_descriptions.append(texts[s])
#     return class_descriptions

def load_model_classes(metadata_file, synset_file):
    texts = {}
    with open(metadata_file) as f:
        for line in f.read().splitlines():
            parts = line.split('\t')
            assert len(parts) == 2
            texts[parts[0]] = parts[1]

    sysnets_map = {}
    proto_as_ascii = tf.gfile.GFile(synset_file).readlines()
    for line in proto_as_ascii:
        if line.startswith('  target_class:'):
            target_class = int(line.split(': ')[1])
        if line.startswith('  target_class_string:'):
            target_class_string = line.split(': ')[1]
            sysnets_map[target_class] = target_class_string[1:-2]
    class_descriptions = ['unused background']
    for i in range(1, 1001):
        class_descriptions.append(texts[sysnets_map[i]])
    return class_descriptions



def preprocess_image(jpeg):
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


def load_incept_graph_def(graph_path):
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def lookup_top_classes(logits, class_tensor, num_tops):
    with tf.device('/cpu:0'):
        values, indices = tf.nn.top_k(logits, num_tops)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(class_tensor)
        classes = table.lookup(tf.to_int64(indices))
    return classes, values


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


# def build_prediction_signature(jpegs, embeddings):
#     predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(jpegs)
#     embedding_tensor_info = tf.saved_model.utils.build_tensor_info(embeddings)
#     prediction_signature = (
#         tf.saved_model.signature_def_utils.build_signature_def(
#             inputs={"images": predict_inputs_tensor_info},
#             outputs={
#                 'features': embedding_tensor_info
#             },
#             method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
#         ))
#     return prediction_signature


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


def main():
    graph_path = "/home/andy/Projects/Python/scope_pipeline/models/classify_image_graph_def.pb"
    output_dir = "models/servable_v3"
    SYNSET_FILE = os.path.join("/home/andy/Projects/Python/test/model/inception", 'imagenet_synset_to_human_label_map.txt')
    METADATA_FILE = os.path.join("/home/andy/Projects/Python/test/model/inception", 'imagenet_2012_challenge_label_map_proto.pbtxt')
    NUM_TOP_CLASSES = 5

    with tf.Graph().as_default() as tf_graph:
        graph_def = load_incept_graph_def(graph_path)
        tf.import_graph_def(graph_def, name='')
        pool_tensor = tf_graph.get_tensor_by_name('pool_3:0')

        ops = pool_tensor.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                # o._shape = tf.TensorShape(new_shape)
                o.set_shape(tf.TensorShape(new_shape))
        w = tf_graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        # logits = tf.matmul(tf.squeeze(pool3), w)
        inception_embeddings = tf.squeeze(pool_tensor, [1, 2], name='SpatialSqueeze')
        logits = tf.matmul(inception_embeddings, w)
        softmax = tf.nn.softmax(logits, name='softmax_result')
        # logits = tf_graph.get_tensor_by_name('softmax:0')


        output_path = os.path.join(tf.compat.as_bytes(output_dir), tf.compat.as_bytes(str(1)))

    with tf.Session(graph=tf_graph) as sess_train:
        sess_train.run([tf.global_variables_initializer()])
        output_graph_def = graph_util.convert_variables_to_constants(
            sess_train,
            tf_graph.as_graph_def(add_shapes=True), ["softmax_result", "SpatialSqueeze"])

    with tf.Graph().as_default() as export_graph:
        print('Exporting trained model to', output_path)
        incoming_data = tf.placeholder(tf.string, name='incoming_data')
        jpegs, images = parse_incoming_data(incoming_data)
        # tf.import_graph_def(output_graph_def, name="", input_map={'Mul:0': images})
        tf.import_graph_def(output_graph_def, name="", input_map={'Mul:0': images})

        sess = tf.Session(graph=export_graph)
        logits = sess.graph.get_tensor_by_name("softmax_result:0")
        inception_embeddings = sess.graph.get_tensor_by_name('SpatialSqueeze:0')
        class_tensor = tf.constant(load_model_classes(SYNSET_FILE, METADATA_FILE))
        top_classes, class_values = lookup_top_classes(logits, class_tensor, NUM_TOP_CLASSES)

        builder = tf.saved_model.builder.SavedModelBuilder(output_path)
        classification_signature = build_classification_signature(incoming_data, top_classes, class_values)
        prediction_signature = build_prediction_signature(jpegs, inception_embeddings, top_classes)
        # prediction_signature = build_prediction_signature(jpegs, inception_embeddings)


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
        print('Successfully exported model to %s' % output_dir)


    # with tf.Session(graph=tf_graph) as sess_train:
    #     sess_train.run([tf.global_variables_initializer()])
    #     output_graph_def = graph_util.convert_variables_to_constants(
    #         sess_train,
    #         tf_graph.as_graph_def(add_shapes=True))
    # with tf.Graph().as_default() as export_graph:
    #     tf.import_graph_def(output_graph_def, name="")

    # sess = tf.Session(graph=export_graph)

    # legacy_init_op = tf.group(tf.global_variables_initializer(), name='legacy_init_op')
    # sess.run(tf.global_variables_initializer())
    # print('Exporting trained model to', output_path)
    # builder = tf.saved_model.builder.SavedModelBuilder(output_path)
    # prediction_signature = build_prediction_signature(jpegs, prelogits)
    # builder.add_meta_graph_and_variables(
    #     sess, [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #         'predict_images':
    #             prediction_signature,
    #     })
    # builder.save()
    # print('Successfully exported model to %s' % output_path)

# def inference():
#     import numpy as np
#     from PIL import Image
#     graph_path = "/home/andy/Projects/Python/scope_pipeline/models/classify_image_graph_def.pb"
#     img = Image.open("test2.jpg")
#     longersize = max(img.size)
#     background = Image.new('RGB', (longersize, longersize), (255, 255, 255))
#     background.paste(img, (int((longersize - img.size[0]) / 2), int((longersize - img.size[1]) / 2)))
#     img = background
#     resize_image = img.resize((299, 299), Image.BICUBIC)
#
#     normalzie_image = (np.asarray(resize_image) - 128.0) / 128.0
#     np_img = np.expand_dims(normalzie_image, 0)
#
#     with tf.Graph().as_default() as tf_graph:
#         graph_def = load_incept_graph_def(graph_path)
#         tf.import_graph_def(graph_def, name='')
#         pool_tensor = tf_graph.get_tensor_by_name('pool_3:0')
#         input_tensor = tf_graph.get_tensor_by_name('Mul:0')
#         sess = tf.Session(graph=tf_graph)
#         features = sess.run(pool_tensor, feed_dict={input_tensor: np_img})
#         features = np.squeeze(features)
#         features = features.tolist()
#         print('sss')

main()
# inference()