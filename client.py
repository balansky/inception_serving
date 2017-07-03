import numpy as np

from grpc.beta import implementations
import tensorflow as tf

from pd2 import predict_pb2
from pd2 import prediction_service_pb2
import time
from PIL import Image
import concurrent.futures

tf.app.flags.DEFINE_string("host", "localhost", "gRPC server host")
tf.app.flags.DEFINE_integer("port", 9000, "gRPC server port")
tf.app.flags.DEFINE_string("model_name", "inception", "TensorFlow model name")
tf.app.flags.DEFINE_integer("model_version", 1, "TensorFlow model version")
tf.app.flags.DEFINE_float("request_timeout", 10.0, "Timeout of gRPC request")
tf.app.flags.DEFINE_integer("concurrency", 1000, "Concurrency Num")
FLAGS = tf.app.flags.FLAGS


def predict(enable_result=True):
    host = FLAGS.host
    port = FLAGS.port
    model_name = FLAGS.model_name
    model_version = FLAGS.model_version
    request_timeout = FLAGS.request_timeout
    with open('test.jpeg', 'rb') as f:
        data = f.read()
    curr = time.time()
    image_tensor_proto = tf.contrib.util.make_tensor_proto(data, shape=[1])
    # Create gRPC client and request
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'predict_images'
    if model_version > 0:
        request.model_spec.version.value = model_version
    request.inputs['images'].CopyFrom(image_tensor_proto)
    result = stub.Predict(request, request_timeout)
    end_time = time.time() - curr
    if enable_result:
        print(result)
    return float(end_time)


def concurrent_predict():
    concurrency_num = FLAGS.concurrency
    st = time.time()
    e_t = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
      futures = []
      for _ in range(0, concurrency_num):
        futures.append(executor.submit(predict, False))
      for future in futures:
        tc = future.result()
        print(tc)
        e_t += tc
    print("avg time: " + str(e_t/concurrency_num))
    print("end time: " + str(time.time() - st))

def main(unused_argv=None):
    if FLAGS.concurrency > 1:
        concurrent_predict()
    else:
        print(predict())


if __name__== "__main__":
    tf.app.run()