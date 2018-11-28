launch server
nvidia-docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source=$(pwd)/models/servable_v3,target=/models/servable_v3 -t --entrypoint=tensorflow_model_server tensorflow/serving:1.11.1-devel-gpu --enable_batching --port=8500 --rest_api_port=8501 --model_name=inceptionV3 --model_base_path=/models/servable_v3


export model: python export_model.py -checkpoint_dir /home/andy/Data/ckpts/pretrained/inception-v3


run server: docker run -p 8501:8501 --mount type=bind,source=/home/andy/Projects/Python/inception_serving/models/servable_v3,target=/models/servable_v3 -e MODEL_NAME=servable_v3 -t tensorflow/serving:latest

NV_GPU=1 nvidia-docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source=$(pwd)/models,target=/models -it tensorflow/serving:latest-gpu --model_config_file=/models/models.config --batching_parameters_file=models/batch.config

build wraper: docker build -t feature_server .
run wraper: docker run -it -e servable_url='http://192.168.0.117:8501/v1/models/servable_v3:predict' -p 5048:80 feature_server
