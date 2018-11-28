from flask import Flask, abort, request, jsonify
import requests
import json
import os
from io import BytesIO
from PIL import Image
import base64
import numpy as np

servable_url = os.environ["servable_url"]

app = Flask(__name__)


@app.route("/features", methods=["POST"])
def extract_features():
    try:
        if "url" in request.json and request.json["url"]:
            res = requests.get(request.json['url'], timeout=10)
            res.raise_for_status()
            img = Image.open(BytesIO(res.content))
        elif "base64" in request.json:
            img = Image.open(BytesIO(base64.b64decode(request.json['base64'].split(',')[-1])))
        else:
            raise Exception("Not Support Inputs")
        longersize = max(img.size)
        background = Image.new('RGB', (longersize, longersize), (255, 255, 255))
        background.paste(img, (int((longersize - img.size[0]) / 2), int((longersize - img.size[1]) / 2)))
        img = background
        resize_image = img.resize((299, 299), Image.BICUBIC)

        normalzie_image = (np.asarray(resize_image) - 128.0) / 128.0


        base64_img = str(base64.b64encode(normalzie_image.tobytes()))


        # byte_io = BytesIO()
        #
        # resize_image.save(byte_io, format='JPEG')
        # byte_img = byte_io.getvalue()
        # base64_img = str(base64.b64encode(byte_img))

        req_content = {
            "signature_name": "predict_images",
            "instances": [
                {"images": {"b64": base64_img[2:-1]}}
            ]
        }

        resp = requests.post(servable_url, json=req_content, timeout=20)
        if resp.status_code == 200:
            result = json.loads(resp.text)
            features = result['predictions'][0]['features']
            # print(result['predictions'][0]['classes'])
            return jsonify({"features": features})
        else:
            raise Exception(resp.text)

    except Exception as err:
        app.logger.error(str(err))
        abort(400)


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=True, port=5048)