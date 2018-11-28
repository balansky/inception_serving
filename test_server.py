import requests
import json

from io import BytesIO
from PIL import Image
import base64
import numpy as np
import concurrent.futures
import time

def encode_base64(np_img):
    byte_img = image_byte(np_img)
    base64_img = str(base64.b64encode(byte_img))
    return base64_img

def image_byte(np_img):
    pil_img = Image.fromarray(np_img)
    byte_io = BytesIO()
    pil_img.save(byte_io, format='JPEG')
    byte_img = byte_io.getvalue()
    return byte_img


def main(base64_img, i=0):
    server_url = "http://localhost:8501/v1/models/inception-v3:predict"
    # pil_img = Image.open("/home/andy/Pictures/0012.jpg")
    # pil_img = pil_img.resize((299, 299))
    # np_img = (np.asarray(pil_img) - 128.0)/128.0
    # base64_img = str(base64.b64encode(np_img.tobytes()))

    # byte_io = BytesIO()
    #
    # pil_img.save(byte_io, format='JPEG')
    # byte_img = byte_io.getvalue()
    # base64_img = str(base64.b64encode(byte_img))

    req_content = {
        "signature_name": "predict_images",
        "instances":[
            {"images": {"b64": base64_img[2:-1]}}
        ]
    }

    resp = requests.post(server_url, json=req_content, timeout=20)
    if resp.status_code == 200:
        # ff = json.loads(resp.text)
        print("[%d]Success" % i)
        return "Success"
    else:
        print("[%d]Fail: %s" % (i, resp.text))
        return "Fail"


def predict(i=0):
    predict_url = "http://207.216.142.92:49023/features"
    # predict_url = "http://localhost:5048/features"

    pil_img = Image.open("test.jpeg")
    byte_io = BytesIO()
    #
    pil_img.save(byte_io, format='JPEG')
    byte_img = byte_io.getvalue()
    base64_img = str(base64.b64encode(byte_img))
    req_content = {
        "base64": base64_img[2:-1]
    }

    req_content = {
        "url": "https://cdn.shopify.com/s/files/1/0001/8336/9770/products/df824611b256b6239e83c537f397d8a3_d972b132-00f0-4656-bdb7-a11458c47811.png?v=1532060017"
    #     "url": "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/12233437/Entlebucher-Mountain-Dog-On-White-01.jpg"
    }

    resp = requests.post(predict_url, json=req_content, timeout=20)

    if resp.status_code == 200:
        ff = json.loads(resp.text)
        f = np.asarray(ff['features'], dtype=np.float64).tolist()
        print("[%d]Success" % i)
        return "Success"
    else:
        print("[%d]Fail: %s" % (i, resp.text))
        return "Fail"


def concurrent_test():
    concurrency_num = 1000
    st = time.time()
    pil_img = Image.open("/home/andy/Pictures/0012.jpg")
    pil_img = pil_img.resize((299, 299))
    np_img = (np.asarray(pil_img) - 128.0)/128.0
    base64_img = str(base64.b64encode(np_img.tobytes()))

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for i in range(0, concurrency_num):
            futures.append(executor.submit(predict, i))
            # futures.append(executor.submit(main, base64_img, i))
        for future in futures:
            tc = future.result()
            # print(tc)
    e_t = time.time() - st
    print("avg time: " + str(e_t / concurrency_num))
    print("end time: " + str(e_t))

# main()
predict()
# concurrent_test()