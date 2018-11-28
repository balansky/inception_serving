import requests
import json

api_address = "https://api.scopemedia.com/search/v2/medias"

def get_test_images(client_id, client_secret):
    query = api_address + "?page={0}&size=2000"
    count = 0
    medias_all = []
    while True:
        req = requests.get(query.format(count), headers={'Content-Type': 'application/json', 'Client-Id': client_id,
                                                         'Client-Secret': client_secret})
        if req.status_code == 200:
            medias = json.loads(req.text)['medias']
            print("Get %d Images from: %s" % (len(medias), query.format(count)))
            medias_all.extend(medias)
            if 'next' in req.links:
                count += 1
            else:
                break

    results = [(m['mediaId'], m['mediaUrl']) for m in medias_all]
    with open(client_id + ".txt", 'w') as f:
        for media_id, media_url in results:
            f.write(str(media_id) + "," + media_url + "\n")


def get_feature(url):
    # predict_url = "http://207.216.142.92:49023/features"
    predict_url = "http://localhost:5048/features"

    # pil_img = Image.open("test.jpeg")
    # byte_io = BytesIO()
    #
    # pil_img.save(byte_io, format='JPEG')
    # byte_img = byte_io.getvalue()
    # base64_img = str(base64.b64encode(byte_img))
    # req_content = {
    #     "base64": base64_img[2:-1]
    # }
    req_content = {
        "url": url
    }
    resp = requests.post(predict_url, json=req_content, timeout=20)
    if resp.status_code == 200:
        res = json.loads(resp.text)
        features = ','.join(res['features'])
        return features[1:-1]
    else:
        return None


def delete_media(client_id, client_secret):

    upload_server = "http://192.168.0.142:8080/search/v2/medias/%s"
    with open(client_id + ".txt", 'r') as f:
    # with open("medias.txt", 'r') as f:
        for l in f.readlines():
            media_id, media_url = l.split(',')
            print("remove media: %s" % media_id)
            resp = requests.delete(upload_server % media_id,
                            headers={'Content-Type': 'application/json',
                                     # 'Client-Id': '05700-test-004-1',
                                     # 'Client-Secret': 'OEPhezWNLNfxsEc7D0bQ0z8wgydaJ2kJmzHhWAAKvniuO60FUjTIVVgAFb9Eqwg7'}
                                     'Client-Id': client_id,
                                     'Client-Secret': client_secret}
                            )
            if resp.status_code == 200:
                print("removed media: %s" % media_id)
            else:
                print("remove media %s fail: %s" % (media_id, str(resp.text)))



def upload_feature(client_id, client_secret):
    upload_server = "http://192.168.0.142:8080/search/v2/medias"
    count = 0
    with open(client_id + ".txt", 'r') as f:
        for l in f.readlines():
            media_id, media_url = l.split(',')
            # tq = {"url": media_url}
            # rr = requests.post("http://207.216.142.92:49023/features", json=tq, timeout=20)
            print("uploading image : %s" % media_id)
            req_content = {
                "medias": [{"mediaId": media_id,
                            "mediaUrl": media_url}
                           ]
            }
            resp = requests.post(upload_server, data=json.dumps(req_content),
                                 headers={'Content-Type': 'application/json',
                                          # 'Client-Id': '05700-test-004-1',
                                          # 'Client-Secret': 'OEPhezWNLNfxsEc7D0bQ0z8wgydaJ2kJmzHhWAAKvniuO60FUjTIVVgAFb9Eqwg7'},
                                          'Client-Id': client_id,
                                          'Client-Secret': client_secret},
                                 timeout=20)
            if resp.status_code == 200:
                count += 1
            else:
                print(resp.text)


# c_id = "shopify-6607110209-app"
# c_secret = "8ebef896-0239-5491-a4b2-4bb7df5b1154"

# c_id = "shopify@eteeo.myshopify.com"
# c_secret = "9839f3c9-4f42-5183-a5bb-7b1ef78f28f4"

# c_id = "shopify@belle-regina.myshopify.com"
# c_secret = "08f0b2d1-3d8e-5fba-a461-ac6a4cf2366c"

c_id = "shopify-183369770-app"
c_secret = "b0c951ad-d16a-5fc6-a6bd-4b7f65f7db76"

delete_media(c_id, c_secret)

# get_test_images(c_id, c_secret)
upload_feature(c_id, c_secret)


# upload_feature()
# delete_media()