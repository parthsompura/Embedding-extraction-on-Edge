import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from mtcnn import MTCNN
import pytesseract
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request, jsonify
import base64
import pickle
import os
from model_ops import EmbeddingModel

app = Flask(__name__)
CORS(app)

# Face Model
mod = MTCNN()

# Embedding Model
def get_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=True,
                                               weights='imagenet')
    x = base_model.layers[-1].output

    x = tf.keras.layers.Dense(512)(x)
    model = tf.keras.Model(inputs = base_model.input, outputs = x)
    return model

model = get_model()


# Sentence Encoder

embedding_model = EmbeddingModel('./use-models/universal-sentence-encoder_4', version=1)
embedding_model.load_model()

with open('average_vector.pkl', 'rb') as f:
    img_vec = pickle.load(f)

with open('document_vector.pkl', 'rb') as f:
    doc_vec = pickle.load(f)


def read_image(image_stream):
    nparr = np.fromstring(base64.b64decode(image_stream.split(',')[-1]), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def check_image(img):
    print(img.shape)
    faces = mod.detect_faces(img)

    if len(faces) > 0:
        return 0, ''

    txt = pytesseract.image_to_string(img)
    print('****', txt)
    if len(txt) > 1000:
        return 1, txt

    else:
        return 2, ''


def get_document_embedding(txt):
    doc_emb = embedding_model.get_embedding(txt)[0]
    return np.array(doc_emb)


def get_image_embedding(img):
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (1, 224, 224, 3))
    img_emb = model.predict(img)[0]
    return np.array(img_emb)


def driver(req):
    img = read_image(req)

    flag, txt = check_image(img)

    print('%%%', flag)

    if flag == 0:
        print('Found a Face!!')
        return {'embedding': '', 'image_type': 'face'}

    elif flag == 1:
        emb = get_document_embedding(txt)
        return {'embedding': emb, 'image_type': 'document'}

    else:
        emb = get_image_embedding(img)
        return {'embedding': emb, 'image_type': 'image'}


def convertToImage(string):
    return base64.b64decode(string.split(',')[-1])


def moveToFolder(image,filename,entity):
    image = convertToImage(image)
    if entity == "face":
        path = "/home/amishra/Classification/faces/"
    if entity == "document":
        path = "/home/amishra/Classification/documents/"

    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+filename, "wb") as f:
        f.write(image)

    return path+filename


def classifyAndSave(embedding,image,filename, entity):
    if entity == 'image':
        dic = img_vec
        pre_path = '/home/amishra/Classification/images/'
    else:
        dic = doc_vec
        pre_path = '/home/amishra/Classification/documents/'

    image = convertToImage(image)


    maxVal=-1
    predictedAnimal=""

    pred_vals = [(x, abs(np.dot(embedding,v))) for x, v in dic.items() ]
    pred_vals = sorted(pred_vals, key=lambda x: x[1], reverse=True)
    print('***', pred_vals)
    predictedAnimal = pred_vals[0][0]

    # for animalName, animalVector in dic.items():
    #     print(animalName,len(animalVector))
    #     val = abs(np.dot(embedding,animalVector))
    #     print(val, animalName)
    #     if val > maxVal:
    #         predictedAnimal = animalName
    #     else:
    #         maxVal = val
    path = pre_path+predictedAnimal+"/"

    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+filename, "wb") as f:
        f.write(image)
    return predictedAnimal,path+filename


@app.route('/embedding', methods=['POST'])
def hello():
    req = request.get_json()
    img = req['imageInfo']
    fil_name = req.get('fileName', 'dummy.jpg')
    resp = driver(img)
    if resp['image_type'] == 'face':
        path = moveToFolder(img, fil_name, "face")
        response = jsonify(**{'doc_type': resp['image_type'], 'path': path, 'prediction': 'face'})
    elif resp['image_type'] == 'document':
        prediction, path = classifyAndSave(resp['embedding'],img,fil_name, 'document')
        response = jsonify(**{'doc_type': resp['image_type'], 'path': path, 'prediction': prediction})
    else:
        prediction, path = classifyAndSave(resp['embedding'],img,fil_name, 'image')
        response = jsonify(**{'doc_type': resp['image_type'], 'path': path, 'prediction': prediction})
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0')


