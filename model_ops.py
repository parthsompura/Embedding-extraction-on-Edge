#download model
#load model
#ask model_for_embedding
import logging
import os

import numpy as np
import tensorflow as tf
import config as cfg


class EmbeddingModel():
    def __init__(self, path=cfg.model_path, version=cfg.model_version):
        self.path = path
        self.version = version
        self.model = None


    def load_model(self):
        try:
            print(os.getcwd())
            self.model = tf.saved_model.load(self.path)
            logging.info("model loaded properly")
        except Exception as e:
            logging.error('model not loaded '+str(e))

    @staticmethod
    def prepare_clean_embedding(prediction):
        embedding = prediction.numpy()[0]
        return embedding.tolist()


    def get_embedding(self, text):
        embedding = []
        err_message = None
        try:
            prediction = self.model([text])
        except Exception as e:
            err_message = str(e)
            logging.error("problem with embedding creation"+err_message)
            print("problem with embedding creation", err_message)
        else:
            if isinstance(prediction, tf.Tensor):
                embedding = self.prepare_clean_embedding(prediction)
        return (embedding, err_message)
