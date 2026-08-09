import os

class Session:
    def __init__(self, inference_client=None, model_name=None, api_key=None):
        self._inference_client = inference_client
        self._model_name = model_name
        self._api_key = api_key
        self._title = "Chat Session 1"

    @property
    def inference_client(self):
        return self._inference_client

    @inference_client.setter
    def set_inference_client(self, inference_client):
        self._inference_client = inference_client

    @inference_client.deleter
    def delete_inference_client(self):
        del self._inference_client

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def set_model_name(self, model_name):
        self._model_name = model_name

    @model_name.deleter
    def delete_model_name(self):
        del self._model_name

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def set_api_key(self, api_key):
        self._api_key = api_key

    @api_key.deleter
    def delete_api_key(self):
        del self._api_key

    @property
    def title(self):
        return self._title

    @title.setter
    def set_title(self, title):
        self._title = title

    @title.deleter
    def delete_title(self):
        del self._title

