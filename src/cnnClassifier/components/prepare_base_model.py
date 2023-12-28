import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
                                                


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
