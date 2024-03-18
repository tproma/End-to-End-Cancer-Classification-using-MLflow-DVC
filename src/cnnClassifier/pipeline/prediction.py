import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


      
    def predict(self):
        ## load model
        # model = load_model(os.path.join("artifacts","training", "model.h5"))
        model = load_model(os.path.join("model", "ct_vgg_best_model.hdf5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (350,350))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        result = int(result)

        class_names = ["Adenocarcinoma", "Large cell carcinoma", "Normal", "Squamous cell carcinoma"]
        print("the class is ", class_names[result])
        

'''
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()

        class_indice = training.class_names()
'''    
        
