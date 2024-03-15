import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from cnnClassifier.components.model_trainer import Training

class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


      
    def predict(self):
        ## load model
        
        # model = load_model(os.path.join("artifacts","training", "model.h5"))
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(Training.model.predict(test_image), axis=1)
        result = int(result)
        class_indice = dict((v,k) for k,v in Training.train_generator.class_indices.items())

        print("the class is ", class_indice[result])
        
