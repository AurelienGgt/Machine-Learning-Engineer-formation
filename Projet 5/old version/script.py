import cv2
import numpy as np
import tensorflow as tf

efficient_model = tf.keras.models.load_model('/content/drive/My Drive/Colab Notebooks/Projet_6/efficient_net_model')

img = cv2.imread("nom_de-l'image")
img = cv2.resize(img,(224, 224))
img = np.expand_dims(img,axis =0)

preds = efficient_model.predict(img) # using model for the prediction

pred_label =  breed_names[np.argmax(preds)]

print(f" The prediction for the breed is :", pred_label)
print(f"The probability is :", round(max(preds[0]),4) )