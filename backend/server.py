from flask import Flask, jsonify, request
from flask_cors import CORS
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Enable CORS for all routes
import numpy as np
import joblib
from PIL import Image
import cv2

from skimage.color import rgb2gray
# coding: utf-8


import numpy as np
import joblib
from PIL import Image
import cv2

from skimage.color import rgb2gray
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data

def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, nbit+1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm



def fast_glcm_contrast(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm contrast
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i,j] * (i-j)**2

    return cont


def fast_glcm_dissimilarity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm dissimilarity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i,j] * np.abs(i-j)

    return diss


def fast_glcm_homogeneity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm homogeneity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            homo += glcm[i,j] / (1.+(i-j)**2)

    return homo


def fast_glcm_ASM(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm  += glcm[i,j]**2

    ene = np.sqrt(asm)
    return asm, ene

def fast_glcm_entropy(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    return ent




def calculate_glcm_features(images):
    features = []
    # Resize image to reduce memory usage
    h,w = images.shape
    glcm_contrast = fast_glcm_contrast(images)
    glcm_dissimilarity = fast_glcm_dissimilarity(images)
    glcm_homogeneity = fast_glcm_homogeneity(images)
    glcm_asm, glcm_energy = fast_glcm_ASM(images)
    glcm_entropy = fast_glcm_entropy(images)

    features = np.concatenate([ glcm_contrast.ravel(),
                            glcm_dissimilarity.ravel(), glcm_homogeneity.ravel(),
                                glcm_energy.ravel(),
                            glcm_entropy.ravel()])
    return np.array(features)

import numpy as np
import cv2
from PIL import Image

def predict_class_with_svm(image_path):
    model_path_svm= 'svm_model_final.pkl'
    target_names_svm = ['Blight_S', 'Brownspot_S', 'blast_S','Cant_Ss']
    threshold_svm=30
    # Load the saved model
    loaded_model_svm = joblib.load(model_path_svm)
    
    # Read and preprocess the new image
    new_img_svm = np.array(Image.open(image_path).convert('L'))
    image_size = (128, 128)
    resizedimage_svm = cv2.resize(new_img_svm, image_size)
    
    # Calculate GLCM features for the new image
    new_image_features_svm = calculate_glcm_features(resizedimage_svm)  # Assuming this function is defined
    
    # Use the trained model to predict the class of the new image
    predicted_class_SVM = loaded_model_svm.predict([new_image_features_svm])[0]
    predicted_class_name_svm = target_names_svm[predicted_class_SVM]
    
    # Use the trained model to predict the class and obtain the decision function values
    decision_function_values_svm = loaded_model_svm.decision_function([new_image_features_svm])
    
    # Compute the confidence score for the prediction
    confidence_score_svm= np.abs(decision_function_values_svm) / np.linalg.norm(loaded_model_svm.coef_)
    
    # Get the maximum confidence score
    max_confidence_svm = np.max(confidence_score_svm)
    
    # Check if the maximum confidence score is above the threshold
    if max_confidence_svm >= threshold_svm:
        return predicted_class_name_svm, max_confidence_svm
    else:
        predicted_class_name_svm=target_names_svm[3]
        return predicted_class_name_svm, max_confidence_svm
    
def predict_class_with_rf(image_path):
    model_path_rf= 'random_forest_model_final.pkl'
    target_names_rf= ['Blight_R', 'Brownspot_R', 'blast_R','Cant_R']
    threshold_rf=0.75
    # Load the saved model
    loaded_model_rf = joblib.load(model_path_rf)
    
    # Read and preprocess the new image
    new_img_rf = np.array(Image.open(image_path).convert('L'))
    image_size = (128, 128)
    resizedimage_rf= cv2.resize(new_img_rf, image_size)
    
    # Calculate GLCM features for the new image
    new_image_features_rf = calculate_glcm_features(resizedimage_rf)  # Assuming this function is defined
    
    # Use the trained model to predict the class of the new image
    predicted_class_rf = loaded_model_rf.predict([new_image_features_rf])[0]
    predicted_class_name_rf = target_names_rf[predicted_class_rf]
    
    # Predict probabilities for each class for the new image
    class_probabilities_rf= loaded_model_rf.predict_proba([new_image_features_rf])
    
    # Get the maximum probability
    max_prob_rf= np.max(class_probabilities_rf)
    
    # Check if the maximum probability is above the threshold
    if max_prob_rf >= threshold_rf:
        return predicted_class_name_rf, max_prob_rf
    else:
        predicted_class_name_rf=target_names_rf[3]
        return predicted_class_name_rf, max_prob_rf

# Define an endpoint for image classification
@app.route('/', methods=['POST','GET'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['image']
    image_path = 'temp_image.jpg'  # Save the uploaded image temporarily
    file.save(image_path)

    # Classify the image
    predicted_class,c= predict_class_with_svm(image_path)
    predicted_class1,c1= predict_class_with_rf(image_path)
    print(predicted_class)
    print(c)



    return jsonify({
        'predicted_class': predicted_class,'c':c,
        'predicted_class1': predicted_class1,'c1':c1
         }), 200

if __name__ == '__main__':
    app.run(debug=True)
