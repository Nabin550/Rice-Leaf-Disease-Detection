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
    
    if isinstance(images, list):  # Check if images is a list (batch processing)
        for image in images:
            h,w = image.shape   
            # glcm_mean = fast_glcm_mean(image)  
            # glcm_std = fast_glcm_std(image)
            glcm_contrast = fast_glcm_contrast(image)
            glcm_dissimilarity = fast_glcm_dissimilarity(image)
            glcm_homogeneity = fast_glcm_homogeneity(image)
            glcm_asm, glcm_energy = fast_glcm_ASM(image)
            # glcm_max = fast_glcm_max(image)
            glcm_entropy = fast_glcm_entropy(image)

            features = np.concatenate([  glcm_contrast.ravel(),
                                    glcm_dissimilarity.ravel(), glcm_homogeneity.ravel(),
                                     glcm_energy.ravel(),
                                    glcm_entropy.ravel()])
            pass
    else:
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
        pass
    fs=16
    plt.subplot(2,5,4)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(glcm_contrast)
    plt.title('contrast', fontsize=fs)

    plt.subplot(2,5,5)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(glcm_dissimilarity)
    plt.title('dissimilarity', fontsize=fs)

    plt.subplot(2,5,6)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(glcm_homogeneity)
    plt.title('homogeneity', fontsize=fs)

    plt.subplot(2,5,8)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(glcm_energy)
    plt.title('energy', fontsize=fs)

    plt.subplot(2,5,10)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(glcm_entropy)
    plt.title('entropy', fontsize=fs)

    return np.array(features)



# Load the saved model
loaded_model = joblib.load(r'U:\GLCM\final\svm_model22.pkl')
new_image_path = r"C:\Users\nitro\Desktop\test\6.png"
new_img = np.array(Image.open(new_image_path).convert('L'))
image_size = (128, 128)
resizedimage = cv2.resize(new_img, image_size)
new_image_features = calculate_glcm_features(resizedimage)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(resizedimage, cmap='gray')
plt.title('Preprocessed Image')
plt.axis('off')
# Use the trained model to predict the class of the new image
predicted_class = loaded_model.predict([new_image_features])[0]
print("Predicted class for the new image:", predicted_class)
