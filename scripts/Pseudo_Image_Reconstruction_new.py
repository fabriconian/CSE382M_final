# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 08:19:27 2023

@author: karlf
"""

from sklearn.decomposition import PCA

import numpy as np
import datetime
import scipy as sc
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib import style

start_time = datetime.datetime.now()
d = 32*32*3
#generate 3 RP matrices as given in sklearn pachage
R_s=[]
k_range=[1, 10, 100]
for k in k_range:
    R_s.append(np.random.normal(loc=0,scale=np.sqrt(1/k),size=(d,k)))


#Generate 3 SRP Matrices as given in sklearn package
#density = np.sqrt(1/d)
#s = 1/density
s = np.sqrt(d)
p1 = 1/(2*s)
p3 = 1/(2*s)
p2 = 1-p1-p3
SRP_s=[]
for k in k_range:
    vk_1 = -1*np.sqrt(s/k)
    vk_2 = 0
    vk_3 = np.sqrt(s/k)
    SRP_s.append(np.random.choice(a=[vk_1,vk_2,vk_3],size=(d,k),p=[p1,p2,p3]))

#Generate 3 PCA objects with different values of n_conponents, or k
pca_set = []
for k in k_range:
    pca_set.append(PCA(n_components=k,svd_solver='full'))
pca1 = pca_set[0]
pca10 = pca_set[1]
pca100 = pca_set[2]

#Extract images from reshaped flower image dataset
data_path = 'C:/Users/karlf/Downloads/dataset/input_resized/flowers'

#select 10 samples from our dataet and change them to numpy
#single line arrays
category_paths = []
categories = []
samples = [] #list of original sample images, 32 by 32 by 3
numpy_samples = [] #list of image data as numpy arrays
for category in os.listdir(data_path):
    category_path = os.path.join(data_path, category)
    category_paths.append(category_path)
    categories.append(category)
    length = len(os.listdir(category_path))
    i1 = np.int(length/2)
    i2 = i1+1
    image1_dir = os.listdir(category_path)[i1]
    image2_dir = os.listdir(category_path)[i2]
    image1 = Image.open(os.path.join(category_path,image1_dir))
    image2 = Image.open(os.path.join(category_path,image2_dir))
    samples.append(image1)
    samples.append(image2)
    array1 = np.array(image1).reshape(-1) / 255
    array2 = np.array(image2).reshape(-1) / 255
    numpy_samples.append(array1)
    numpy_samples.append(array2)
    #print(length)
numpy_samples = np.array(numpy_samples)   

#extract all samples, change to numpy single line array form
all_samples = []
for category in os.listdir(data_path):
    category_path = os.path.join(data_path, category)
    for file in os.listdir(category_path):
        file_path = os.path.join(category_path, file)
        if os.path.isfile(file_path):
            image = Image.open(file_path)
            array_here=np.array(image).reshape(-1) / 255
            all_samples.append(array_here)

#project numpy array for 10 images, and then project back
#Gaussian Random Projection
RP_reconstruct_arrays = []
for i in range(len(R_s)):
    Rk = R_s[i]
    RP_reconstruct_arrays.append(numpy_samples@Rk@Rk.T)
#Sparse Random Projection
SRP_reconstruct_arrays = []
for i in range(len(SRP_s)):
    SRPk = SRP_s[i]
    SRP_reconstruct_arrays.append(numpy_samples@SRPk@SRPk.T)

#PCA for k=1,k=10,k=100
PCA_reconstruct_arrays = []
for i in range(len(pca_set)):
    pca_k = pca_set[i]
    pca_k.fit(all_samples)
    x_k_raw = pca_k.transform(numpy_samples)
    reconstruct_k = pca_k.inverse_transform(x_k_raw)
    PCA_reconstruct_arrays.append(reconstruct_k)

output_folder = 'C:/Users/karlf/Downloads/projected_flowers_sp1'
#reconstruct images from collection of numpy arrays
RP_reconstruction = []
SRP_reconstruction = []
PCA_reconstruction = []
for i in range(0,len(RP_reconstruct_arrays)):
    lis = []
    str0 = 'Random Projection'
    str1 = 'k=' + str(k_range[i])
    path0 = os.path.join(output_folder,str0,str1)
    os.makedirs(path0)
    for j in range(0,len(RP_reconstruct_arrays[i])):
        name = 'RP_sample_' + str(j) + '.jpg'
        array_here = RP_reconstruct_arrays[i][j]
        a = array_here.reshape((32,32,3))*255
        img = Image.fromarray(a, mode="RGB")
        lis.append(img)
        output_path = os.path.join(path0,name)
        img.save(output_path)
    RP_reconstruction.append(lis)
for i in range(0,len(SRP_reconstruct_arrays)):
    lis = []
    str0 = 'Sparse Random Projection'
    str1 = 'k=' + str(k_range[i])
    path0 = os.path.join(output_folder,str0,str1)
    os.makedirs(path0)
    for j in range(0,len(SRP_reconstruct_arrays[i])):
        name = 'SRP_sample_' + str(j) + '.jpg'
        array_here = SRP_reconstruct_arrays[i][j]
        a = array_here.reshape((32,32,3))*255
        img = Image.fromarray(a, mode="RGB")
        lis.append(img)
        output_path = os.path.join(path0,name)
        img.save(output_path)
    SRP_reconstruction.append(lis)
    
for i in range(0,len(PCA_reconstruct_arrays)):
    lis = []
    str0 = 'PCA Projection'
    str1 = 'k=' + str(k_range[i])
    path0 = os.path.join(output_folder,str0,str1)
    os.makedirs(path0)
    for j in range(0,len(PCA_reconstruct_arrays[i])):
        name = 'PCA_sample_' + str(j) + '.jpg'
        array_here = PCA_reconstruct_arrays[i][j]
        a = array_here.reshape((32,32,3))*255
        img = Image.fromarray(a, mode="RGB")
        lis.append(img)
        output_path = os.path.join(path0,name)
        img.save(output_path)
    PCA_reconstruction.append(lis)

str0 = 'Original Sample Images'
path0 = os.path.join(output_folder,str0)
os.makedirs(path0)
for i in range(0,len(samples)):
    name = 'Original sample' + str(i) +'.jpg'
    output_path = os.path.join(path0,name)
    samples[i].save(output_path)

#Showing sample images for RP Projection
fig,ax=plt.subplots(5,4)
fig.set_size_inches(13,15)
for i in range(5):
    for j in range (4):
        if j==0:
            ax[i,j].imshow(samples[2*i])
            ax[i,j].set_title('Flower: '+categories[i])
        else:
            ax[i,j].imshow(RP_reconstruction[j-1][2*i])
            ax[i,j].set_title('RP k= '+str(k_range[j-1]))        
plt.tight_layout()
plt.savefig('RP Projection Combined.jpg')
#Showing sample images for SRP Projection
fig,ax=plt.subplots(5,4)
fig.set_size_inches(13,15)
for i in range(5):
    for j in range (4):
        if j==0:
            ax[i,j].imshow(samples[2*i])
            ax[i,j].set_title('Flower: '+categories[i])
        else:
            ax[i,j].imshow(SRP_reconstruction[j-1][2*i])
            ax[i,j].set_title('SRP k= '+str(k_range[j-1]))  
plt.savefig('SRP Projection Combined.jpg')            
#Showing sample images for PCA Projection
fig,ax=plt.subplots(5,4)
fig.set_size_inches(13,15)
for i in range(5):
    for j in range (4):
        if j==0:
            ax[i,j].imshow(samples[2*i])
            ax[i,j].set_title('Flower: '+categories[i])
        else:
            ax[i,j].imshow(PCA_reconstruction[j-1][2*i])
            ax[i,j].set_title('PCA k= '+str(k_range[j-1]))  
plt.tight_layout()    
plt.savefig('PCA Projection Combined.jpg')
end_time = datetime.datetime.now()
print("time_cost: ",end_time-start_time)
'''
Help from
https://www.geeksforgeeks.org/python-pil-image-show-method/
https://pillow.readthedocs.io/en/stable/reference/Image.html
'''