
# coding: utf-8

# # Rezolvare laborator 3

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import collections


# In[2]:


dataPath = "/home/stl_man/Desktop/Fac/ML/lab1/data/"
#load train images
train_images = np.loadtxt(dataPath + "train_images.txt")
print(train_images.shape)
print(train_images.ndim)
print(type(train_images[0,0]))
print(train_images.size)
print(train_images.nbytes)


# In[3]:


#plot the first image
image = train_images[0,:]
image = np.reshape(image,(28,28))
# plt.imshow(image,cmap = "gray")
# plt.show()


# In[4]:


#load training labels
train_labels = np.loadtxt(dataPath + "train_labels.txt",'int8')
print(train_labels.shape)
print(type(train_labels[0]))
print(train_labels.size)
print(train_labels.nbytes)
#show the label of the first training image
print(train_labels[0])


# In[5]:


#plot the first 100 training images with their labels in a 10 x 10 subplot
nbImages = 10
# plt.figure(figsize=(5,5))
# for i in range(nbImages**2):
#     plt.subplot(nbImages,nbImages,i+1)
#     plt.axis('off')
#     plt.imshow(np.reshape(train_images[i,:],(28,28)),cmap = "gray")
# plt.show()
labels_nbImages = train_labels[:nbImages**2]
print(np.reshape(labels_nbImages,(nbImages,nbImages)))

# default
numberClasses = 10
PC = np.zeros(numberClasses)
for i in range(numberClasses):
    PC[i] = sum(train_labels == i)

for i in range(numberClasses):
    PC[i] = PC[i] / 1000.0

# clasa = 7
# pos = 370
#
# indexes = np.ravel(np.where(train_labels == clasa))
# print (indexes)
#
# values = train_images[indexes, pos]
# nbBins = 4
# bins = np.linspace(0, 256, nbBins + 1)
# print (bins)
#
# h = np.histogram(values, bins)
# hh = h[0] / sum(h[0])
# print(hh)

nb_classes = 10
nb_pixels = 784
nb_bins = 4

M = np.zeros((nb_classes, nb_pixels, nb_bins))
bins = np.linspace(0, 256, nb_bins + 1)

for c in range(nb_classes):
    index = np.ravel(np.where(train_labels == c))
    for pos in range(nb_pixels):
        values = train_images[index, pos]
        h = np.histogram(values, bins)
        hh = h[0] / sum(h[0])
        M[c, pos, :] = hh

# values = np.array([0, 35, 70, 190, 255])
# intervalul ( 0 .. 4 ) in care gasim un pixel anumit
# b = np.digitize(values, bins) - 1

# TESTARE
test_images = np.loadtxt(dataPath + "test_images.txt")
test_labels = np.loadtxt(dataPath + "test_labels.txt",'int8')

# img = test_images[0, :]
# logPC = np.log(PC)
#
# for pos in range(nb_pixels):
#     values = img[pos]
#     b = np.digitize(values, bins) - 1
#     for c in range(nb_classes):
#         logPC[c] += np.log(M[c, pos, b] + 0.000001)
#
# # print(logPC)
# predicted_class = logPC.argmax()
# # print (predicted_class)
#
# plt.imshow(np.reshape(img, (28, 28)), cmap = "gray")
# plt.show()


conf_mat = np.zeros((nb_classes, nb_classes))
predicted_class = np.zeros(500)

cnt = 0
for idx in range(500):
    img = test_images[idx, :]
    logPC = np.log(PC)

    for pos in range(nb_pixels):
        values = img[pos]
        b = np.digitize(values, bins) - 1
        for c in range(nb_classes):
            logPC[c] += np.log(M[c, pos, b] + 0.000001)

    # print(logPC)
    predicted_class[idx] = int(logPC.argmax())
    # print(predicted_class)
    cnt += predicted_class[idx] == test_labels[idx]

    # print (predicted_class[idx], test_labels[idx])
    conf_mat[int(predicted_class[idx]), test_labels[idx]] += 1


print (cnt / 500.0)

# Tema
# Clasificator / matricea de confuzii / greselile

print (conf_mat)

worst_pairs = []
pairs = []

for idx in range(500):
    img = test_images[idx, :]
    if predicted_class[idx] != test_labels[idx]:
        print(predicted_class[idx], test_labels[idx])
        pairs.append((int(predicted_class[idx]), test_labels[idx]))
        # plt.imshow(np.reshape(img, (28, 28)), cmap = "gray")
        # plt.show()

freq = collections.Counter(pairs)
freq_sorted = freq.most_common()

print ('(Output, Answer) -> Freq')
for key, value in freq_sorted:
    print (key, ' -> ', value)