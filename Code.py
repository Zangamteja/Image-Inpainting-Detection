import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from focal_loss import BinaryFocalLoss

dir_seg = "Dataset/Mask_images/"
dir_img = "Dataset/Inpainted_Images/"
dir_seg1 = "Dataset/Test_Mask/"
dir_img1 = "Dataset/Test_Images/"

ldseg = np.array(os.listdir(dir_seg))
ldseg1 = np.array(os.listdir(dir_img))
ldseg2 = np.array(os.listdir(dir_seg1))
ldseg3 = np.array(os.listdir(dir_img1))

fnm = ldseg[0]
fnm1 = ldseg1[0]
fnm2 = ldseg2[0]
fnm3 = ldseg3[0]
print(fnm)
print(dir_img + fnm1)

seg = cv2.imread(dir_seg + fnm, 0)
img_is = cv2.imread(dir_img + fnm1)
print("seg.shape={}, img_is.shape={}".format(seg.shape, img_is.shape))

seg1 = cv2.imread(dir_seg1 + fnm2, 0)
img_is1 = cv2.imread(dir_img1 + fnm3)

n_classes = 1

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(img_is)
ax.set_title("original image")
plt.show()

def getImageArr(path, width, height):
    img = load_img(path, target_size=(width, height))
    img_array = img_to_array(img)
    img_array = np.float32(img_array) / 127.5 - 1
    return img_array

def getSegmentationArr(path, nClasses, width, height):
    seg_labels = np.zeros((height, width, nClasses))
    img = load_img(path, target_size=(width, height))
    img_array = img_to_array(img)
    img_array = img_array[:, :, 0]
    for c in range(nClasses):
        seg_labels[:, :, c] = (img_array == c).astype(int)
    return seg_labels

images = os.listdir(dir_img)
images.sort()
segmentations = os.listdir(dir_seg)
segmentations.sort()

input_width = 256
input_height = 256
output_width = 256
output_height = 256
X_train = []
Y_train = []

for im, seg in zip(images, segmentations):
    X_train.append(getImageArr(dir_img + im, input_width, input_height))
    Y_train.append(getSegmentationArr(dir_seg + seg, n_classes, output_width, output_height))

X_train, Y_train = np.array(X_train), np.array(Y_train)
print(X_train.shape, Y_train.shape)

images1 = os.listdir(dir_img1)
images1.sort()
segmentations1 = os.listdir(dir_seg1)
segmentations1.sort()

X_test = []
Y_test = []

for im, seg in zip(images1, segmentations1):
    X_test.append(getImageArr(dir_img1 + im, input_width, input_height))
    Y_test.append(getSegmentationArr(dir_seg1 + seg, n_classes, output_width, output_height))

X_test, Y_test = np.array(X_test), np.array(Y_test)
print(X_test.shape, Y_test.shape)

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)

    return x

def build_mobilenetv2_unet(input_shape):
    inputs = Input(shape=input_shape)

    encoder = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs, alpha=1.4)

    s1 = encoder.get_layer("input_1").output
    s2 = encoder.get_layer("block_1_expand_relu").output
    s3 = encoder.get_layer("block_3_expand_relu").output

    b1 = encoder.get_layer("block_6_expand_relu").output

    d1 = decoder_block(b1, s3, 256)
    d2 = decoder_block(d1, s2, 128)
    d3 = decoder_block(d2, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d3)

    model = Model(inputs, outputs, name="MobileNetV2_U-Net")
    return model

input_shape = (256, 256, 3)
model = build_mobilenetv2_unet(input_shape)

sgd = SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=sgd, metrics=['accuracy'])

Unet_trained = 'files_25/model.h5'
model.load_weights(Unet_trained)
print("Loaded model from disk")

output_dir = 'predict_test_25'
os.makedirs(output_dir, exist_ok=True)

for i in range(X_test.shape[0]):
    x = X_test[i]
    y1 = np.reshape(x, (1, x.shape[0], x.shape[1], x.shape[2]))
    preds = model.predict(y1)
    mask = preds > 0.5
    mask = mask.astype(np.int32) * 255
    z = np.reshape(mask, (mask.shape[1], mask.shape[2]))
    name = str(i) + '.png'
    cv2.imwrite(os.path.join(output_dir, name), z)
