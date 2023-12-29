import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras import mixed_precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# mixed_precision.set_policy(policy)

import matplotlib.pyplot as plt


# Function to create file paths for images and masks
def create_file_paths(base_path, data_folder='data', rgb_folder='CameraRGB', mask_folder='CameraMask'):
    image_path = os.path.join(base_path, data_folder, rgb_folder)
    mask_path = os.path.join(base_path, data_folder, mask_folder)
    
    image_list_orig = os.listdir(image_path)
    image_list = [os.path.join(image_path, i) for i in image_list_orig]
    mask_list = [os.path.join(mask_path, i) for i in image_list_orig]
    
    return image_list, mask_list

# Function to process image and mask paths
def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    
    return img, mask

# Function to preprocess images and masks
def preprocess(image, mask):
    input_image = tf.image.resize(image, (96*2, 128*2), method='bilinear')
    input_mask = tf.image.resize(mask, (96*2, 128*2), method='bilinear')

    return input_image, input_mask

# Function to create a convolutional block
def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob, input_shape=(n_filters,))(conv)
        
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv)
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection

# Function to create an upsampling block
def upsampling_block(expansive_input, contractive_input, n_filters=32):
    up = Conv2DTranspose(n_filters, 3, strides=(2, 2), padding='same')(expansive_input)
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    return conv

# Function to create the U-Net model
def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    inputs = tf.keras.layers.Input(input_size, dtype=tf.float32)
    cblock1 = conv_block(inputs, n_filters)
    cblock2 = conv_block(cblock1[0], n_filters*2)
    cblock3 = conv_block(cblock2[0], n_filters*4)
    cblock4 = conv_block(cblock3[0], n_filters*8, dropout_prob=0.3) 
    cblock5 = conv_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 
    ublock6 = upsampling_block(cblock5[0], cblock4[1],  n_filters*8)
    ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters*4)
    ublock8 = upsampling_block(ublock7, cblock2[1],  n_filters*2)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)

    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
    conv10 = tf.keras.layers.Conv2D(n_classes, 1, padding='same', dtype='float32')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model

# Function to display images
def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# Constants and paths
img_height = 96*2
img_width = 128*2
num_channels = 3
base_path = ''
data_folder = 'data'
rgb_folder = 'CameraRGB'
mask_folder = 'CameraMask'

# Create file paths
image_list, mask_list = create_file_paths(base_path, data_folder, rgb_folder, mask_folder)

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))

# Process dataset
image_ds = dataset.map(process_path)
processed_image_ds = image_ds.map(preprocess)

# Build and compile the U-Net model
unet = unet_model((img_height, img_width, num_channels))
unet.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training parameters
EPOCHS = 10
BUFFER_SIZE = 500
VAL_SUBSPLITS = 5
BATCH_SIZE = 16

# Create and train the dataset
train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
model_history = unet.fit(train_dataset, epochs=EPOCHS)

# Function to create a mask from model predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# Display training history
plt.plot(model_history.history["accuracy"])

# Function to show predictions
def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = unet.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    # else:
    #     display([sample_image, sample_mask,
    #              create_mask(unet.predict(sample_image[tf.newaxis, ...]))])

# Show predictions on the training dataset
show_predictions(train_dataset, 6)

# Save the entire model to a single HDF5 file
model_save_path = 'model.h5'
unet.save(model_save_path)

