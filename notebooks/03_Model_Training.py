#!/usr/bin/env python
# coding: utf-8

# # LungsNet: Model Training
# Training DenseNet121 for Pneumonia Detection.

# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os


# ## 1. Data Loading & Augmentation

# In[ ]:


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# Paths
TRAIN_DIR = "../data/balanced/train"
TEST_DIR = "../data/raw/chest_xray/test"
# Note: We use the balanced TRAIN_DIR for both train and validation split

# Add validation_split=0.2 to create a 20% validation set from training data
train_val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training Generator (Subset: Training)
train_generator = train_val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

# Validation Generator (Subset: Validation, uses same TRAIN_DIR)
val_generator = train_val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Test Generator (Independent Test Set)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)


# ## 2. Build Model (DenseNet121)

# In[ ]:


base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
# for layer in base_model.layers:
#     layer.trainable = False
# Or fine-tune (optional)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# ## 3. Train Model

# In[ ]:


EPOCHS = 10

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)


# ## 4. Save Model

# In[ ]:


model.save('../models/lungsnet_densenet121.h5')
print("Model saved successfully!")

