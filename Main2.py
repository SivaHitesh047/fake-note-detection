#!/usr/bin/env python
# coding: utf-8

# In[33]:


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import pickle
import matplotlib.pyplot as plt


# In[34]:


train_dir = "Dataset/Training"
validation_dir = "Dataset/Validation"
test_dir = "Dataset/Testing"


# In[35]:


img_height = 300
img_width = 300
batch_size = 16


# In[36]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# In[37]:


base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze layers in the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)
finetune_model = Model(inputs=base_model.input, outputs=predictions)


# In[38]:


# Compile the model
finetune_model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10)
def schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)
lr_schedule = LearningRateScheduler(schedule)


# In[39]:


history = finetune_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stop, lr_schedule])

# Save the model weights
finetune_model.save_weights("Final_model.h5")


# In[40]:


def testing_func(test_img):
    test_img = np.expand_dims(test_img, axis=0) / 255.0
    output = finetune_model.predict(test_img)
    if output[0][0] > output[0][1]:
        return "Fake"
    else:
        return "Real"


# In[41]:


test_img = load_img("Dataset/Testing/Fake3.png", target_size=(img_height, img_width))
result = testing_func(test_img)
print(result)
plt.imshow(test_img)


# In[42]:


test_img = load_img("Dataset/Validation/Fake/10f.jpg", target_size=(img_height, img_width))
plt.imshow(test_img)
result = testing_func(test_img)
print(result)


# In[ ]:




