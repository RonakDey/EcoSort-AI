import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # oneDNN warning দূর করার জন্য
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # অপ্রয়োজনীয় warning কমানোর জন্য

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

print("✅ TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Dataset path
data_dir = 'data'
IMG_SIZE = 224
BATCH_SIZE = 32

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

class_names = list(train_generator.class_indices.keys())
print("Classes:", class_names)
print("Training images:", train_generator.samples)
print("Validation images:", validation_generator.samples)

# Transfer Learning with MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False   # প্রথমে freeze

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)            # Overfitting কমানোর জন্য
predictions = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping (যাতে অপ্রয়োজনে ট্রেনিং না চলে)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
epochs = 10   # শুরুতে ১০ রাখো, পরে বাড়াতে পারো
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[early_stop]
)





# Save model
model.save('model.h5')
print("✅ মডেল সেভ হয়েছে: model.h5")

# Optional: Save in .keras format (নতুন TF-এ ভালো)
model.save('model.keras')
print("✅ মডেল সেভ হয়েছে: model.keras")