import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

# -------------------------------
# ✅ SETUP PARAMETERS
# -------------------------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 70
DATA_DIR = r"D:\Projects\computer vision\version 2.4\Eurosat Dataset"

# Define the paths for your new train and test folders
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

ALLOWED_FORMATS = (".jpg", ".jpeg", ".png")

print("🔍 Cleaning dataset (removing unsupported files)...")
for subdir, _, files in os.walk(DATA_DIR):
    for f in files:
        if not f.lower().endswith(ALLOWED_FORMATS):
            full_path = os.path.join(subdir, f)
            print(f"❌ Removing unsupported file: {full_path}")
            os.remove(full_path)
print("✅ Dataset cleaned.")

# Get class names from the train directory
CLASS_NAMES = sorted([name for name in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, name))])
NUM_CLASSES = len(CLASS_NAMES)
print(f"Found {NUM_CLASSES} classes: {CLASS_NAMES}")

# -------------------------------
# ✅ DATA AUGMENTATION AND GENERATORS
# -------------------------------
# Training and validation generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Use 20% of the training data for validation
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Test generator should NOT have any data augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator points to the 'train' folder
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    classes=CLASS_NAMES,
    color_mode='rgb'
)

# Validation data generator also points to the 'train' folder
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    classes=CLASS_NAMES,
    color_mode='rgb'
)

# Test data generator points directly to the 'test' folder
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    color_mode='rgb',
    shuffle=False
)

# -------------------------------
# ✅ BUILD AND TRAIN RESNET-BASED MODEL
# -------------------------------
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
checkpoint = ModelCheckpoint("eurosat_best_model.keras", monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

print("\n🚀 Stage 1: Training only the custom top layers...")
history_stage1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

print("\n🚀 Stage 2: Fine-tuning the entire model with a low learning rate...")
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_stage2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# -------------------------------
# ✅ EVALUATE ON TEST SET
# -------------------------------
print("\n🔬 Evaluating the final model on the test set...")
model.load_weights("eurosat_best_model.keras")
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_accuracy:.4f}")

# -------------------------------
# ✅ SAVE FINAL MODEL
# -------------------------------
model.save("eurosat_resnet50_fine-tuned.keras")
print("✅ Model saved as eurosat_resnet50_fine-tuned.keras")

# -------------------------------
# ✅ VISUALIZE RESULTS
# -------------------------------
hist_accuracy = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
hist_val_accuracy = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
hist_loss = history_stage1.history['loss'] + history_stage2.history['loss']
hist_val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_accuracy, label='Train Acc')
plt.plot(hist_val_accuracy, label='Val Acc')
plt.title("Accuracy")
plt.legend()
plt.axvline(x=len(history_stage1.history['accuracy']), color='r', linestyle='--', label='Fine-tuning starts')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_loss, label='Train Loss')
plt.plot(hist_val_loss, label='Val Loss')
plt.title("Loss")
plt.legend()
plt.axvline(x=len(history_stage1.history['loss']), color='r', linestyle='--', label='Fine-tuning starts')
plt.legend()

plt.tight_layout()
plt.show()