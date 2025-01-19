import keras
from keras import backend as K
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.regularizers import l2, l1, l1_l2

#This code creates 3 CNNs that form an ensemble model


#I added these functions so I can view the metrics of the CNNs in greater depth
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    specificity = true_negatives / (possible_negatives + K.epsilon())
    return specificity

def build_cnn():
    model = keras.Sequential([
        #All values seclected from the results of KerasTuner
        #This example is to build the second best architecture for the system
        #This was not selected due to the lack of effiency from the high amount of parameters
        # First Convolutional Layer
        keras.layers.Conv2D(
            filters=80,  
            kernel_size=3,  
            use_bias=False,  
            input_shape=(64, 64, 3),
            kernel_regularizer=l2(0.05) 
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'), 
        keras.layers.Dropout(0.2), 
        
        # Second Convolutional Layer
        keras.layers.Conv2D(
            filters=64,  
            kernel_size=5,  
            use_bias=False,  
            kernel_regularizer=l1(0.05) 
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'), 
        keras.layers.Dropout(0.3),  

        # Flatten
        keras.layers.Flatten(),
        
        # Dense Layer
        keras.layers.Dense(
            224, 
            use_bias=False,
            kernel_regularizer=l1_l2(0.05)
        ),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('elu'), 
        keras.layers.Dropout(0.1),  
        # Output Layer
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01), 
        loss='categorical_crossentropy',
        #Here I add the metrics I defined
        metrics=['accuracy', recall, precision, f1, specificity]
    )
    
    return model


def load_images(data_directory, target_size=(64, 64)):
    images = []
    labels = []
    class_dirs = sorted(os.listdir(data_directory))
    label_dict = {class_dir: i for i, class_dir in enumerate(class_dirs)}

    for class_dir in class_dirs:
        class_dir_path = os.path.join(data_directory, class_dir)
        for img_file in os.listdir(class_dir_path):
            img_path = os.path.join(class_dir_path, img_file)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label_dict[class_dir])

    images = np.array(images)
    labels = keras.utils.to_categorical(labels, num_classes=len(class_dirs))
    return images, labels

def train_model(images, labels, model, epochs=30, validation_images=None, validation_labels=None):
    #Real time data augmentation to reduce overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow(images, labels, batch_size=16)

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow(validation_images, validation_labels, batch_size=16)

    if validation_images is not None and validation_labels is not None:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator
        )
    else:
        history = model.fit(train_generator, epochs=epochs)
    
    return history


def plot_metrics(history, model_index):
    plt.figure(figsize=(15, 10))

    # Plot training accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.title(f'Model {model_index} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model {model_index} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'smallermodel_{model_index}_training_validation_metrics.png')
    plt.close()



images, labels = load_images('dataset/Training')

# Split data into three subsets
indices = np.arange(len(images))
np.random.shuffle(indices)
subset_size = len(indices) // 3
subset_indices = [indices[:subset_size], indices[subset_size:2*subset_size], indices[2*subset_size:]]


validation_images, validation_labels = load_images('dataset/Validation')

# Train three models and plot their training metrics
models = [build_cnn() for _ in range(3)]
histories = []
# Example call within your training loop:
for i, model in enumerate(models):
    print(f"Training model {i+1}")
    subset_images = images[subset_indices[i]]
    subset_labels = labels[subset_indices[i]]
    history = train_model(subset_images, subset_labels, model, epochs=30, validation_images=validation_images, validation_labels=validation_labels)
    histories.append(history)
    plot_metrics(history, i+1)

        # Save the trained model
    model.save(f'smallermodel_{i+1}.h5')
    print(f'Model {i+1} saved as model_{i+1}.h5')


# Load Test data and evaluate ensemble predictions
validation_images, validation_labels = load_images('dataset/Testing')
predictions = [model.predict(validation_images / 255.0) for model in models]
average_predictions = np.mean(predictions, axis=0)
predicted_classes = np.argmax(average_predictions, axis=1)