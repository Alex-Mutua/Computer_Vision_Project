import tensorflow as tf

def get_data(batch_size=32):
    """Load brain tumor dataset from data/training/ and data/testing/ for TensorFlow"""
    try:
        # Load training dataset with augmentation
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            'data/training/',
            labels='inferred',
            label_mode='categorical',
            class_names=['glioma', 'meningioma', 'notumor', 'pituitary'],
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=True
        )
        
        # Load test dataset
        test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            'data/testing/',
            labels='inferred',
            label_mode='categorical',
            class_names=['glioma', 'meningioma', 'notumor', 'pituitary'],
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=False
        )
        
        # Apply preprocessing (ResNet normalization)
        def preprocess(image, label):
            image = tf.keras.applications.resnet.preprocess_input(image)
            return image, label
        
        train_dataset = train_dataset.map(preprocess)
        test_dataset = test_dataset.map(preprocess)
        
        # Apply data augmentation for training
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2)
        ])
        
        train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
        
    except Exception as e:
        raise ValueError(f"Failed to load TensorFlow datasets: {str(e)}")
    
    return train_dataset, test_dataset