import tensorflow as tf

def get_pretrained_model(num_classes=4):
    """Initialize a pretrained ResNet50 model for brain tumor classification"""
    # Load pretrained ResNet50 without top layer
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model