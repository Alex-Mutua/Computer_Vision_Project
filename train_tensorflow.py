import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
import os

class Trainer:
    def __init__(self, model, train_data, val_data, lr, epochs):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, save=False, filename="alex_model.tensorflow", plot=False):
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=self.epochs,
            verbose=2
        )

        self.train_loss = history.history['loss']
        self.train_acc = [v * 100 for v in history.history['accuracy']]
        self.val_loss = history.history['val_loss']
        self.val_acc = [v * 100 for v in history.history['val_accuracy']]

        if save:
            try:
                if os.path.exists(filename):
                    print(f"Warning: Overwriting existing model at '{filename}'.")
                self.model.save(filename)
                print(f"Model saved successfully to '{filename}'.")
            except Exception as e:
                print(f"Failed to save model: {str(e)}")

        if plot:
            self.plot_training_history()

    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.val_data, verbose=1)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%  |  Test Loss: {loss:.4f}")
        return accuracy, loss

    def plot_training_history(self):
        epochs = range(1, self.epochs + 1)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color_loss = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color_loss)
        ax1.plot(epochs, self.train_loss, color=color_loss, label='Training Loss')
        ax1.plot(epochs, self.val_loss, color=color_loss, linestyle='--', label='Validation Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        color_acc = 'tab:red'
        ax2.set_ylabel('Accuracy (%)', color=color_acc)
        ax2.plot(epochs, self.train_acc, color=color_acc, label='Training Accuracy')
        ax2.plot(epochs, self.val_acc, color=color_acc, linestyle='--', label='Validation Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)
        ax2.legend(loc='upper right')

        plt.title('Training and Validation Loss and Accuracy (TensorFlow)')
        fig.tight_layout()
        try:
            st.pyplot(fig)
        except NameError:
            plt.show()