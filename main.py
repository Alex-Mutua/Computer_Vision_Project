import argparse
import tensorflow as tf
import torch
from models.cnn_tensorflow import get_pretrained_model as get_tensorflow_model
from models.cnn_pytorch import get_pretrained_model as get_pytorch_model
from models.train_tensorflow import Trainer as TensorFlowTrainer
from models.train_pytorch import Trainer as PyTorchTrainer
from utils import prep_tensorflow, prep_pytorch

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Classification")
    parser.add_argument('--backend', type=str, default='tensorflow', choices=['pytorch', 'tensorflow'], help='AI backend to use')
    parser.add_argument('--plot', action='store_true', help='Enable plotting of training history')
    args = parser.parse_args()

    print(f"Selected backend: {args.backend}")

    if args.backend == 'tensorflow':
        print("Loading TensorFlow model via get_tensorflow_model()...")
        try:
            model = tf.keras.models.load_model("alex_model.tensorflow")
            print("Loaded existing TensorFlow model.")
        except Exception as e:
            print(f"Failed to load model: {str(e)}. Initializing new model.")
            model = get_tensorflow_model(num_classes=4)  # 4 classes for brain tumors
            print("Initialized new TensorFlow model.")

        print(f"Using prep_tensorflow from: {prep_tensorflow.__file__}")
        print("Loading TensorFlow data via prep_tensorflow.get_data()...")
        try:
            train_data, val_data = prep_tensorflow.get_data()
            print(f"TensorFlow data types: train_data={type(train_data)}, val_data={type(val_data)}")
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

        # Verify data type
        if not isinstance(train_data, tf.data.Dataset):
            raise ValueError(f"Expected tf.data.Dataset, got {type(train_data)}")

        # Training parameters
        lr = 0.001
        epochs = 10

        print(f"Starting training for {epochs} epochs...")
        trainer = TensorFlowTrainer(model, train_data, val_data, lr, epochs)
        trainer.train(save=True, filename="alex_model.tensorflow", plot=args.plot)
        accuracy, loss = trainer.evaluate()
        print(f"Training completed! Accuracy: {accuracy:.2%}, Loss: {loss:.4f}")

    else:  # pytorch
        print("Loading PyTorch model via get_pytorch_model()...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_pytorch_model(num_classes=4).to(device)  # 4 classes for brain tumors
        try:
            model.load_state_dict(torch.load("alex.torch", map_location=device))
            model.eval()
            print("Loaded existing PyTorch model.")
        except FileNotFoundError:
            print("Initialized new PyTorch model.")

        print(f"Using prep_pytorch from: {prep_pytorch.__file__}")
        print("Loading PyTorch data via prep_pytorch.get_data()...")
        try:
            train_data, val_data = prep_pytorch.get_data()
            print(f"PyTorch data types: train_data={type(train_data)}, val_data={type(val_data)}")
        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

        # Training parameters
        lr = 0.001
        wd = 0.0001
        epochs = 10

        print(f"Starting training for {epochs} epochs...")
        trainer = PyTorchTrainer(model, train_data, val_data, lr, wd, epochs, device)
        trainer.train(save=True, plot=args.plot, use_streamlit=False)
        accuracy, loss = trainer.evaluate()
        print(f"Training completed! Accuracy: {accuracy:.2%}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()