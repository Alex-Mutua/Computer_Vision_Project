import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

class Trainer:
    def __init__(self, model, train_dataloader, test_dataloader, lr, wd, epochs, device):
        self.epochs = epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr, weight_decay=wd)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []

    def train(self, save=False, plot=False, use_streamlit=True):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)

            for batch in progress_bar:
                input_datas, labels = batch
                input_datas, labels = input_datas.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_datas)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Accuracy calculation
                _, preds = outputs.max(1)
                correct = (preds == labels).sum().item()
                total = labels.size(0)

                total_correct += correct
                total_samples += total
                total_loss += loss.item()

                batch_accuracy = 100.0 * correct / total
                average_accuracy = 100.0 * total_correct / total_samples
                average_loss = total_loss / total_samples

                progress_bar.set_postfix({
                    'Batch Acc': f'{batch_accuracy:.2f}%',
                    'Avg Acc': f'{average_accuracy:.2f}%',
                    'Loss': f'{average_loss:.4f}'
                })

            print(f"Epoch {epoch + 1}/{self.epochs} finished — Avg Accuracy: {average_accuracy:.2f}%, Avg Loss: {average_loss:.4f}")

            self.train_acc.append(average_accuracy)
            self.train_loss.append(average_loss)

            # Validation
            val_accuracy, val_loss = self.evaluate()
            self.val_acc.append(val_accuracy)
            self.val_loss.append(val_loss)
            print(f"Validation — Accuracy: {val_accuracy:.2f}%, Loss: {val_loss:.4f}")

        if save:
            torch.save(self.model.state_dict(), "alex.torch")
        if plot:
            self.plot_training_history(use_streamlit=use_streamlit)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for inputs, labels in tqdm(self.test_dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples

        return accuracy, avg_loss

    def plot_training_history(self, use_streamlit=True):
        epochs = range(1, len(self.train_loss) + 1)

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

        plt.title('Training and Validation Loss and Accuracy (PyTorch)')
        fig.tight_layout()

        if use_streamlit:
            st.pyplot(fig)
        else:
            plt.show()