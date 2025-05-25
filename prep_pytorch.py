import torch
from torchvision import datasets, transforms

def get_data(batch_size=32):
    """Load brain tumor dataset from data/training/ and data/testing/ for PyTorch"""
    try:
        # Define transforms with augmentation for training
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder('data/training/', transform=train_transforms)
        test_dataset = datasets.ImageFolder('data/testing/', transform=test_transforms)
        
        # Verify class names
        expected_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        if train_dataset.classes != expected_classes:
            raise ValueError(f"Expected classes {expected_classes}, got {train_dataset.classes}")
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
    except Exception as e:
        raise ValueError(f"Failed to load PyTorch datasets: {str(e)}")
    
    return train_loader, test_loader