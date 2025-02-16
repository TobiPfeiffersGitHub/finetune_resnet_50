import os
import kagglehub
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from collections import Counter
import torch

class DataLoaderHandler:
    def __init__(self, dataset_name="pacificrm/skindiseasedataset", batch_size=32, input_size=256, val_split=0.2):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.input_size = input_size
        self.val_split = val_split
        self.dataset_path = self.download_dataset()
    
    def download_dataset(self):
        """Downloads the dataset using KaggleHub."""
        path = kagglehub.dataset_download(self.dataset_name)
        dataset_path = f"{path}/SkinDisease/SkinDisease/train"
        return dataset_path

    def get_data_loaders(self, exclude_classes=[]):
        """Returns train and validation DataLoaders while excluding specified classes."""
        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        all_classes = os.listdir(self.dataset_path)
        included_classes = [cls for cls in all_classes if cls not in exclude_classes]

        dataset = ImageFolder(self.dataset_path, transform=transform)
        dataset.samples = [(s, included_classes.index(c)) for s, c in dataset.samples if c in included_classes]
        dataset.class_to_idx = {cls: idx for idx, cls in enumerate(included_classes)}

        val_size = int(self.val_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Handle class imbalance
        target_list = [dataset.targets[i] for i in train_dataset.indices]
        class_counts = Counter(target_list)
        class_weights = {c: 1.0 / class_counts[c] for c in class_counts}
        sample_weights = [class_weights[label] for label in target_list]

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, included_classes
