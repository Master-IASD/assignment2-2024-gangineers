#!/usr/bin/env python3
import os
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
from torchvision.utils import save_image

# Define namedtuples for structured data
from collections import namedtuple
Manifold = namedtuple('Manifold', ['features', 'radii'])
PrecisionAndRecall = namedtuple('PrecisionAndRecall', ['precision', 'recall'])

def convert_mnist_to_images(output_dir, num_images=100):
    """
    Convert MNIST dataset images to PNG format and save them to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_data = datasets.MNIST(root='data/MNIST', train=True, transform=transform, download=True)

    for i in tqdm(range(num_images), desc='Converting MNIST to Images'):
        img, label = mnist_data[i]
        save_image(img, os.path.join(output_dir, f'real_{i}.png'))
    print(f"Converted {num_images} MNIST images to PNG format in '{output_dir}'.")

class ImageDataset(Dataset):
    """
    Custom Dataset for loading images from a directory.
    """
    def __init__(self, root, transform=None, num_samples=-1):
        self.fnames = glob(os.path.join(root, '**', '*.jpg'), recursive=True) + \
                      glob(os.path.join(root, '**', '*.png'), recursive=True) + \
                      glob(os.path.join(root, '**', '*.jpeg'), recursive=True)
        if num_samples > 0:
            self.fnames = self.fnames[:num_samples]
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        image_path = self.fnames[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

class PrecisionRecallEvaluator:
    """
    Class to compute Precision and Recall for GANs using feature embeddings.
    """
    def __init__(self, device='cuda', batch_size=64, k=3, num_workers=4):
        self.device = device
        self.batch_size = batch_size
        self.k = k
        self.num_workers = num_workers
        self.model = self._load_model()

    def _load_model(self):
        """
        Load a pretrained InceptionV3 model for feature extraction.
        """
        print("Loading InceptionV3 model for feature extraction...")
        model = models.inception_v3(pretrained=True, transform_input=False)
        # Remove the final classification layer
        model.fc = torch.nn.Identity()
        model.to(self.device)
        model.eval()
        print("InceptionV3 model loaded.")
        return model

    def _extract_features(self, dataloader):
        """
        Extract features from images using the pretrained model.
        """
        features = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Extracting Features'):
                batch = batch.to(self.device)
                # InceptionV3 expects 299x299 images
                if batch.shape[2] != 299 or batch.shape[3] != 299:
                    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                feats = self.model(batch)
                features.append(feats.cpu())
        features = torch.cat(features, dim=0)
        return features

    def compute_manifold(self, dataloader):
        """
        Compute the manifold (features and radii) for a given dataset.
        """
        features = self._extract_features(dataloader)
        # Compute radii based on k-NN
        radii = self._compute_radii(features)
        return Manifold(features=features, radii=radii)

    def _compute_radii(self, features):
        """
        Compute radii for each feature vector based on k-NN distances.
        """
        print("Computing radii for k-NN...")
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        # Compute cosine similarity
        similarity = torch.mm(features_norm, features_norm.t())
        # Convert similarity to distance
        distance = 1 - similarity
        # Fill diagonal with large value to exclude self-distance
        distance.fill_diagonal_(float('inf'))
        # Find k-nearest neighbors
        radii, _ = torch.topk(distance, self.k, dim=1, largest=False)
        radii = radii[:, -1]  # The k-th smallest distance
        return radii.cpu()

    def compute_precision_recall(self, real_loader, fake_loader):
        """
        Compute Precision and Recall between real and fake datasets.
        """
        print("Computing manifold for real images...")
        real_manifold = self.compute_manifold(real_loader)

        print("Computing manifold for fake images...")
        fake_manifold = self.compute_manifold(fake_loader)

        # Normalize features
        real_features = F.normalize(real_manifold.features, p=2, dim=1).to(self.device)  # (N_real, D)
        fake_features = F.normalize(fake_manifold.features, p=2, dim=1).to(self.device)  # (N_fake, D)

        # Compute pairwise distances
        print("Computing pairwise distances...")
        # Efficient computation using batch processing to avoid memory issues
        N_real = real_features.size(0)
        N_fake = fake_features.size(0)
        batch_size = self.batch_size
        precision_count = 0
        recall_count = 0

        # Precision: How much of the fake data lies within the real data manifold
        print("Calculating Precision...")
        for i in tqdm(range(0, N_fake, batch_size), desc='Precision Calculation'):
            end = min(i + batch_size, N_fake)
            batch = fake_features[i:end]  # (batch_size, D)
            # Compute cosine similarity
            similarity = torch.mm(batch, real_features.t())  # (batch_size, N_real)
            distance = 1 - similarity  # (batch_size, N_real)
            # Compare with real radii
            radii = real_manifold.radii.unsqueeze(0).to(self.device)  # (1, N_real)
            within_radius = (distance < radii)  # (batch_size, N_real)
            precision_count += within_radius.any(dim=1).sum().item()

        precision = precision_count / N_fake

        # Recall: How much of the real data manifold is covered by fake data
        print("Calculating Recall...")
        for i in tqdm(range(0, N_real, batch_size), desc='Recall Calculation'):
            end = min(i + batch_size, N_real)
            batch = real_features[i:end]  # (batch_size, D)
            # Compute cosine similarity
            similarity = torch.mm(batch, fake_features.t())  # (batch_size, N_fake)
            distance = 1 - similarity  # (batch_size, N_fake)
            # Compare with fake radii
            radii = fake_manifold.radii.unsqueeze(0).to(self.device)  # (1, N_fake)
            within_radius = (distance < radii)  # (batch_size, N_fake)
            recall_count += within_radius.any(dim=1).sum().item()

        recall = recall_count / N_real

        return PrecisionAndRecall(precision=precision, recall=recall)

def main():
    parser = argparse.ArgumentParser(description='Efficient Precision and Recall Calculation for GANs')
    parser.add_argument('path_real', type=str, help='Path to the real images directory')
    parser.add_argument('path_fake', type=str, help='Path to the fake images directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for feature extraction')
    parser.add_argument('--k', type=int, default=3, help='k value for k-NN radius computation')
    parser.add_argument('--num_samples_real', type=int, default=-1, help='Number of real samples to use (-1 for all)')
    parser.add_argument('--num_samples_fake', type=int, default=-1, help='Number of fake samples to use (-1 for all)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--conversion', '-c', action='store_true', help='Convert MNIST binary files to images')
    parser.add_argument('--num_convert', type=int, default=1000, help='Number of MNIST images to convert')
    args = parser.parse_args()

    # Handle conversion flag
    if args.conversion:
        print("Converting MNIST binary files to images...")
        convert_mnist_to_images(output_dir=args.path_real, num_images=args.num_convert)
        print("Conversion completed.")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # InceptionV3 expects 299x299 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])

    # Create datasets and dataloaders
    print("Loading real images...")
    real_dataset = ImageDataset(root=args.path_real, transform=transform, num_samples=args.num_samples_real)
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print("Loading fake images...")
    fake_dataset = ImageDataset(root=args.path_fake, transform=transform, num_samples=args.num_samples_fake)
    fake_loader = DataLoader(fake_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Initialize evaluator
    evaluator = PrecisionRecallEvaluator(device=args.device, batch_size=args.batch_size, k=args.k,
                                        num_workers=args.num_workers)

    # Compute Precision and Recall
    print("Starting Precision and Recall computation...")
    precision_recall = evaluator.compute_precision_recall(real_loader, fake_loader)
    print(f"Precision: {precision_recall.precision:.4f}")
    print(f"Recall: {precision_recall.recall:.4f}")

    # Optionally, save the results
    results_path = 'precision_recall_results.txt'
    with open(results_path, 'w') as f:
        f.write(f"Precision: {precision_recall.precision:.4f}\n")
        f.write(f"Recall: {precision_recall.recall:.4f}\n")
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()
