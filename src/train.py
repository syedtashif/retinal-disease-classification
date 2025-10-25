"""
Main training script

"""

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from data.dataset import get_dataloaders
from models.complete_model import CompleteMSFMViTModel
from losses.focal_loss import FocalLoss
from losses.poly_loss import PolyBCELoss
from utils.metrics import compute_metrics, optimize_thresholds
from utils.transforms import get_train_transforms, get_test_transforms


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_loss_function(config):
    loss_type = config['loss']['type']
    if loss_type == 'focal':
        return FocalLoss(
            alpha=config['loss']['focal']['alpha'],
            gamma=config['loss']['focal']['gamma']
        )
    elif loss_type == 'poly1':
        return PolyBCELoss(
            epsilon=config['loss']['poly1']['epsilon']
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate(model, test_loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images)
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
    y_true = np.vstack(all_labels)
    y_prob = np.vstack(all_probs)
    return y_true, y_prob


def main(args):
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = get_train_transforms(config['training']['img_size'])
    test_transform = get_test_transforms(config['training']['img_size'])

    train_loader, test_loader = get_dataloaders(
        config['data']['train_csv'],
        config['data']['test_csv'],
        config['data']['images_dir'],
        train_transform,
        test_transform,
        config['disease_names'],
        config['training']['batch_size'],
        config['training']['num_workers']
    )

    model = CompleteMSFMViTModel(
        num_classes=config['model']['num_classes'],
        embed_dim=config['model']['embed_dim']
    ).to(device)

    criterion = get_loss_function(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config['scheduler']['mode'],
        factor=config['scheduler']['factor'],
        patience=config['scheduler']['patience'],
        verbose=True
    )

    best_model_score = 0
    epochs_no_improve = 0

    print("🚀 Starting Training")
    print("=" * 60)

    for epoch in range(1, config['training']['epochs'] + 1):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_prob = evaluate(model, test_loader, device)
        thresholds = optimize_thresholds(y_true, y_prob)
        metrics = compute_metrics(y_true, y_prob)

        print(f"Epoch {epoch}/{config['training']['epochs']} | "
              f"Loss: {avg_loss:.4f} | "
              f"ML F1: {metrics['ML_F1']:.4f} | "
              f"ML mAP: {metrics['ML_mAP']:.4f} | "
              f"ML AUC: {metrics['ML_AUC']:.4f} | "
              f"Model Score: {metrics['Model_Score']:.4f}")

        scheduler.step(metrics['Model_Score'])

        if metrics['Model_Score'] > best_model_score:
            best_model_score = metrics['Model_Score']
            epochs_no_improve = 0
            torch.save(model.state_dict(), config['output']['model_save_path'])
            print(f"✅ Best model saved! Score: {best_model_score:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config['training']['patience']:
            print("⚠️ Early stopping triggered")
            break

    print(f"\n🎉 Training Complete! Best Model Score: {best_model_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train retinal disease classification model")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    main(args)
