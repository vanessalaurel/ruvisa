import argparse
import glob
import json
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


ACNE04_ROOT = "/home/vanessa/acne04"  # Update if your ACNE04 dataset lives elsewhere
TRAINVAL_INDEX = "NNEW_trainval_*.txt"
TEST_INDEX = "NNEW_test_*.txt"
CLASS_NAMES = ["level0", "level1", "level2", "level3"]  # severity grades 0-3
LESION_NAMES = ["non_inflammatory", "inflammatory", "cystic"]
DEFAULT_RUN_DIR = "acne_severity_runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Acne04Dataset(Dataset):
    """ACNE04 severity dataset loader using index files (e.g., NNEW_trainval.txt)."""

    def __init__(self, root_dir: str, index_pattern: str, transform=None):
        index_paths = self._resolve_index_paths(root_dir, index_pattern)
        if not index_paths:
            raise FileNotFoundError(f"No index files matched pattern '{index_pattern}' in {root_dir}")

        self.root_dir = root_dir
        self.transform = transform
        filenames, levels, lesions = [], [], []
        for path in index_paths:
            f, l, le = self._load_index(path)
            filenames.append(f)
            levels.append(l)
            lesions.append(le)

        if filenames:
            self.filenames = np.concatenate(filenames)
            self.levels = np.concatenate(levels)
            self.lesions = np.concatenate(lesions)
        else:
            raise RuntimeError(f"No entries found in index pattern '{index_pattern}'")

    @staticmethod
    def _resolve_index_paths(root_dir: str, index_pattern: str):
        """Return sorted list of index files matching the pattern."""
        candidates = [
            glob.glob(index_pattern),
            glob.glob(os.path.join(root_dir, index_pattern)),
            glob.glob(os.path.join(root_dir, "Classification", index_pattern)),
            glob.glob(os.path.join(root_dir, "classification", index_pattern)),
            glob.glob(os.path.join(root_dir, "metadata", index_pattern)),
        ]
        paths = sorted({os.path.abspath(p) for group in candidates for p in group})
        return paths

    @staticmethod
    def _load_index(index_path: str):
        filenames, levels, lesions = [], [], []
        with open(index_path, "r") as fp:
            for raw in fp:
                raw = raw.strip()
                if not raw:
                    continue
                parts = raw.split()
                if len(parts) < 3:
                    raise ValueError(f"Malformed line in {index_path}: '{raw}'")
                fname, level, lesion = parts[:3]
                filenames.append(fname)
                levels.append(int(level))
                lesions.append(int(lesion))
        return np.array(filenames), np.array(levels), np.array(lesions)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        from PIL import Image  # Lazy import to avoid global dependency when unused

        img_rel_path = self.filenames[idx]
        # ACNE04 images typically live in Classification/JPEGImages
        candidates = [
            os.path.join(self.root_dir, img_rel_path),
            os.path.join(self.root_dir, "Classification", "JPEGImages", img_rel_path),
            os.path.join(self.root_dir, "classification", "JPEGImages", img_rel_path),
            os.path.join(self.root_dir, "JPEGImages", img_rel_path),
        ]
        img_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {img_rel_path}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        level = torch.tensor(self.levels[idx], dtype=torch.long)
        lesion = torch.tensor(self.lesions[idx], dtype=torch.long)
        return image, level, lesion, img_rel_path


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(labels: np.ndarray, val_ratio: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices_by_class = defaultdict(list)
    for idx, label in enumerate(labels):
        indices_by_class[int(label)].append(idx)

    train_indices, val_indices = [], []
    for label, idxs in indices_by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        cut = max(1, int(len(idxs) * (1 - val_ratio)))
        train_indices.extend(idxs[:cut])
        val_indices.extend(idxs[cut:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def create_dataloaders(root: str, train_index: str, test_index: str, batch_size: int = 32,
                       val_ratio: float = 0.1, num_workers: int = 4):
    train_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_train = Acne04Dataset(root, train_index, transform=train_tfms)
    train_idx, val_idx = stratified_split(full_train.levels, val_ratio=val_ratio)
    val_dataset = Acne04Dataset(root, train_index, transform=eval_tfms)

    train_subset = torch.utils.data.Subset(full_train, train_idx)
    val_subset = torch.utils.data.Subset(val_dataset, val_idx)

    test_dataset = Acne04Dataset(root, test_index, transform=eval_tfms)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, eval_tfms


def build_model(num_classes: int = 4):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels, _, _ in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


def compute_metrics(preds: np.ndarray, labels: np.ndarray):
    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
    cm = confusion_matrix(labels, preds)
    return report, cm.tolist()


def train(args):
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("🚀 Training ACNE04 Severity Classifier")
    print("=" * 60)
    print(f"Dataset root: {args.data_root}")
    print(f"Train index:  {args.train_index}")
    print(f"Test index:   {args.test_index}")
    print(f"Output dir:   {run_dir}")
    print(f"Device:       {DEVICE}")

    train_loader, val_loader, test_loader, eval_tfms = create_dataloaders(
        root=args.data_root,
        train_index=args.train_index,
        test_index=args.test_index,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.workers,
    )

    model = build_model(num_classes=len(CLASS_NAMES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for images, labels, _, _ in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"loss": loss.item(), "acc": correct / max(total, 1)})

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc + args.min_delta:
            best_val_acc = val_acc
            patience_counter = 0
            best_path = os.path.join(run_dir, "best_model.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "lesion_names": LESION_NAMES,
                    "config": vars(args),
                },
                best_path,
            )
            print(f"✨ New best model saved to {best_path} (val_acc={best_val_acc:.3f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("⏹️  Early stopping triggered")
            break

    # Save training history
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate best model on test set
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
    report, cm = compute_metrics(test_preds, test_labels)

    print("Test accuracy:", f"{test_acc:.3f}")
    print("\nClassification report:\n", report)
    print("Confusion matrix (rows=true, cols=pred):")
    for row in cm:
        print(row)

    summary = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Training complete. Results saved to {run_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACNE04 severity classification model")
    parser.add_argument("--data-root", default=ACNE04_ROOT,
                        help="Path to ACNE04 dataset root (contains Classification/JPEGImages and index files)")
    parser.add_argument("--train-index", default=TRAINVAL_INDEX,
                        help="Glob pattern for training/validation indexes (e.g., NNEW_trainval_*.txt)")
    parser.add_argument("--test-index", default=TEST_INDEX,
                        help="Glob pattern for testing indexes (e.g., NNEW_test_*.txt)")
    parser.add_argument("--output-dir", default=DEFAULT_RUN_DIR,
                        help="Directory to store checkpoints and logs")
    parser.add_argument("--run-name", default=None,
                        help="Optional name for this training run")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of train index reserved for validation")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=1e-3,
                        help="Minimum improvement in val_acc to reset patience")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
