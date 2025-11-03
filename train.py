# train.py - Following BOTRGCN_Original structure with enhanced metrics
from model import BotRGCN
from Dataset_Original import Twibot22  # Use original-style preprocessed data loader
from pathlib import Path
import torch
from torch import nn
from utils import accuracy, init_weights, infer_num_relations  # <— added infer_num_relations
import numpy as np
import json
import time
from datetime import datetime

from sklearn.metrics import (
    precision_score, recall_score, f1_score, matthews_corrcoef,
    precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_curve, auc, average_precision_score
)

# -------------------- Configuration --------------------
device = 'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Model save paths
MODEL_SAVE_PATH = "trained_bot_detection_model_optimized.pth"
BEST_MODEL_PATH = "best_bot_detection_model.pth"
TRAINING_LOG_PATH = "training_log.json"

# Root directory for preprocessed data (matching original)
root = './processed_data/'

# -------------------- Adaptive Hyperparameters Based on Dataset Size --------------------
def get_adaptive_hyperparams(num_samples):
    """
    Automatically adjust hyperparameters based on dataset size:
    - Small (<100k): Strong regularization (high dropout, edge drop, label smoothing)
    - Medium (100k-5M): Moderate regularization
    - Large (>5M): Light regularization (model can learn from abundant data)
    """
    print(f"\nDataset size: {num_samples:,} samples")

    if num_samples < 100_000:
        # Small dataset: Need strong regularization
        print("→ Using SMALL dataset hyperparameters (strong regularization)")
        return {
            "dropout": 0.5,
            "edge_drop_prob": 0.3,
            "label_smoothing": 0.15,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "epochs": 150,
            "patience": 20,
        }
    elif num_samples < 5_000_000:
        # Medium dataset: Moderate regularization
        print("→ Using MEDIUM dataset hyperparameters (moderate regularization)")
        return {
            "dropout": 0.4,
            "edge_drop_prob": 0.2,
            "label_smoothing": 0.1,
            "lr": 8e-5,
            "weight_decay": 8e-5,
            "epochs": 120,
            "patience": 18,
        }
    else:
        # Large dataset (>5M): Light regularization
        print("→ Using LARGE dataset hyperparameters (light regularization)")
        return {
            "dropout": 0.35,
            "edge_drop_prob": 0.15,
            "label_smoothing": 0.08,
            "lr": 5e-5,
            "weight_decay": 5e-5,
            "epochs": 100,
            "patience": 15,
        }

# We'll determine actual hyperparams after loading dataset
# Placeholder values for now
embedding_size = 256

# Base hyperparameters (will be updated after dataset loading)
HYPERPARAMS = {
    "seed": 42,
    "embedding_size": embedding_size,
    "lr_patience": 10,
    "lr_factor": 0.5,
    "min_lr": 1e-6,
    "gradient_clip": 1.0,
    "warmup_epochs": 10,
    "use_ema": True,  # Use Exponential Moving Average for more stable predictions
    "ema_decay": 0.999,
}

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(HYPERPARAMS["seed"])

# -------------------- Dataset Loading (Original Style) --------------------
print("\nLoading preprocessed dataset...")
dataset = Twibot22(root=root, device=device, process=False, save=False)
(des_tensor, tweets_tensor, num_prop, category_prop,
 edge_index, edge_type, labels, train_idx, val_idx, test_idx) = dataset.dataloader()

# -------------------- Adaptive Hyperparameter Selection --------------------
# Now that we know dataset size, update hyperparameters accordingly
num_training_samples = len(train_idx)
adaptive_params = get_adaptive_hyperparams(num_training_samples)

# Merge adaptive params into HYPERPARAMS
HYPERPARAMS.update(adaptive_params)
HYPERPARAMS["embedding_size"] = embedding_size

# Extract commonly used variables
dropout = HYPERPARAMS["dropout"]
lr = HYPERPARAMS["lr"]
weight_decay = HYPERPARAMS["weight_decay"]
epochs = HYPERPARAMS["epochs"]

print(f"\nFinal Hyperparameters:")
print(json.dumps(HYPERPARAMS, indent=2))

des_tensor = des_tensor.to(device)
tweets_tensor = tweets_tensor.to(device)
num_prop = num_prop.to(device)
category_prop = category_prop.to(device)
edge_index = edge_index.to(device)
edge_type = edge_type.to(device)
labels = labels.to(device)

# --- Minimal shape guard to match features to the number of labeled users ---
N = labels.shape[0]

def _align_rows(t: torch.Tensor, target_n: int) -> torch.Tensor:
    """Pad with zeros (or trim) so t.shape[0] == target_n, keeping dtype/device."""
    if t.shape[0] == target_n:
        return t
    if t.shape[0] > target_n:
        print(f"[warn] trimming {tuple(t.shape)} -> ({target_n}, {t.shape[1]})")
        return t[:target_n]
    # pad
    pad_rows = target_n - t.shape[0]
    print(f"[warn] padding {tuple(t.shape)} -> ({target_n}, {t.shape[1]}) with zeros")
    pad = torch.zeros((pad_rows, t.shape[1]), device=t.device, dtype=t.dtype)
    return torch.cat([t, pad], dim=0)

des_tensor    = _align_rows(des_tensor, N)
tweets_tensor = _align_rows(tweets_tensor, N)
num_prop      = _align_rows(num_prop, N)
category_prop = _align_rows(category_prop, N)

print("Aligned feature shapes:",
      f"des={tuple(des_tensor.shape)},",
      f"tweets={tuple(tweets_tensor.shape)},",
      f"num_prop={tuple(num_prop.shape)},",
      f"cat_prop={tuple(category_prop.shape)}")

# -------------------- Feature Normalization --------------------
print("\nApplying feature normalization for better generalization...")
import pickle
from sklearn.preprocessing import RobustScaler

# Normalize numerical properties (robust to outliers)
num_prop_np = num_prop.cpu().numpy()
num_scaler = RobustScaler()
num_prop_normalized = num_scaler.fit_transform(num_prop_np)
num_prop = torch.tensor(num_prop_normalized, device=device, dtype=torch.float32)

# Save scaler for inference
scaler_dir = Path("./scalers")
scaler_dir.mkdir(exist_ok=True)
with open(scaler_dir / "num_scaler.pkl", 'wb') as f:
    pickle.dump(num_scaler, f)
print(f"Saved numerical feature scaler to {scaler_dir / 'num_scaler.pkl'}")

# Normalize categorical properties
cat_prop_np = category_prop.cpu().numpy()
cat_scaler = RobustScaler()
cat_prop_normalized = cat_scaler.fit_transform(cat_prop_np)
category_prop = torch.tensor(cat_prop_normalized, device=device, dtype=torch.float32)

with open(scaler_dir / "cat_scaler.pkl", 'wb') as f:
    pickle.dump(cat_scaler, f)
print(f"Saved categorical feature scaler to {scaler_dir / 'cat_scaler.pkl'}")
# ---------------------------------------------------------------------------

# <— use the shared helper so we're robust if edge_type is empty or non-0..R-1
NUM_RELS = infer_num_relations(edge_type)

n_train = len(train_idx)  # <— correct split sizes
n_val   = len(val_idx)
n_test  = len(test_idx)

print("Dataset loaded successfully!")
print(f"Number of nodes: {des_tensor.shape[0]}")
print(f"Number of relations: {NUM_RELS}")
print(f"Train/Val/Test sizes: {n_train}/{n_val}/{n_test}")

# Display class distribution by split
train_labels = labels[train_idx]
val_labels = labels[val_idx]
test_labels = labels[test_idx]

train_bots = (train_labels == 1).sum().item()
train_humans = (train_labels == 0).sum().item()
val_bots = (val_labels == 1).sum().item()
val_humans = (val_labels == 0).sum().item()
test_bots = (test_labels == 1).sum().item()
test_humans = (test_labels == 0).sum().item()

print(f"Split distributions:")
print(f"  Train - Bots: {train_bots} ({train_bots/n_train*100:.1f}%), Humans: {train_humans} ({train_humans/n_train*100:.1f}%)")
print(f"  Val   - Bots: {val_bots} ({val_bots/n_val*100:.1f}%), Humans: {val_humans} ({val_humans/n_val*100:.1f}%)")
print(f"  Test  - Bots: {test_bots} ({test_bots/n_test*100:.1f}%), Humans: {test_humans} ({test_humans/n_test*100:.1f}%)")

# -------------------- Model Initialization --------------------
model = BotRGCN(
    num_relations=NUM_RELS,
    des_size=des_tensor.shape[1],
    tweet_size=tweets_tensor.shape[1],
    num_prop_size=num_prop.shape[1],
    cat_prop_size=category_prop.shape[1],
    embedding_dimension=embedding_size
).to(device)

# Class weights for imbalanced dataset
pos = (labels == 1).sum().item()
neg = (labels == 0).sum().item()
total = pos + neg
class_ratio = neg / pos if pos > 0 else 1.0
w1 = min(class_ratio, 10.0)
w0 = 1.0
class_weights = torch.tensor([w0, w1], device=device, dtype=torch.float)

print(f"Class distribution: Human={neg} ({neg/total*100:.1f}%), Bot={pos} ({pos/total*100:.1f}%)")
print(f"Class weights: human={w0:.2f}, bot={w1:.2f}")

# Loss function and optimizer (with increased label smoothing)
loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=HYPERPARAMS["label_smoothing"])
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay,
    betas=(0.9, 0.999)
)

# Enhanced scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=HYPERPARAMS["lr_factor"],
    patience=HYPERPARAMS["lr_patience"], min_lr=HYPERPARAMS["min_lr"]
)

def get_lr_scale(epoch):
    if epoch < HYPERPARAMS["warmup_epochs"]:
        return (epoch + 1) / HYPERPARAMS["warmup_epochs"]
    return 1.0

model.apply(init_weights)

# -------------------- EMA (Exponential Moving Average) Setup --------------------
class EMA:
    """Exponential Moving Average for model parameters - provides more stable predictions"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model (for evaluation)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters (for training)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

# Initialize EMA if enabled
ema = None
if HYPERPARAMS.get("use_ema", True):
    ema = EMA(model, decay=HYPERPARAMS.get("ema_decay", 0.999))
    print(f"EMA enabled with decay={HYPERPARAMS.get('ema_decay', 0.999)}")

# -------------------- Training Setup --------------------
training_log = {
    "hyperparameters": HYPERPARAMS,
    "start_time": datetime.now().isoformat(),
    "epochs": [],
    "best_metrics": {},
    "final_threshold": 0.5,
    "num_relations": NUM_RELS,  # <— record it
}

best_val_f1 = 0.0
best_epoch = 0
patience_counter = 0

# -------------------- Graph Augmentation Helper --------------------
def augment_edges(edge_index, edge_type, drop_prob=0.3):
    """Randomly drop edges for graph augmentation during training"""
    if drop_prob <= 0:
        return edge_index, edge_type

    num_edges = edge_index.shape[1]
    keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_prob
    return edge_index[:, keep_mask], edge_type[keep_mask]

# -------------------- Training Functions --------------------
def train(epoch):
    """Training function following BOTRGCN_Original structure but with enhanced features"""
    model.train()

    # Warmup learning rate
    lr_scale = get_lr_scale(epoch)
    for pg in optimizer.param_groups:
        pg["lr"] = lr * lr_scale

    # Graph augmentation: randomly drop edges during training
    edge_index_aug, edge_type_aug = augment_edges(
        edge_index, edge_type,
        drop_prob=HYPERPARAMS["edge_drop_prob"]
    )

    # Forward pass with augmented graph
    output = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index_aug, edge_type_aug)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])

    # Validation accuracy (following original pattern)
    model.eval()

    # Apply EMA weights for validation if enabled
    if ema is not None:
        ema.apply_shadow()

    with torch.no_grad():
        val_output = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
        acc_val = accuracy(val_output[val_idx], labels[val_idx])
        val_loss = loss(val_output[val_idx], labels[val_idx])

        # Enhanced validation metrics
        val_probs = torch.softmax(val_output[val_idx], dim=1)[:, 1].detach().cpu().numpy()
        val_true = labels[val_idx].detach().cpu().numpy()
        val_preds = (val_probs > 0.5).astype(int)
        val_f1 = f1_score(val_true, val_preds, zero_division=0)

    # Restore original weights for training
    if ema is not None:
        ema.restore()

    model.train()

    # Backward pass with gradient clipping
    optimizer.zero_grad()
    loss_train.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), HYPERPARAMS["gradient_clip"])
    optimizer.step()

    # Update EMA after optimizer step
    if ema is not None:
        ema.update()

    # Logging (enhanced version)
    epoch_metrics = {
        "epoch": epoch + 1,
        "train_loss": loss_train.item(),
        "train_acc": float(acc_train),
        "val_loss": val_loss.item(),
        "val_acc": float(acc_val),
        "val_f1": val_f1,
        "lr": optimizer.param_groups[0]["lr"],
    }
    training_log["epochs"].append(epoch_metrics)

    # Print progress (following original format but enhanced)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'val_f1: {:.4f}'.format(val_f1),
          'lr: {:.2e}'.format(optimizer.param_groups[0]['lr']))

    return acc_train, loss_train, val_f1

# -------------------- Enhanced Test Function --------------------
def test():
    """Enhanced test function with comprehensive metrics following original pattern"""
    model.eval()

    # Apply EMA weights for testing if enabled
    if ema is not None:
        ema.apply_shadow()

    # Basic test metrics (following original structure)
    output = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
    loss_test = loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])

    # Convert to numpy for sklearn metrics
    output_np = output.max(1)[1].to('cpu').detach().numpy()
    label_np = labels.to('cpu').detach().numpy()
    test_idx_np = test_idx.to('cpu').detach().numpy()

    # Basic metrics (following original)
    f1 = f1_score(label_np[test_idx_np], output_np[test_idx_np])
    precision = precision_score(label_np[test_idx_np], output_np[test_idx_np])
    recall = recall_score(label_np[test_idx_np], output_np[test_idx_np])

    # Enhanced metrics with probabilities
    probs = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
    fpr, tpr, thresholds = roc_curve(label_np[test_idx_np], probs[test_idx_np], pos_label=1)
    Auc = auc(fpr, tpr)

    # Print results (following original format but enhanced)
    print("Test set results:",
          "test_loss= {:.4f}".format(loss_test.item()),
          "test_accuracy= {:.4f}".format(acc_test.item()),
          "precision= {:.4f}".format(precision),
          "recall= {:.4f}".format(recall),
          "f1_score= {:.4f}".format(f1),
          "auc= {:.4f}".format(Auc))

    # Restore original weights after testing
    if ema is not None:
        ema.restore()

    return {
        "test_loss": loss_test.item(),
        "test_accuracy": acc_test.item(),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": Auc
    }

# -------------------- Enhanced Evaluation Functions --------------------
def enhanced_threshold_optimization():
    print("\n" + "="*80)
    print("ENHANCED THRESHOLD OPTIMIZATION")
    print("="*80)

    model.eval()

    # Apply EMA weights for threshold optimization
    if ema is not None:
        ema.apply_shadow()

    with torch.no_grad():
        logits = model(des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_true = labels.detach().cpu().numpy()

    # Convert index tensors to numpy arrays (these contain actual indices, not boolean masks)
    val_indices = val_idx.detach().cpu().numpy()
    test_indices = test_idx.detach().cpu().numpy()

    # Extract validation and test data using index arrays
    probs_val = probs[val_indices]
    y_val = y_true[val_indices]
    probs_test = probs[test_indices]
    y_test = y_true[test_indices]

    print(f"Validation set size: {len(probs_val)}")
    print(f"Test set size: {len(probs_test)}")

    # Check if we have valid data
    if len(probs_val) == 0 or len(probs_test) == 0:
        print("ERROR: Empty validation or test set!")
        return 0.5, {}

    thresholds = np.linspace(0.01, 0.99, 991)
    metrics = ["f1", "precision", "recall", "balanced_accuracy"]
    best_thresholds, best_scores = {}, {}

    for metric in metrics:
        best_thr, best_score = 0.5, -1.0
        for thr in thresholds:
            preds_val = (probs_val >= thr).astype(int)

            if metric == "f1":
                score = f1_score(y_val, preds_val, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_val, preds_val, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_val, preds_val, zero_division=0)
            elif metric == "balanced_accuracy":
                tn = np.sum((y_val == 0) & (preds_val == 0))
                tp = np.sum((y_val == 1) & (preds_val == 1))
                fn = np.sum((y_val == 1) & (preds_val == 0))
                fp = np.sum((y_val == 0) & (preds_val == 1))
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                score = 0.5 * (sensitivity + specificity)

            if score > best_score:
                best_score, best_thr = score, thr

        best_thresholds[metric] = best_thr
        best_scores[metric] = best_score
        print(f"Best threshold for {metric}: {best_thr:.4f} (val {metric}={best_score:.4f})")

    primary_threshold = best_thresholds["f1"]

    print("\n" + "-"*60)
    print("TEST SET EVALUATION")
    print("-"*60)

    test_results = {}
    for metric, thr in best_thresholds.items():
        preds_test = (probs_test >= thr).astype(int)

        prec = precision_score(y_test, preds_test, zero_division=0)
        rec  = recall_score(y_test, preds_test, zero_division=0)
        f1   = f1_score(y_test, preds_test, zero_division=0)

        fpr, tpr, _ = roc_curve(y_test, probs_test)
        auc_score = auc(fpr, tpr)
        ap_score  = average_precision_score(y_test, probs_test)
        acc = (preds_test == y_test).mean()

        test_results[metric] = {
            "threshold": thr,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc_score,
            "average_precision": ap_score,
        }

        print(f"\n{metric.upper()}-optimized threshold ({thr:.4f}):")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall: {rec:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"  AUC: {auc_score:.4f}")
        print(f"  AP: {ap_score:.4f}")

    print("\n" + "="*60)
    print(f"DETAILED ANALYSIS (F1-optimized threshold: {primary_threshold:.4f})")
    print("="*60)

    preds_test_primary = (probs_test >= primary_threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, preds_test_primary, target_names=["Human", "Bot"], zero_division=0))

    print("\nConfusion Matrix [rows=true, cols=pred]:")
    cm = confusion_matrix(y_test, preds_test_primary)
    print(f"                Predicted")
    print(f"Actual    Human    Bot")
    print(f"Human     {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"Bot       {cm[1,0]:5d}  {cm[1,1]:5d}")

    training_log["threshold_optimization"] = {
        "best_thresholds": best_thresholds,
        "validation_scores": best_scores,
        "test_results": test_results,
        "primary_threshold": primary_threshold,
    }
    training_log["final_threshold"] = primary_threshold

    # Restore original weights
    if ema is not None:
        ema.restore()

    return primary_threshold, test_results

# -------------------- Model Saving Function --------------------
def save_model_and_logs():
    """Save trained model and logs for use in test_data.py"""
    print(f"\nSaving final model to {MODEL_SAVE_PATH}...")

    # Apply EMA weights for final saved model (better for inference)
    if ema is not None:
        ema.apply_shadow()

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_relations": NUM_RELS,
            "des_size": des_tensor.shape[1],
            "tweet_size": tweets_tensor.shape[1],
            "num_prop_size": num_prop.shape[1],
            "cat_prop_size": category_prop.shape[1],
            "embedding_dimension": embedding_size,
        },
        "class_weights": class_weights,
        "device": device,
        "hyperparameters": HYPERPARAMS,
        "training_log": training_log,
        "final_threshold": training_log.get("final_threshold", 0.5),
    }, MODEL_SAVE_PATH)

    # Save training log
    training_log["end_time"] = datetime.now().isoformat()
    training_log["total_epochs"] = len(training_log["epochs"])
    with open(TRAINING_LOG_PATH, "w") as f:
        json.dump(training_log, f, indent=2)

    print("Model saved successfully!")
    print(f"Final model: {MODEL_SAVE_PATH}")
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Training log: {TRAINING_LOG_PATH}")

# Apply weight initialization
model.apply(init_weights)

# -------------------- Main Training Loop --------------------
print("\n" + "="*80)
print("STARTING TRAINING (BOTRGCN_Original Flow + Enhanced Metrics)")
print("="*80)

start_time = time.time()

# Training loop following original structure
for epoch in range(epochs):
    acc_train, loss_train, val_f1 = train(epoch)

    # Enhanced scheduler
    scheduler.step(val_f1)

    # Save best model (enhanced checkpoint saving)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch
        patience_counter = 0

        # Save best model checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_f1": val_f1,
            "model_config": {
                "num_relations": NUM_RELS,
                "des_size": des_tensor.shape[1],
                "tweet_size": tweets_tensor.shape[1],
                "num_prop_size": num_prop.shape[1],
                "cat_prop_size": category_prop.shape[1],
                "embedding_dimension": embedding_size,
            }
        }, BEST_MODEL_PATH)

        training_log["best_metrics"] = {
            "epoch": epoch + 1,
            "val_f1": val_f1,
            "train_acc": float(acc_train),
            "train_loss": float(loss_train.item()),
        }
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= HYPERPARAMS["patience"]:
        print(f"\nEarly stopping triggered at epoch {epoch+1}")
        print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
        break

    # Progress checkpoints
    if (epoch + 1) % 50 == 0:
        print(f"Checkpoint at epoch {epoch+1}: Best Val F1 = {best_val_f1:.4f}")

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time/60:.2f} minutes")
print(f"Best validation F1: {best_val_f1:.4f} achieved at epoch {best_epoch+1}")

# Load best model for final evaluation
print("\nLoading best model for final evaluation...")
best_checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(best_checkpoint["model_state_dict"])

# Final test evaluation (following original pattern)
print("\n" + "="*80)
print("FINAL TEST EVALUATION")
print("="*80)
final_test_results = test()

# Enhanced threshold optimization and detailed evaluation
optimal_threshold, detailed_results = enhanced_threshold_optimization()

# Save final model and logs
save_model_and_logs()

# Final summary
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Training time: {training_time/60:.2f} minutes")
print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Final test F1: {final_test_results['f1_score']:.4f}")
print(f"Final test AUC: {final_test_results['auc']:.4f}")
print("\nModel files saved:")
print(f"  • Best model: {BEST_MODEL_PATH}")
print(f"  • Final model: {MODEL_SAVE_PATH}")
print(f"  • Training log: {TRAINING_LOG_PATH}")
print("\nReady for use with test_data.py!")
