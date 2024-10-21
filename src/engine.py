import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
from utils import *
from metrics import *  # Make sure to import your metrics functions here


def train_step(model: nn.Module, 
               dataloader: DataLoader, 
               loss_fn: nn.Module, 
               optimizer: optim.Optimizer,
               device: torch.device) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.train()
    train_loss, correct = 0, 0
    total = 0
    all_predictions = []
    all_labels = []

    train_progress = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, (X, y) in enumerate(train_progress):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        y_pred_logits = model(X)
        loss = loss_fn(y_pred_logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)
        _, predicted = y_pred_logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        all_predictions.append(y_pred_logits)
        all_labels.append(y)

        train_progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct/total:.4f}"
        )

    train_loss /= total
    train_acc = correct / total
    train_preds = torch.cat(all_predictions, dim=0)
    train_labels = torch.cat(all_labels, dim=0)

    return train_loss, train_acc, train_preds, train_labels


def test_step(model: nn.Module, 
              dataloader: DataLoader, 
              loss_fn: nn.Module,
              device: torch.device) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    test_loss, correct = 0, 0
    total = 0
    all_predictions = []
    all_labels = []

    val_progress = tqdm(dataloader, desc="Validation", leave=False)
    with torch.inference_mode():
        for batch_idx, (X, y) in enumerate(val_progress):
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)
            loss = loss_fn(y_pred_logits, y)

            test_loss += loss.item() * X.size(0)
            _, predicted = y_pred_logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            all_predictions.append(y_pred_logits)
            all_labels.append(y)

            val_progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct/total:.4f}"
            )

    test_loss /= total
    test_acc = correct / total
    test_preds = torch.cat(all_predictions, dim=0)
    test_labels = torch.cat(all_labels, dim=0)

    return test_loss, test_acc, test_preds, test_labels


def train(model: nn.Module, 
          train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          optimizer: optim.Optimizer,
          loss_fn: nn.Module,
          epochs: int,
          device: torch.device,
          model_name: str,
          save_dir: str,
          patience: int = 5,
          tolerance: float = 1e-4
          ) -> Dict[str, List]:
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "mean_auc": [],
        "balanced_accuracy": []
    }
    
    model.to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    logger = setup_logger(model_name)
    logger.info(f"Training started for model: {model_name}")
    
    best_score = -float('inf')
    best_epoch = 0
    no_improvement_count = 0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        train_loss, train_acc, train_preds, train_labels = train_step(
            model, train_dataloader, loss_fn, optimizer, device
        )
        test_loss, test_acc, test_preds, test_labels = test_step(
            model, test_dataloader, loss_fn, device
        )

        scheduler.step()

        train_preds_probs = torch.softmax(train_preds, dim=1)
        test_preds_probs = torch.softmax(test_preds, dim=1)

        train_metrics = generate_metrics_report(train_labels, train_preds_probs)
        test_metrics = generate_metrics_report(test_labels, test_preds_probs)

        logger.info(f"Train Metrics:\n{train_metrics}")
        logger.info(f"Test Metrics:\n{test_metrics}")
        logger.info(
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        test_metrics_dict = json.loads(test_metrics)
        current_mean_auc = test_metrics_dict['mean_auc']
        current_balanced_accuracy = test_metrics_dict['balanced_accuracy']
        
        current_score = (current_mean_auc + current_balanced_accuracy) / 2

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["mean_auc"].append(current_mean_auc)
        results["balanced_accuracy"].append(current_balanced_accuracy)

        if current_score > best_score + tolerance:
            best_score = current_score
            best_epoch = epoch + 1
            no_improvement_count = 0
            save_model(model, save_dir, f"{model_name}_best.pth")
            logger.info(f"Best model saved with combined score: {best_score:.4f} (Mean AUC: {current_mean_auc:.4f}, Balanced Accuracy: {current_balanced_accuracy:.4f})")
        else:
            no_improvement_count += 1
            logger.info(f"No improvement for {no_improvement_count} consecutive epochs.")

        if no_improvement_count >= patience:
            logger.info(f"Early stopping after {patience} epochs of no improvement.")
            break

        save_metrics_report({
            "epoch": epoch + 1,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }, model_name, epoch)

    logger.info(f"Training completed. Best model at epoch {best_epoch} with combined score: {best_score:.4f}")

    del model, optimizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return results