import torch
from torch import nn, optim
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from utils import *
from metrics import *

def train_step(model: nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: nn.Module, 
               optimizer: optim.Optimizer,
               device: torch.device) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.train()
    train_loss, correct = 0, 0
    total = 0
    all_predictions = []
    all_labels = []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)  # More efficient than optimizer.zero_grad()

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)
        _, predicted = y_pred.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        all_predictions.append(predicted)
        all_labels.append(y)

    train_loss /= total
    train_acc = correct / total
    return train_loss, train_acc, torch.cat(all_predictions), torch.cat(all_labels)

def test_step(model: nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: nn.Module,
              device: torch.device) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    test_loss, correct = 0, 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            test_loss += loss.item() * X.size(0)
            _, predicted = y_pred.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            all_predictions.append(predicted)
            all_labels.append(y)

    test_loss /= total
    test_acc = correct / total
    return test_loss, test_acc, torch.cat(all_predictions), torch.cat(all_labels)

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          model_name: str,
          save_dir: str) -> Dict[str, List]:
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    model.to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    logger = setup_logger(model_name)
    logger.info(f"Training started for model: {model_name}")

    for epoch in range(epochs):
        train_loss, train_acc, train_preds, train_labels = train_step(
            model, train_dataloader, loss_fn, optimizer, device
        )
        test_loss, test_acc, test_preds, test_labels = test_step(
            model, test_dataloader, loss_fn, device
        )

        scheduler.step()

        # Generate and log metrics
        train_metrics = generate_metrics_report(train_labels, train_preds)
        test_metrics = generate_metrics_report(test_labels, test_preds)

        logger.info(f"Epoch: {epoch+1}")
        logger.info(f"Train Metrics:\n{train_metrics}")
        logger.info(f"Test Metrics:\n{test_metrics}")
        logger.info(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            save_model(model, save_dir, f"{model_name}.pth")
            logger.info(f"Model checkpoint saved at epoch {epoch+1}")

        # Save metrics report
        save_metrics_report({
            "epoch": epoch + 1,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }, model_name, epoch)

        # Update results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    logger.info(f"Training completed for model: {model_name}")

    del model, optimizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return results