import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from tqdm import tqdm
from utils import *
from metrics import *  # Make sure to import the function here


def train_step(model: nn.Module, 
               dataloader: DataLoader, 
               loss_fn: nn.Module, 
               optimizer: optim.Optimizer,
               device: torch.device,
               epoch_progress: tqdm) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.train()
    train_loss, correct = 0, 0
    total = 0
    all_predictions = []
    all_labels = []

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        y_pred_logits = model(X)  # Get model logits
        # print(f"y_pred_logits shape after model: {y_pred_logits.shape}")  # Debug shape
        
        loss = loss_fn(y_pred_logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)
        _, predicted = y_pred_logits.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        # Collect logits instead of raw predictions
        all_predictions.append(y_pred_logits)  # Append logits
        all_labels.append(y)

        # Update the epoch progress bar description with batch information
        epoch_progress.set_postfix(
            batch=f"{batch_idx + 1}/{len(dataloader)}", 
            loss=f"{loss.item():.4f}", 
            mode="training"
        )

    train_loss /= total
    train_acc = correct / total
    
    # Concatenate along the first dimension to create a tensor of shape [total_samples, num_classes]
    train_preds = torch.cat(all_predictions, dim=0)  # Shape [total_samples, num_classes]
    train_labels = torch.cat(all_labels, dim=0)     # Shape [total_samples]

    return train_loss, train_acc, train_preds, train_labels


def test_step(model: nn.Module, 
              dataloader: DataLoader, 
              loss_fn: nn.Module,
              device: torch.device,
              epoch_progress: tqdm) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    test_loss, correct = 0, 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)  # Get model logits
            # print(f"y_pred_logits shape after model: {y_pred_logits.shape}")  # Debug shape
            
            loss = loss_fn(y_pred_logits, y)

            test_loss += loss.item() * X.size(0)
            _, predicted = y_pred_logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            # Collect logits instead of raw predictions
            all_predictions.append(y_pred_logits)  # Append logits
            all_labels.append(y)

            # Update the epoch progress bar description with batch information
            epoch_progress.set_postfix(
                batch=f"{batch_idx + 1}/{len(dataloader)}", 
                loss=f"{loss.item():.4f}", 
                mode="validation"
            )

    test_loss /= total
    test_acc = correct / total
    
    # Concatenate along the first dimension to create a tensor of shape [total_samples, num_classes]
    test_preds = torch.cat(all_predictions, dim=0)  # Shape [total_samples, num_classes]
    test_labels = torch.cat(all_labels, dim=0)      # Shape [total_samples]

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
          patience: int = 5,  # Number of epochs to wait for improvement before stopping
          tolerance: float = 1e-4  # Minimum improvement to consider it a significant improvement
          ) -> Dict[str, List]:
    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "mean_auc": []  # To store the mean AUC scores per epoch
    }
    
    model.to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    logger = setup_logger(model_name)
    logger.info(f"Training started for model: {model_name}")
    
    best_mean_auc = -float('inf')  # Initialize to a very low value
    best_epoch = 0
    no_improvement_count = 0  # Track the number of epochs without improvement

    for epoch in range(epochs):
        # Create a tqdm progress bar for each epoch
        with tqdm(desc=f"Epoch {epoch+1}/{epochs}", total=len(train_dataloader) + len(test_dataloader)) as epoch_progress:
            train_loss, train_acc, train_preds, train_labels = train_step(
                model, train_dataloader, loss_fn, optimizer, device, epoch_progress
            )
            test_loss, test_acc, test_preds, test_labels = test_step(
                model, test_dataloader, loss_fn, device, epoch_progress
            )

        scheduler.step()

        # Apply softmax to convert logits to probabilities for metrics reporting
        train_preds_probs = torch.softmax(train_preds, dim=1)  # Shape [total_samples, num_classes]
        test_preds_probs = torch.softmax(test_preds, dim=1)    # Shape [total_samples, num_classes]

        # Generate and log metrics
        train_metrics = generate_metrics_report(train_labels, train_preds_probs)
        test_metrics = generate_metrics_report(test_labels, test_preds_probs)

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

        # Extract the mean AUC score from the test metrics
        test_metrics_dict = json.loads(test_metrics)
        current_mean_auc = test_metrics_dict['mean_auc']  # Mean AUC score for the current epoch

        # Update results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["mean_auc"].append(current_mean_auc)

        # Early Stopping Logic
        if current_mean_auc > best_mean_auc + tolerance:
            best_mean_auc = current_mean_auc
            best_epoch = epoch + 1
            no_improvement_count = 0  # Reset the count since we have an improvement
            save_model(model, save_dir, f"{model_name}_best.pth")
            logger.info(f"Best model saved with mean AUC score: {best_mean_auc:.4f} at epoch {epoch+1}")
        else:
            no_improvement_count += 1
            logger.info(f"No improvement for {no_improvement_count} consecutive epochs.")

        # Stop training if no improvement for the last 'patience' epochs
        if no_improvement_count >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} after {patience} epochs of no improvement.")
            break

        # Save metrics report
        save_metrics_report({
            "epoch": epoch + 1,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }, model_name, epoch)

    logger.info(f"Training completed for model: {model_name}. Best model at epoch {best_epoch} with mean AUC: {best_mean_auc:.4f}")

    del model, optimizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return results


def train_single_model(model_name: str, 
                       model: nn.Module, 
                       train_dataloader: DataLoader, 
                       test_dataloader: DataLoader, 
                       epochs: int, 
                       device: torch.device, 
                       save_dir: str, 
                       engine) -> dict:

    # Move model to target device
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    loss_fn = FocalLoss()
    
    # Train the model using the engine's train function
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
        model_name=model_name,  # Model name for logging and saving
        save_dir=save_dir,  # Directory to save models
    )
    
    return results