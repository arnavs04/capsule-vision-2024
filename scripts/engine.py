import torch
from torch import nn, optim
from typing import Dict, List, Tuple
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from utils import *
from metrics import *


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.train()
    train_loss, train_acc = 0, 0
    all_predictions = []
    all_labels = []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        all_predictions.append(y_pred_class)
        all_labels.append(y)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc, torch.cat(all_predictions), torch.cat(all_labels)


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval() 
    test_loss, test_acc = 0, 0
    all_predictions = []
    all_labels = []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

            all_predictions.append(test_pred_labels)
            all_labels.append(y)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc, torch.cat(all_predictions), torch.cat(all_labels)


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          model_name: str) -> Dict[str, List]:
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    model.to(device)

    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Setup logger
    logger = setup_logger(model_name)

    logger.info(f"Training started for model: {model_name}")

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_preds, train_labels = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc, test_preds, test_labels = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Update learning rate
        scheduler.step()

        # Generate metrics report
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

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    logger.info("Training completed")
    return results