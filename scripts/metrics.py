import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import json

def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return (y_true.argmax(dim=1) == y_pred.argmax(dim=1)).float().mean().item()

def precision_recall_f1(y_true: torch.Tensor, y_pred: torch.Tensor, average: str = 'macro') -> tuple[float, float, float]:
    num_classes = y_true.shape[1]
    y_true_classes = y_true.argmax(dim=1)
    y_pred_classes = y_pred.argmax(dim=1)
    
    if average == 'micro':
        true_positive = (y_true_classes == y_pred_classes).sum().float()
        return (true_positive / len(y_true)).item(), (true_positive / len(y_true)).item(), (true_positive / len(y_true)).item()
    
    tp = torch.zeros(num_classes, device=y_true.device)
    fp = torch.zeros(num_classes, device=y_true.device)
    fn = torch.zeros(num_classes, device=y_true.device)
    
    for c in range(num_classes):
        tp[c] = ((y_true_classes == c) & (y_pred_classes == c)).sum().float()
        fp[c] = ((y_true_classes != c) & (y_pred_classes == c)).sum().float()
        fn[c] = ((y_true_classes == c) & (y_pred_classes != c)).sum().float()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    if average == 'macro':
        return precision.mean().item(), recall.mean().item(), f1.mean().item()
    elif average == 'weighted':
        weights = torch.bincount(y_true_classes).float() / len(y_true)
        return (precision * weights).sum().item(), (recall * weights).sum().item(), (f1 * weights).sum().item()
    else:
        raise ValueError("Averaging method must be 'macro', 'micro', or 'weighted'")

def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    num_classes = y_true.shape[1]
    y_true_classes = y_true.argmax(dim=1)
    y_pred_classes = y_pred.argmax(dim=1)
    return torch.zeros(num_classes, num_classes, dtype=torch.int64, device=y_true.device).scatter_add_(1, y_pred_classes.unsqueeze(1), torch.eye(num_classes, dtype=torch.int64, device=y_true.device)[y_true_classes])

def calculate_specificity(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    conf_matrix = confusion_matrix(y_true, y_pred)
    fp = torch.sum(conf_matrix, dim=0) - torch.diag(conf_matrix)
    tn = torch.sum(conf_matrix) - (fp + torch.sum(conf_matrix, dim=1))
    return tn / (tn + fp + 1e-7)

def generate_metrics_report(y_true: torch.Tensor, y_pred: torch.Tensor) -> str:
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    metrics_report = {}
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    tp = torch.diag(conf_matrix)
    fp = torch.sum(conf_matrix, dim=0) - tp
    fn = torch.sum(conf_matrix, dim=1) - tp
    tn = torch.sum(conf_matrix) - (fp + fn + tp)
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    specificity = tn / (tn + fp + 1e-7)
    
    for i, class_name in enumerate(class_columns):
        metrics_report[class_name] = {
            'precision': precision[i].item(),
            'recall': recall[i].item(),
            'f1-score': f1[i].item(),
            'specificity': specificity[i].item()
        }
    
    auc_roc_scores = {}
    average_precision_scores = {}
    for i, class_name in enumerate(class_columns):
        y_true_class = y_true[:, i].cpu().numpy()
        y_pred_class = y_pred[:, i].cpu().numpy()
        
        try:
            auc_roc_scores[class_name] = roc_auc_score(y_true_class, y_pred_class)
        except ValueError:
            auc_roc_scores[class_name] = 0.0
        
        try:
            precision, recall, _ = precision_recall_curve(y_true_class, y_pred_class)
            average_precision_scores[class_name] = auc(recall, precision)
        except ValueError:
            average_precision_scores[class_name] = 0.0
    
    metrics_report['accuracy'] = accuracy(y_true, y_pred)
    metrics_report['macro avg'] = {
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1-score': f1.mean().item(),
        'specificity': specificity.mean().item()
    }
    metrics_report['auc_roc_scores'] = auc_roc_scores
    metrics_report['average_precision_scores'] = average_precision_scores
    metrics_report['mean_auc'] = np.mean(list(auc_roc_scores.values()))
    metrics_report['mean_average_precision'] = np.mean(list(average_precision_scores.values()))
    
    return json.dumps(metrics_report, indent=4)