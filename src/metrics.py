import torch
import torchmetrics
from torch import nn
import json
from typing import List


class FocalLoss(nn.Module):
    """
    Focal Loss to address class imbalance by focusing on hard-to-classify examples.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha) if isinstance(alpha, (float, int)) else alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)  # Compute cross-entropy loss
        p_t = torch.exp(-CE_loss)  # Compute model's confidence

        # Ensure alpha is on the correct device
        if self.alpha.device != inputs.device: 
            self.alpha = self.alpha.to(inputs.device)
        
        # Apply alpha weighting for each class
        alpha_t = self.alpha[targets] if self.alpha.numel() > 1 else self.alpha

        # Focal loss computation
        loss = alpha_t * (1 - p_t) ** self.gamma * CE_loss
        
        # Apply reduction (mean/sum)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class MetricsCalculator:
    """
    Metric calculation and reporting for multiclass classification tasks.
    """
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.metrics = None

    def _initialize_metrics(self, device):
        """
        Initialize the necessary metrics and move them to the appropriate device.
        """
        self.metrics = {
            'confusion_matrix': torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes).to(device),
            'accuracy': torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(device),
            'precision': torchmetrics.Precision(task="multiclass", num_classes=self.num_classes, average=None).to(device),
            'recall': torchmetrics.Recall(task="multiclass", num_classes=self.num_classes, average=None).to(device),
            'f1_score': torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes, average=None).to(device),
            'specificity': torchmetrics.Specificity(task="multiclass", num_classes=self.num_classes, average=None).to(device),
            'auroc': torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes, average=None).to(device),
            'auprc': torchmetrics.AveragePrecision(task="multiclass", num_classes=self.num_classes, average=None).to(device)
        }

    @staticmethod
    def to_cpu(t):
        """Move tensors to CPU and convert to list if needed."""
        return t.cpu().tolist() if isinstance(t, torch.Tensor) else t

    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        Compute various metrics for multiclass classification, including accuracy, precision, recall, F1-score, etc.
        """
        device = y_true.device
        y_pred = y_pred.to(device)

        if self.metrics is None or next(iter(self.metrics.values())).device != device:
            self._initialize_metrics(device)

        # Softmax for probability-based metrics
        y_pred_softmax = torch.softmax(y_pred, dim=1)

        # Argmax for class-based metrics
        y_pred_classes = torch.argmax(y_pred, dim=1)

        # Ensure ground truth labels are long tensor
        y_true = y_true.long().squeeze(1) if y_true.dim() == 2 else y_true

        # Compute metrics
        metrics_values = {
            'confusion_matrix': self.metrics['confusion_matrix'](y_pred_classes, y_true),
            'accuracy': self.metrics['accuracy'](y_pred_classes, y_true),
            'precision': self.metrics['precision'](y_pred_classes, y_true),
            'recall': self.metrics['recall'](y_pred_classes, y_true),
            'f1_score': self.metrics['f1_score'](y_pred_classes, y_true),
            'specificity': self.metrics['specificity'](y_pred_classes, y_true),
            'auroc': self.metrics['auroc'](y_pred_softmax, y_true),
            'auprc': self.metrics['auprc'](y_pred_softmax, y_true),
        }

        # Compute balanced accuracy as the mean recall across all classes
        metrics_values['balanced_accuracy'] = metrics_values['recall'].mean()

        return metrics_values

    def generate_metrics_report(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> str:
        """
        Generate a detailed JSON report of the computed metrics.
        """
        metrics_values = self.compute_metrics(y_true, y_pred)

        # Create metrics report
        metrics_report = {}
        for i, class_name in enumerate(self.class_names):
            metrics_report[class_name] = {
                'precision': self.to_cpu(metrics_values['precision'][i]),
                'recall': self.to_cpu(metrics_values['recall'][i]),
                'f1-score': self.to_cpu(metrics_values['f1_score'][i]),
                'specificity': self.to_cpu(metrics_values['specificity'][i])
            }

        # Macro averages and overall metrics
        metrics_report['macro avg'] = {
            'precision': self.to_cpu(metrics_values['precision'].mean()),
            'recall': self.to_cpu(metrics_values['recall'].mean()),
            'f1-score': self.to_cpu(metrics_values['f1_score'].mean()),
            'specificity': self.to_cpu(metrics_values['specificity'].mean())
        }
        metrics_report['accuracy'] = self.to_cpu(metrics_values['accuracy'])
        metrics_report['auc_roc_scores'] = {class_name: self.to_cpu(score) for class_name, score in zip(self.class_names, metrics_values['auroc'])}
        metrics_report['mean_auc'] = self.to_cpu(metrics_values['auroc'].mean())
        metrics_report['average_precision_scores'] = {class_name: self.to_cpu(score) for class_name, score in zip(self.class_names, metrics_values['auprc'])}
        metrics_report['mean_average_precision'] = self.to_cpu(metrics_values['auprc'].mean())
        metrics_report['mean_f1_score'] = self.to_cpu(metrics_values['f1_score'].mean())
        metrics_report['mean_specificity'] = self.to_cpu(metrics_values['specificity'].mean())
        metrics_report['mean_sensitivity'] = self.to_cpu(metrics_values['recall'].mean())
        metrics_report['balanced_accuracy'] = self.to_cpu(metrics_values['balanced_accuracy'])

        return json.dumps(metrics_report, indent=4)


def generate_metrics_report(y_true: torch.Tensor, y_pred: torch.Tensor) -> str:
    """
    Generate a metrics report for multiclass classification using given true labels and predictions.
    """
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    calculator = MetricsCalculator(num_classes=len(class_columns), class_names=class_columns)
    return calculator.generate_metrics_report(y_true, y_pred)


# Example usage (uncomment to run):
# if __name__ == "__main__":
#     num_samples = 100
#     num_classes = 10
#     y_true = torch.randint(0, num_classes, (num_samples,))
#     y_pred = torch.randn(num_samples, num_classes)
    
#     # CPU Test
#     report_cpu = generate_metrics_report(y_true, y_pred)
#     print("CPU Report:")
#     print(report_cpu)

#     # Test on GPU if available
#     if torch.cuda.is_available():
#         y_true_gpu = y_true.cuda()
#         y_pred_gpu = y_pred.cuda()
#         report_gpu = generate_metrics_report(y_true_gpu, y_pred_gpu)
#         print("\nGPU Report:")
#         print(report_gpu)