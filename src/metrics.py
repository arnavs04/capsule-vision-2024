import torch
import torchmetrics
from torch import nn
import json
from typing import List


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        p_t = torch.exp(-CE_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class MetricsCalculator:
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.metrics = None

    def _initialize_metrics(self, device):
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
        return t.cpu().tolist() if isinstance(t, torch.Tensor) else t

    def compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        device = y_true.device
        y_pred = y_pred.to(device)

        if self.metrics is None or next(iter(self.metrics.values())).device != device:
            self._initialize_metrics(device)

        # For AUROC, Average Precision, and probability-based metrics, use raw logits.
        y_pred_softmax = torch.softmax(y_pred, dim=1)

        # For class-prediction-based metrics (Accuracy, Precision, Recall), use argmax of logits.
        y_pred_classes = torch.argmax(y_pred, dim=1)

        # Ensure y_true is a long tensor with shape [batch_size]
        y_true = y_true.long()
        if y_true.dim() == 2:
            y_true = y_true.squeeze(1)

        # Compute the metrics (class-prediction metrics on argmax, probability-based on softmax)
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

        # Balanced accuracy manually using recall
        balanced_accuracy = metrics_values['recall'].mean()  # Mean recall across all classes
        metrics_values['balanced_accuracy'] = balanced_accuracy

        return metrics_values

    def generate_metrics_report(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> str:
        metrics_values = self.compute_metrics(y_true, y_pred)

        metrics_report = {}

        # Class-wise metrics
        for i, class_name in enumerate(self.class_names):
            metrics_report[class_name] = {
                'precision': self.to_cpu(metrics_values['precision'][i]),
                'recall': self.to_cpu(metrics_values['recall'][i]),
                'f1-score': self.to_cpu(metrics_values['f1_score'][i]),
                'specificity': self.to_cpu(metrics_values['specificity'][i])
            }

        # Macro (mean) averages for class-wise metrics
        metrics_report['macro avg'] = {
            'precision': self.to_cpu(metrics_values['precision'].mean()),
            'recall': self.to_cpu(metrics_values['recall'].mean()),
            'f1-score': self.to_cpu(metrics_values['f1_score'].mean()),
            'specificity': self.to_cpu(metrics_values['specificity'].mean())
        }

        # Overall accuracy
        metrics_report['accuracy'] = self.to_cpu(metrics_values['accuracy'])

        # AUROC per class and mean
        metrics_report['auc_roc_scores'] = {class_name: self.to_cpu(score) for class_name, score in zip(self.class_names, metrics_values['auroc'])}
        metrics_report['mean_auc'] = self.to_cpu(metrics_values['auroc'].mean())

        # Average precision (AUPRC) per class and mean
        metrics_report['average_precision_scores'] = {class_name: self.to_cpu(score) for class_name, score in zip(self.class_names, metrics_values['auprc'])}
        metrics_report['mean_average_precision'] = self.to_cpu(metrics_values['auprc'].mean())

        # Mean values for F1, Specificity, Sensitivity
        metrics_report['mean_f1_score'] = self.to_cpu(metrics_values['f1_score'].mean())
        metrics_report['mean_specificity'] = self.to_cpu(metrics_values['specificity'].mean())
        metrics_report['mean_sensitivity'] = self.to_cpu(metrics_values['recall'].mean())  # Sensitivity is equivalent to recall

        # Balanced accuracy
        metrics_report['balanced_accuracy'] = self.to_cpu(metrics_values['balanced_accuracy'])

        return json.dumps(metrics_report, indent=4)


def generate_metrics_report(y_true: torch.Tensor, y_pred: torch.Tensor) -> str:
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