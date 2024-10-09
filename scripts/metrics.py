import torch
import torchmetrics
import json
from typing import List

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

        # Convert y_true to class indices if it's one-hot encoded
        if y_true.dim() == 2 and y_true.shape[1] > 1:
            y_true = y_true.argmax(dim=1)

        return {name: metric(y_pred, y_true) for name, metric in self.metrics.items()}

    def generate_metrics_report(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> str:
        # Ensure y_pred has the correct shape (batch_size, num_classes)
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)
        if y_pred.dim() == 2 and y_pred.shape[1] == 1:
            y_pred = torch.cat([1 - y_pred, y_pred], dim=1)
        
        # Ensure y_true has the correct shape (batch_size,)
        if y_true.dim() == 2:
            y_true = y_true.squeeze(1)
        
        metrics_values = self.compute_metrics(y_true, y_pred)

        metrics_report = {}

        for i, class_name in enumerate(self.class_names):
            metrics_report[class_name] = {
                'precision': self.to_cpu(metrics_values['precision'][i]),
                'recall': self.to_cpu(metrics_values['recall'][i]),
                'f1-score': self.to_cpu(metrics_values['f1_score'][i]),
                'specificity': self.to_cpu(metrics_values['specificity'][i])
            }

        metrics_report['accuracy'] = self.to_cpu(metrics_values['accuracy'])
        metrics_report['macro avg'] = {
            'precision': self.to_cpu(metrics_values['precision'].mean()),
            'recall': self.to_cpu(metrics_values['recall'].mean()),
            'f1-score': self.to_cpu(metrics_values['f1_score'].mean()),
            'specificity': self.to_cpu(metrics_values['specificity'].mean())
        }
        metrics_report['auc_roc_scores'] = {class_name: self.to_cpu(score) for class_name, score in zip(self.class_names, metrics_values['auroc'])}
        metrics_report['average_precision_scores'] = {class_name: self.to_cpu(score) for class_name, score in zip(self.class_names, metrics_values['auprc'])}
        metrics_report['mean_auc'] = self.to_cpu(metrics_values['auroc'].mean())
        metrics_report['mean_average_precision'] = self.to_cpu(metrics_values['auprc'].mean())

        return json.dumps(metrics_report, indent=4)

def generate_metrics_report(y_true: torch.Tensor, y_pred: torch.Tensor) -> str:
    class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    calculator = MetricsCalculator(num_classes=len(class_columns), class_names=class_columns)
    return calculator.generate_metrics_report(y_true, y_pred)

# # Example usage
# if __name__ == "__main__":
#     # Simulating some dummy data
#     num_samples = 100
#     num_classes = 10
#     y_true = torch.randint(0, num_classes, (num_samples,))
#     y_pred = torch.randn(num_samples, num_classes)
    
#     # Test on CPU
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