# Import necessary libraries
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from engine import train  # Assuming the train function is defined in engine.py
from metrics import generate_metrics_report  # Assuming the function is defined in metrics.py

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create dummy dataset
num_samples = 100
num_classes = 10

# Generate random data
X_dummy = torch.randn(num_samples, 3, 224, 224)  # Dummy images (3 channels, 224x224)
y_dummy = torch.randint(0, num_classes, (num_samples,))  # Random labels

# Create DataLoader
dummy_dataset = TensorDataset(X_dummy, y_dummy)
dummy_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)

# Define a simple model for testing (you can replace this with your actual model)
class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)  # Simple Linear layer for testing

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.fc(x)

# Initialize the model, loss function, and optimizer
model = SimpleModel(num_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Test training function (run a single epoch)
def test_train():
    model.train()  # Set the model to training mode
    for epoch in range(1):  # Just one epoch for testing
        with tqdm(total=len(dummy_loader), desc="Training Step") as pbar:
            for X, y in dummy_loader:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()  # Reset gradients
                y_pred = model(X)  # Forward pass
                loss = loss_fn(y_pred, y)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

# Test the metrics generation function
# Test the metrics generation function
def test_metrics():
    # Simulate true labels
    y_true = torch.randint(0, num_classes, (20,))  # True labels

    # Simulated logits for predictions (batch_size, num_classes)
    y_pred_logits = torch.randn(20, num_classes)  # Simulated logits for predictions

    # Generate predicted probabilities using softmax
    y_pred_probs = torch.softmax(y_pred_logits, dim=1)  # Shape: [20, num_classes]

    # Generate metrics report
    metrics_report = generate_metrics_report(y_true, y_pred_probs)  # Pass probabilities instead of class indices
    print("Metrics Report:\n", metrics_report)

# Run the test functions
if __name__ == "__main__":
    print("Testing training function...")
    test_train()
    print("\nTesting metrics generation...")
    test_metrics()