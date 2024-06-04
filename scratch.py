from src.evaluation import Bootstrap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier

# Define a simple model with Skorch
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, X):
        X = X.float()
        X = torch.relu(self.fc1(X))
        X = self.fc2(X)
        return X

if __name__ == "__main__":
    # Generate a dummy dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X = X.astype('float32')
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    net = NeuralNetClassifier(
        SimpleNet,
        max_epochs=10,
        lr=0.01,
        optimizer=optim.Adam,
        criterion=nn.CrossEntropyLoss,
        device='cuda',
        iterator_train__num_workers=3
    )

    # Initialize and run bootstrap
    bootstrap = Bootstrap(X_train, y_train, iters=10, num_processes=2, stratify=True, num_gpu=1)
    ci, final_fpr_tpr, final_y_pred = bootstrap.run(net)
    print(ci)