import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader, random_split

# ------- Disease Classification Model with Softmax ------- #
class DiseaseClassifierSoftmax(nn.Module):
    def __init__(self, bert_embedding_dim=768, structured_input_dim=64, hidden1=256, hidden2=128, num_classes=19):
        super(DiseaseClassifierSoftmax, self).__init__()
        input_dim = bert_embedding_dim + structured_input_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)  # No activation here
        )

    def forward(self, combined_input):
        return self.classifier(combined_input)  # Raw logits