import torch.nn.functional as F
import torch.nn as nn
import torch


class Classifier(nn.Module):

    def __init__(self, hidden_size, num_class):
        super().__init__()

        self.linear = nn.Linear(hidden_size, num_class, bias=True)
        self.reset_parameters()

    def forward(self, x):
        logits = self.linear(x)
        prediction = torch.argmax(logits, dim=1)

        return logits, prediction

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


