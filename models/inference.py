"""Natural Language Inference model."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from cstlstm import encoder
from ext import models


class InferenceModel(models.PyTorchModel):
    """Natural language inference model."""

    def __init__(self, name, config, embedding_matrix):
        super(InferenceModel, self).__init__(name, config, embedding_matrix)

        # Define encoder.
        self.encoder = encoder.ChildSumTreeLSTMEncoder(
            self.embed_size, self.hidden_size, self.embedding,
            self.p_keep_input, self.p_keep_rnn)

        # Define dropouts
        self.drop_fc = nn.Dropout(p=1.0 - self.p_keep_fc)

        # Define MLP
        self.fc1 = nn.Linear(self.hidden_size * 4, self.hidden_size).cuda()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size).cuda()
        self.logits_layer = nn.Linear(self.hidden_size, 3).cuda()

        # Define optimizer
        params = [{'params': self.encoder.cell.parameters()},
                  {'params': self.fc1.parameters()},
                  {'params': self.fc2.parameters()},
                  {'params': self.logits_layer.parameters()}]
        if self.tune_embeddings:
            params.append({'params': self.embeddings.parameters(),
                           'lr': self.learning_rate / 10.})  # Avoid overfitting
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

        # Initialize parameters
        nn.init.xavier_uniform(self.fc1.weight.data, gain=np.sqrt(2.0))
        nn.init.xavier_uniform(self.fc2.weight.data, gain=np.sqrt(2.0))
        nn.init.xavier_uniform(self.logits_layer.weight.data, gain=1.)

    @staticmethod
    def current_batch_size(forest):
        return int(len(forest.nodes[0]) / 2)

    def forward(self, forest):
        labels = Variable(
            torch.from_numpy(np.array(forest.labels)),
            requires_grad=False).cuda()
        logits = self.logits(forest)
        loss = self.loss(logits, labels)
        predictions = self.predictions(logits).type_as(labels)
        correct = self.correct_predictions(predictions, labels)
        accuracy = self.accuracy(correct, self.current_batch_size(forest))[0]
        return predictions, loss, accuracy

    def logits(self, forest):
        # Following the DataLoader collate fn, the premises and hypotheses are
        # concatenated in the forest, in order, so splitting the root level of
        # the forest into two yields premises and hypotheses encodings.
        encodings = self.encoder.forward(forest)[0][1]  # 1 selects hs, not cs.
        premises, hypotheses = encodings.split(
            self.current_batch_size(forest), 0)

        # Mou et al. concat layer
        diff = premises - hypotheses
        mul = premises * hypotheses
        x = torch.cat([premises, hypotheses, diff, mul], 1)

        # MLP
        h1 = self.drop_fc(F.relu(self.fc1(x)))
        h2 = self.drop_fc(F.relu(self.fc2(h1)))
        logits = self.logits_layer(h2)
        return logits
