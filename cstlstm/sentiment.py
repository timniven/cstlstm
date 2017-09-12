"""Model for sentiment analysis with the Stanford Sentiment Treebank."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from cstlstm import encoder
from ext import models


class SentimentModel(models.PyTorchModel):
    """Classifier for Stanford Sentiment Treebank."""

    def __init__(self, name, config, embedding_matrix):
        super(SentimentModel, self).__init__(name, config, embedding_matrix)

        # Define encoder.
        self.encoder = encoder.ChildSumTreeLSTMEncoder(
            self.embed_size, self.hidden_size, self.embedding,
            self.p_keep_input, self.p_keep_rnn)

        # Define linear classification layer.
        self.logits_layer = nn.Linear(self.hidden_size, 5).cuda()

        # Define optimizer.
        print(len(list(self.encoder.cell.parameters())))
        params = [{'params': self.encoder.cell.parameters()},
                  {'params': self.logits_layer.parameters()}]
        if self.tune_embeddings:
            params.append({'params': self.embeddings.parameters(),
                           'lr': self.learning_rate / 10.})  # Avoid overfitting
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

        # Init params with xavier.
        nn.init.xavier_uniform(self.logits_layer.weight.data, gain=1)

    @staticmethod
    def annotated_encodings(encodings, annotation_ixs):
        selected = []
        for l in range(max(encodings.keys()) + 1):
            selected += [encodings[l][1][i] for i in annotation_ixs[l]]
        return torch.stack(selected, 0)

    @staticmethod
    def current_batch_size(forest):
        return len(forest.labels)

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
        encodings = self.encoder.forward(forest)
        annotated = self.annotated_encodings(encodings, forest.annotation_ixs)
        logits = self.logits_layer(annotated)
        return logits
