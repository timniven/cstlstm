"""Model for sentiment analysis with the Stanford Sentiment Treebank."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from cstlstm import encoder


class SentimentModel(nn.Module):
    """Classifier for Stanford Sentiment Treebank."""

    def __init__(self, name, config, embedding_matrix):
        super(SentimentModel, self).__init__()

        self.name = name

        # Save config settings locally.
        self.config = config
        for key in config.keys():
            setattr(self, key, config[key])

        # Define embedding.
        self.embedding = nn.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1], sparse=False)
        embedding_tensor = torch.from_numpy(embedding_matrix)
        self.embedding.weight = nn.Parameter(
            embedding_tensor,
            requires_grad=self.tune_embeddings)
        self.embedding.cuda()

        # Define encoder.
        self.encoder = encoder.ChildSumTreeLSTMEncoder(
            self.embed_size, self.hidden_size, self.embedding,
            self.p_keep_input, self.p_keep_rnn)

        # Define linear classification layer.
        self.logits_layer = nn.Linear(self.hidden_size, 5).cuda()

        # Define loss
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

        # Define optimizer.
        params = [{'params': self.encoder.cell.parameters()},
                  {'params': self.logits_layer.parameters()}]
        if self.tune_embeddings:
            params.append({'params': self.embeddings.parameters(),
                           'lr': self.learning_rate / 10.})  # Avoid overfitting
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

        # Init params with xavier.
        nn.init.xavier_uniform(self.logits_layer.weight.data, gain=1)

    def accuracy(self, correct_predictions):
        correct = correct_predictions.cpu().sum().data.numpy()
        return correct / float(self.current_batch_size)

    @staticmethod
    def annotated_encodings(encodings, annotation_ixs):
        selected = []
        for l in range(max(encodings.keys()) + 1):
            selected += [encodings[l][1][i] for i in annotation_ixs[l]]
        return torch.stack(selected, 0)

    def _biases(self):
        return [p for n, p in self.named_parameters() if n in ['bias']]

    @staticmethod
    def correct_predictions(predictions, labels):
        return predictions.eq(labels)

    def forward(self, forest):
        self.current_batch_size = len(forest.labels)
        labels = Variable(
            torch.from_numpy(np.array(forest.labels)),
            requires_grad=False).cuda()
        logits = self.logits(forest)
        loss = self.loss(logits, labels)
        predictions = self.predictions(logits).type_as(labels)
        correct = self.correct_predictions(predictions, labels)
        accuracy = self.accuracy(correct)[0]
        return predictions, loss, accuracy

    def logits(self, forest):
        encodings = self.encoder.forward(forest)
        annotated = self.annotated_encodings(encodings, forest.annotation_ixs)
        logits = self.logits_layer(annotated)
        return logits

    def loss(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss

    def optimize(self, loss):
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def predictions(logits):
        return logits.max(1)[1]

    def _weights(self):
        return [p for n, p in self.named_parameters() if n in ['weight']]

    def zero_grad(self):
        self.optimizer.zero_grad()
