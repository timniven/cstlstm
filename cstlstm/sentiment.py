"""Model for sentiment analysis."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from cstlstm import encoder


class SentimentModel(nn.Module):
    """Classifier for Stanford Sentiment Treebank."""

    def __init__(self, config, embedding_matrix):
        super(SentimentModel, self).__init__()

        # Save config settings locally.
        self.config = config
        for key in config.keys():
            setattr(self, key, config[key])

        # Define dropout layer for MLP.
        self.drop_fc = nn.Dropout(p=1.0 - self.config.p_keep_fc).cuda()

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

        # Define MLP layers.
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size).cuda()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size).cuda()
        self.logits_layer = nn.Linear(self.hidden_size, 5).cuda()

        # Define optimizer.
        params = [{'params': self.encoder.cell.parameters()},
                  {'params': self.fc1.parameters()},
                  {'params': self.fc2.parameters()},
                  {'params': self.logits_layer.parameters()}]
        if self.tune_embeddings:
            params.append({'params': self.embeddings.parameters(),
                           'lr': self.learning_rate / 10.})  # Avoid overfitting
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

        # Init params with xavier.
        pass

    def accuracy(self, correct_predictions):
        correct = correct_predictions.cpu().sum().data.numpy()
        return correct / float(self.current_batch_size(self.batch))

    def _biases(self):
        return [p for n, p in self.named_parameters() if n in ['bias']]

    def correct_predictions(self, predictions, labels):
        return predictions.eq(labels)

    def forward(self, batch):
        self.batch = batch
        labels = Variable(
            torch.from_numpy(batch.labels),
            requires_grad=False).cuda()
        logits = self.logits(batch)
        loss = self.loss(logits, labels)
        predictions = self.predictions(logits).type_as(labels)
        correct = self.correct_predictions(predictions, labels)
        accuracy = self.accuracy(correct)[0]
        return predictions, loss, accuracy

    def logits(self, batch):
        encoded_sents = self.encoder.forward(batch.forest.nodes,
                                             batch.forest.child_ixs)
        h1 = self.drop_fc(F.relu(self.fc1(encoded_sents)))
        h2 = self.drop_fc(F.relu(self.fc2(h1)))
        logits = self.logits_layer(h2)
        return logits

    def loss(self, logits, labels):
        criterion = torch.nn.CrossEntropyLoss().cuda()
        loss = criterion(logits, labels)
        return loss

    def optimize(self, loss):
        loss.backward()
        self.optimizer.step()

    def predictions(self, logits):
        return logits.max(1)[1]

    def _weights(self):
        return [p for n, p in self.named_parameters() if n in ['weight']]

    def zero_grad(self):
        self.optimizer.zero_grad()
