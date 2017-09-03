"""Child-Sum Tree-LSTM batch sentence encoder module."""
import torch
import torch.nn as nn
from cstlstm import prev_states, cell
from torch.autograd import Variable


class ChildSumTreeLSTMEncoder(nn.Module):
    """Child-Sum Tree-LSTM Encoder Module.

    This module encodes sentences, returning hidden states for each sentence.
    """

    def __init__(self, embed_size, hidden_size, embedding,
                 p_keep_input, p_keep_rnn):
        """Create a new ChildSumTreeLSTMEncoder.

        Args:
          embed_size: Integer, number of units in word embeddings vectors.
          hidden_size: Integer, number of units in hidden state vectors.
          embedding: torch.nn.Embedding.
          p_keep_input: Float, the probability of keeping an input unit.
          p_keep_rnn: Float, the probability of keeping an rnn unit.
        """
        super(ChildSumTreeLSTMEncoder, self).__init__()

        # Save local reference to the embedding
        self.embedding = embedding

        # Define dropout layer for embedding lookup
        self.drop_input = nn.Dropout(p=1.0 - p_keep_input)

        # Initialize the batch Child-Sum Tree-LSTM cell
        self.cell = cell.BatchChildSumTreeLSTMCell(
            input_size=embed_size,
            hidden_size=hidden_size,
            p_dropout=1.0 - p_keep_rnn).cuda()

        # Initialize previous states (to get wirings from nodes on lower level)
        self.prev_states = prev_states.PreviousStates(hidden_size)

    def forward(self, nodes, up_wirings):
        """Get encoded vectors for each sentence in the batch.

        Args:
          nodes: Dictionary of structure {Integer (level_index): List (nodes)}
            where each node is represented by a ext.Node object.
          up_wirings: Dictionary of structure
            {Integer (level_index): List of Lists (up wirings)}, where the up
            wirings List is the same length as the number of nodes on the
            current level, and each sublist gives the indices of it's children
            on the lower level's node list, thus defining the upward wiring.

        Returns:
          Tensor of shape 2 * batch_size x hidden_size.
        """
        # Dictionary to hold the outputs of each level.
        outputs = {}

        # Determine the max level in the forest.
        max_level = max(nodes.keys())

        # Work backwards through level indices - i.e. bottom up.
        for l in reversed(range(max_level + 1)):
            # Get input word vectors for this level.
            inputs = [(self.word2vec(n.vocab_ix) if n.token
                       else self.prev_states.zero_vec())
                      for n in nodes[l]]

            # Get previous hidden states for this level.
            if l == max_level:
                hidden_states = self.prev_states.zero_level(len(nodes[l]))
            else:
                hidden_states = self.prev_states(
                    level=l,
                    max_level=max_level,
                    level_nodes=nodes[l],
                    level_up_wirings=up_wirings[l],
                    prev_outputs=outputs[l+1])

            # Calculate the new outputs and store for subsequent processing.
            outputs[l] = self.cell(inputs, hidden_states)
        return outputs[0][1]

    def word2vec(self, vocab_ix):
        lookup_tensor = Variable(
            torch.LongTensor([vocab_ix]),
            requires_grad=False).cuda()
        word_vec = self.embeddings(lookup_tensor).type(torch.FloatTensor).cuda()
        word_vec = self.drop_input(word_vec)
        return word_vec
