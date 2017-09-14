"""Child-Sum Tree-LSTM batch sentence encoder module."""
import torch
import torch.nn as nn
from cstlstm import prev_states, cell
from torch.autograd import Variable


class ChildSumTreeLSTMEncoder(nn.Module):
    """Child-Sum Tree-LSTM Encoder Module."""

    def __init__(self, embed_size, hidden_size, embeddings,
                 p_keep_input, p_keep_rnn):
        """Create a new ChildSumTreeLSTMEncoder.

        Args:
          embed_size: Integer, number of units in word embeddings vectors.
          hidden_size: Integer, number of units in hidden state vectors.
          embeddings: torch.nn.Embedding.
          p_keep_input: Float, the probability of keeping an input unit.
          p_keep_rnn: Float, the probability of keeping an rnn unit.
        """
        super(ChildSumTreeLSTMEncoder, self).__init__()

        self._embeddings = embeddings

        # Define dropout layer for embedding lookup
        self._drop_input = nn.Dropout(p=1.0 - p_keep_input)

        # Initialize the batch Child-Sum Tree-LSTM cell
        self.cell = cell.BatchChildSumTreeLSTMCell(
            input_size=embed_size,
            hidden_size=hidden_size,
            p_dropout=1.0 - p_keep_rnn).cuda()

        # Initialize previous states (to get wirings from nodes on lower level)
        self._prev_states = prev_states.PreviousStates(hidden_size)

    def forward(self, forest):
        """Get encoded vectors for each node in the forest.

        Args:
          nodes: Dictionary of structure {Integer (level_index): List (nodes)}
            where each node is represented by a ext.Node object.
          up_wirings: Dictionary of structure
            {Integer (level_index): List of Lists (up wirings)}, where the up
            wirings List is the same length as the number of nodes on the
            current level, and each sublist gives the indices of it's children
            on the lower level's node list, thus defining the upward wiring.

        Returns:
          Dictionary of hidden states for all nodes on all levels, indexed by
            level number, with the list order following that of forest.nodes[l]
            for each level, l.
        """
        outputs = {}

        # Work backwards through level indices - i.e. bottom up.
        for l in reversed(range(forest.max_level + 1)):
            # Get input word vectors for this level.
            inputs = [(self._word_vec(n.vocab_ix) if n.token
                       else self._prev_states.zero_vec())
                      for n in forest.nodes[l]]

            # Get previous hidden states for this level.
            if l == forest.max_level:
                hidden_states = self._prev_states.zero_level(
                    len(forest.nodes[l]))
            else:
                hidden_states = self._prev_states(
                    level_nodes=forest.nodes[l],
                    level_up_wirings=forest.child_ixs[l],
                    prev_outputs=outputs[l+1])

            outputs[l] = self.cell(inputs, hidden_states)

        return outputs

    def _word_vec(self, vocab_ix):
        lookup_tensor = Variable(
            torch.LongTensor([vocab_ix]),
            requires_grad=False).cuda()
        word_vec = self._embeddings(lookup_tensor)\
            .type(torch.FloatTensor)\
            .cuda()
        word_vec = self._drop_input(word_vec)
        return word_vec
