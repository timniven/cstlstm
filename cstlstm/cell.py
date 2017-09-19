"""Batch Child-Sum Tree-LSTM cell for parallel processing of nodes per level."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchChildSumTreeLSTMCell(nn.Module):
    """Child-Sum Tree-LSTM Cell implementation for mini batches.

    Based on https://arxiv.org/abs/1503.00075.
    Equations on p.3 as follows.

    .. math::

        \begin{array}{ll}
          \tilde{h_j} = \sum_{k \in C(j)} h_k \\
          i_j = \mathrm{sigmoid}(W^{(i)} x_j + U^{(i)} \tilde{h}_j + b^{(i)}) \\
          f_{jk} = \mathrm{sigmoid}(W^{(f)} x_j + U^{(f)} h_k + b^{(f)}) \\
          o_j = \mathrm{sigmoid}(W^{(o)} x_j + U^{(o)} \tilde{h}_j + b^{(o)}) \\
          u_j = \tanh(W^{(u)} x_j + U^{(u)} \tilde{h}_j + b^{(u)}) \\
          c_j = i_j \circ u_j + \sum_{k \in C(j)} f_{jk} \circ c_k \\
          h_j = o_j \circ \tanh(c_j)
        \end{array}
    """

    def __init__(self, input_size, hidden_size, p_dropout):
        """Create a new ChildSumTreeLSTMCell.

        Args:
          input_size: Integer, the size of the input vector.
          hidden_size: Integer, the size of the hidden state to return.
          dropout: torch.nn.Dropout module.
        """
        super(BatchChildSumTreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=p_dropout)
        self.W_combined = nn.Parameter(
            torch.Tensor(input_size + hidden_size, 3 * hidden_size),
            requires_grad=True)
        self.b_combined = nn.Parameter(
            torch.zeros(1, 3 * hidden_size),
            requires_grad=True)
        self.W_f = nn.Parameter(
            torch.Tensor(input_size, hidden_size),
            requires_grad=True)
        self.U_f = nn.Parameter(
            torch.Tensor(hidden_size, hidden_size),
            requires_grad=True)
        self.b_f = nn.Parameter(
            torch.zeros(1, hidden_size),
            requires_grad=True)
        nn.init.xavier_uniform(self.W_combined, gain=1.0)
        nn.init.xavier_uniform(self.W_f, gain=1.0)
        nn.init.xavier_uniform(self.U_f, gain=1.0)

    def forward(self, inputs, previous_states):
        """Calculate the next hidden state given the inputs.

        This is for custom control over a batch, designed for efficiency.
        I hope it is efficient...

        Args:
          inputs: List of Tensors of shape (1, input_size) - row vectors.
          previous_states: Tuple of (List, List), being cell_states and
            hidden_states respectively. Inside the lists, for nodes with
            multiple children, we expect they are already concatenated into
            matrices.

        Returns:
          cell_states, hidden_states: being state tuple, where both states are
             row vectors of length hidden_size.
        """
        # prepare the inputs
        cell_states = previous_states[0]
        hidden_states = previous_states[1]
        inputs_mat = torch.cat(inputs)
        h_tilde_mat = torch.cat([torch.sum(h, 0).expand(1, self.hidden_size)
                                 for h in hidden_states],
                                dim=0)
        prev_c_mat = torch.cat(cell_states, 0)
        big_cat_in = torch.cat([inputs_mat, h_tilde_mat], 1)

        # process in parallel those parts we can
        big_cat_out = big_cat_in.mm(self.W_combined) + self.b_combined.expand(
            big_cat_in.size()[0],
            3 * self.hidden_size)
        z_i, z_o, z_u = big_cat_out.split(self.hidden_size, 1)

        # apply dropout to u, like the Fold boys
        z_u = self.dropout(z_u)

        # forget gates
        f_inputs = inputs_mat.mm(self.W_f)
        # we can concat the matrices along the row axis,
        # but we need to calculate cumsums for splitting after

        # NOTE: I could probably pass this information from pre-processing
        # yes, I think that's the idea: move this out. Test it out there.
        # then come back to here. That's my next job. And moving the other
        # stuff out of the CSTLSTM model.
        lens = [t.size()[0] for t in hidden_states]
        start = [sum([lens[j] for j in range(i)]) for i in range(len(lens))]
        end = [start[i] + lens[i] for i in range(len(lens))]

        # we can then go ahead and concatenate for matmul
        prev_h_mat = torch.cat(hidden_states, 0)
        f_hiddens = prev_h_mat.mm(self.U_f)
        # compute the f_jks by expanding the inputs to the same number
        # of rows as there are prev_hs for each, then just do a simple add.
        f_inputs_split = f_inputs.split(1, 0)
        f_inputs_expanded = [f_inputs_split[i].expand(lens[i], self.hidden_size)
                             for i in range(len(lens))]
        f_inputs_ready = torch.cat(f_inputs_expanded, 0)
        f_jks = F.sigmoid(
            f_inputs_ready + f_hiddens + self.b_f.expand(
                f_hiddens.size()[0], self.hidden_size))

        # cell and hidden state
        fc_mul = f_jks * prev_c_mat
        split_fcs = [fc_mul[start[i]:end[i]] for i in range(len(lens))]
        fc_term = torch.cat([torch.sum(item, 0).expand(1, self.hidden_size)
                             for item in split_fcs])
        c = F.sigmoid(z_i) * F.tanh(z_u) + fc_term
        h = F.sigmoid(z_o) * F.tanh(c)

        return c, h
