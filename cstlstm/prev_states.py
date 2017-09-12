"""For getting hidden states for nodes on lower level."""
import torch
from torch.autograd import Variable


class PreviousStates:
    """For getting previous hidden states from lower level given wirings."""

    def __init__(self, hidden_size):
        """Create a new PreviousStates.

        Args:
          hidden_size: Integer, number of units in a hidden state vector.
        """
        self.hidden_size = hidden_size

    def __call__(self, level_nodes, level_up_wirings, prev_outputs):
        """Get previous hidden states.

        Args:
          level_nodes: List of nodes on the level to be processed.
          level_up_wirings: List of Lists: the list is of the same length as the
            level_nodes list. Each sublist gives the integer indices of the
            child nodes in the node list on the previous (lower) level. This
            defines how the child nodes wire up to the parent nodes.
          prev_outputs: List of previous hidden state tuples for the level below
            from which we will select from.

        Returns:
          ?
        """
        # Count how many nodes on this level of the forest.
        level_length = len(level_nodes)

        # grab the cell states
        cell_states = self.states(
            level_nodes, level_length, prev_outputs[0], level_up_wirings)

        # grab the hidden states
        hidden_states = self.states(
            level_nodes, level_length, prev_outputs[1], level_up_wirings)

        # mind the order of returning
        return cell_states, hidden_states

    @staticmethod
    def children(prev_out, child_ixs_level_i):
        # doesn't work for empty sets of children - in which case don't call
        selector = Variable(torch.LongTensor(child_ixs_level_i)).cuda()
        return prev_out.index_select(0, selector)

    def states(self, level_nodes, level_length, prev_out, child_ixs_level):
        return [(self.zero_vec()
                 if (level_nodes[i].is_leaf or len(child_ixs_level[i]) == 0)
                 else self.children(prev_out, child_ixs_level[i]))
                for i in range(level_length)]

    def zero_level(self, level_length):
        # Doing this right away should save a bit more time each batch.
        cell_states = [self.zero_vec() for _ in range(level_length)]
        hidden_states = [self.zero_vec() for _ in range(level_length)]
        return cell_states, hidden_states

    def zero_vec(self):
        return Variable(torch.zeros(1, self.hidden_size),
                        requires_grad=False).cuda()
