"""Base code for training."""
import torch
import time
import numpy as np
import os


# Utility Functions


def pretty_time(secs):
    """Get a readable string for a quantity of seconds.
    Args:
      secs: Integer, seconds.
    Returns:
      String, nicely formatted.
    """
    if secs < 60.0:
        return '%4.2f secs' % secs
    elif secs < 3600.0:
        return '%4.2f mins' % (secs / 60)
    elif secs < 86400.0:
        return '%4.2f hrs' % (secs / 60 / 60)
    else:
        return '%3.2f days' % (secs / 60 / 60 / 24)


def _print_dividing_lines():
    # For visuals, when reporting results to terminal.
    print('--------\t  ----------------\t------------------'
          '\t--------\t--------')


def _print_epoch_start(epoch):
    _print_dividing_lines()
    print('Epoch %s \t       loss       \t     accuracy     '
          '\tt(avg.)\t\tremaining'
          % epoch)
    print('        \t  last      avg.  \t  last      avg.  \t       \t')
    _print_dividing_lines()


# Base Trainer Class


class TrainerBase:
    """Wraps a model and implements a train method."""

    def __init__(self, model, history, train_loader, tune_loader, ckpt_dir):
        """Create a new training wrapper.
        Args:
          model: any model to be trained, be it TensorFlow or PyTorch.
          history: histories.History object for storing training statistics.
          train_loader: the data to be used for training.
          tune_loader: the data to be used for tuning; can be list of data sets.
          ckpt_dir: String, path to checkpoint file directory.
        """
        self.model = model
        self.history = history
        self.train_loader = train_loader
        self.tune_loader = tune_loader
        self.batches_per_epoch = len(train_loader)
        self.ckpt_dir = ckpt_dir
        # Load the latest checkpoint if necessary
        if self.history.global_step > 1:
            print('Loading last checkpoint...')
            self._load_last()

    def _checkpoint(self, is_best):
        raise NotImplementedError('Deriving classes must implement.')

    def ckpt_path(self, is_best):
        return os.path.join(
            self.ckpt_dir,
            '%s_%s' % (self.model.name, 'best' if is_best else 'latest'))

    def _end_epoch(self):
        self._epoch_end = time.time()
        time_taken = self._epoch_end - self._epoch_start
        avg_time, avg_loss, change_loss, avg_acc, change_acc, is_best = \
            self.history.end_epoch(time_taken)
        self._report_epoch(avg_time)
        self._checkpoint(is_best)
        self.history.save()

    def _end_step(self, loss, acc):
        self.step_end = time.time()
        time_taken = self.step_end - self.step_start
        global_step, avg_time, avg_loss, avg_acc = \
            self.history.end_step(time_taken, loss, acc)
        self._report_step(global_step, loss, avg_loss, acc, avg_acc, avg_time)

    def _load_last(self):
        raise NotImplementedError('Deriving classes must implement.')

    @property
    def progress_percent(self):
        percent = (self.history.global_step % self.batches_per_epoch) \
                  / self.batches_per_epoch \
                  * 100
        rounded = int(np.ceil(percent / 10.0) * 10)
        return rounded

    def _report_epoch(self, avg_time):
        _print_dividing_lines()
        print('\t\t\t\t\t\t\t%s'
              % pretty_time(np.average(avg_time)))

    @property
    def report_every(self):
        return int(np.floor(self.batches_per_epoch / 10))

    def _report_step(self, global_step, loss, avg_loss, acc, avg_acc, avg_time):
        if global_step % self.report_every == 0:
            print('%s%%:\t\t'
                  '%8.4f  %8.4f\t'
                  '%6.4f%%  %6.4f%%\t'
                  '%s\t'
                  '%s'
                  % (self.progress_percent,
                     loss,
                     avg_loss,
                     acc * 100,
                     avg_acc * 100,
                     pretty_time(avg_time),
                     pretty_time(avg_time * self.steps_remaining)))

    def _start_epoch(self):
        _print_epoch_start(self.history.global_epoch)
        self.model.train()
        self._epoch_start = time.time()

    def _start_step(self):
        self.step_start = time.time()

    def step(self, *args):
        """Take a training step.
        Calculate loss and accuracy and do optimization.
        Returns:
          Float, Float: loss, accuracy for the batch.
        """
        raise NotImplementedError('Deriving classes must implement.')

    @property
    def steps_remaining(self):
        return self.batches_per_epoch \
               - (self.history.global_step % self.batches_per_epoch)

    def _stopping_condition_met(self):
        # Override this method to set a custom stopping condition.
        return False

    def train(self):
        """Run the training algorithm."""
        while not self._stopping_condition_met():
            self._start_epoch()
            for _, batch in enumerate(self.train_loader):
                self._start_step()
                loss, acc = self.step(batch)
                self._end_step(loss, acc)
            self._tuning()
            self._end_epoch()

    def _tune(self, tune_loader):
        cum_acc = 0.
        for _, batch in enumerate(tune_loader):
            _, _, acc = self.model.forward(batch)
            cum_acc += acc
        tuning_acc = cum_acc / len(tune_loader)
        avg_acc, change_acc = self.history.end_tuning(tuning_acc)
        print('Tuning accuracy: %5.3f%%' % tuning_acc)
        print('Average tuning accuracy: %5.3f%% (%s%5.3f%%)' %
              (avg_acc * 100,
               '+' if change_acc > 0 else '',
               change_acc * 100))

    def _tuning(self):
        self.model.eval()
        if isinstance(self.tune_loader, list):
            for tune_loader in self.tune_loader:
                self._tune(tune_loader)
        else:
            self._tune(self.tune_loader)
        self.model.train()


# PyTorch Trainer


class PyTorchTrainer(TrainerBase):
    """Training wrapper for a PyTorch model."""

    def __init__(self, model, history, train_loader, tune_loader, ckpt_dir):
        """Create a new PyTorchTrainer.

        Args:
          model: a Pytorch model that inherits from torch.nn.Module.
          history: History object.
          train_loader: torch.util.data.dataloader.DataLoader.
          tune_loader: torch.util.data.dataloader.DataLoader.
        """
        super(PyTorchTrainer, self).__init__(
            model, history, train_loader, tune_loader, ckpt_dir)
        self.model.cuda()

    def _checkpoint(self, is_best):
        file_path = self.ckpt_path(False)
        torch.save(self.model.state_dict(), file_path)
        if is_best:
            print('Saving checkpoint with new best tuning accuracy...')
            file_path = self.ckpt_path(True)
            torch.save(self.model.state_dict(), file_path)

    def _load_last(self):
        file_path = self.ckpt_path(False)
        self.model.load_state_dict(torch.load(file_path))

    def step(self, batch):
        self.model.zero_grad()
        _, loss, acc = self.model.forward(batch)
        self.model.optimize(loss)
        return loss.cpu().data.numpy()[0], acc


class Saver:
    """For loading and saving models."""

    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir

    def ckpt_path(self, name, is_best):
        return os.path.join(
            self.ckpt_dir,
            '%s_%s' % (name, 'best' if is_best else 'latest'))

    def load(self, model, name, is_best):
        path = self.ckpt_path(name, is_best)
        print('Loading checkpoint at %s...' % path)
        model.load_state_dict(torch.load(path))

    def save(self, model, name, is_best):
        path = self.ckpt_path(name, is_best)
        torch.save(model.state_dict(), path)
        if is_best:
            print('Checkpointing with new best accuracy...')
            path = self.ckpt_path(name, is_best=True)
            torch.save(model.state_dict(), path)
