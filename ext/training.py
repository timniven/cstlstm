"""Base code for training."""
import time
import numpy as np
import os


# Training Class Utility Functions


def model_path(ckpt_dir, model_name, is_best):
    """Get the file path to a model checkpoint.
    Args:
      ckpt_dir: String, base directory for data.
      model_name: String, the name of the training run (unique).
      is_best: Boolean, whether this checkpoint is the best result on the
        tuning set. If True, the checkpoint name has "best" appended to it.
        Otherwise it has "latest" appended to it.
    Returns:
      String.
    """
    return os.path.join(
        ckpt_dir, '%s_%s' % (model_name, 'best' if is_best else 'latest'))


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


def progress_percent(global_step, batches_per_epoch):
    """Get progress through the epoch in percentage terms.
    Will round to a multiple of 10.
    Args:
      global_step: Integer, the current global step (can cross epochs).
      batches_per_epoch: Integer.
    Returns:
      Integer, a percentage rounded to the nearest 10.
    """
    percent = (global_step % batches_per_epoch) / batches_per_epoch * 100
    rounded = int(np.ceil(percent / 10.0) * 10)
    return rounded


def _print_dividing_lines():
    # For visuals, when reporting results to terminal.
    print('------\t\t------\t\t------\t\t------\t\t------')


def _print_epoch_start(epoch):
    _print_dividing_lines()
    print('Epoch %s\t\tloss\t\taccuracy\tavg(t)\t\tremaining' % epoch)
    _print_dividing_lines()


def report_every(batches_per_epoch):
    """How many steps before reporting results.
    We will report 10 times per epoch, so this function calculates 10% of the
    number of batches per epoch.
    Args:
      batches_per_epoch: Integer, how many batches per epoch.

    Returns:
      Integer.
    """
    return int(np.floor(batches_per_epoch / 10))


def steps_remaining(batches_per_epoch, step):
    """Determine how many steps remaining in an epoch.
    Args:
      batches_per_epoch: Integer.
      step: Integer, the global step, which can be a few epochs in.
    Returns:
      Integer.
    """
    return batches_per_epoch - (step % batches_per_epoch)


# BASE TRAINER CLASS


class TrainerBase:
    """Wraps a model and implements a train method."""

    def __init__(self, model, history, train_data, tune_data):
        """Create a new training wrapper.
        Args:
          model: any model to be trained, be it TensorFlow or PyTorch.
          history: histories.History object for storing training statistics.
          train_data: the data to be used for training.
          tune_data: the data to be used for tuning; can be list of data sets.
        """
        self.model = model
        self.history = history
        self.train_data = train_data
        self.tune_data = tune_data
        self.batches_per_epoch = train_data.batches_per_epoch
        self.report_every = report_every(train_data.batches_per_epoch)
        # Load the latest checkpoint if necessary
        if self.history.global_step > 1:
            print('Loading last checkpoint...')
            self._load_last()

    def _checkpoint(self, is_best):
        raise NotImplementedError('Deriving classes must implement.')

    def _end_epoch(self):
        self._epoch_end = time.time()
        time_taken = self._epoch_end - self._epoch_start
        avg_time, avg_loss, change_loss, avg_acc, change_acc, is_best = \
            self.history.end_epoch(time_taken)
        self._report_epoch(avg_time, avg_loss, change_loss)
        self._checkpoint(is_best)
        self.history.save()

    def _end_step(self, loss, acc):
        self.step_end = time.time()
        time_taken = self.step_end - self.step_start
        global_step, avg_time, avg_loss, avg_acc = \
            self.history.end_step(time_taken, loss, acc)
        self._report_step(global_step, avg_loss, avg_acc, avg_time)

    def _load_last(self):
        raise NotImplementedError('Deriving classes must implement.')

    def predict(self, batch):
        """Predict labels for a batch and return accuracy."""
        raise NotImplementedError('Deriving classes must implement.')

    def _report_epoch(self, avg_time, change_loss, change_acc):
        _print_dividing_lines()
        print('\t\t%s%10.5f\t%s%6.4f%%\t%s\t'
              % ('+' if change_loss > 0 else '',
                 change_loss,
                 '+' if change_acc > 0 else '',
                 change_acc,
                 pretty_time(np.average(avg_time))))

    def _report_step(self, global_step, avg_loss, avg_acc, avg_time):
        if global_step % self.report_every == 0:
            print('%s%%:\t\t'
                  '%8.5f\t'
                  '%6.4f%%\t'
                  '%s\t'
                  '%s'
                  % (progress_percent(global_step, self.batches_per_epoch),
                     avg_loss,
                     avg_acc * 100,
                     pretty_time(avg_time),
                     pretty_time(avg_time
                                 * steps_remaining(self.batches_per_epoch,
                                                   global_step))))

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

    def _stopping_condition_met(self):
        # Override this method to set a custom stopping condition.
        return False

    def train(self):
        """Run the training algorithm."""
        while not self._stopping_condition_met():
            self._start_epoch()
            for _ in range(self.train_data.batches_per_epoch):
                self._start_step()
                batch = self.train_data.next_batch()
                loss, acc = self.step(batch)
                self._end_step(loss, acc)
            self._tuning()
            self._end_epoch()

    def _tune(self, tune_set):
        cum_acc = 0.
        for _ in range(tune_set.batches_per_epoch):
            batch = tune_set.next_batch()
            acc = self.predict(batch)
            cum_acc += acc
        tuning_acc = cum_acc / tune_set.batches_per_epoch
        avg_acc, change_acc = self.history.end_tuning(tuning_acc)
        print('Average tuning accuracy: %5.3f%% (%s%5.3f%%)' %
              (avg_acc * 100,
               '+' if change_acc > 0 else '',
               change_acc * 100))

    def _tuning(self):
        self.model.eval()
        if isinstance(self.tune_data, list):
            for tune_set in self.tune_data:
                self._tune(tune_set)
        else:
            self._tune(self.tune_data)
        self.model.train()
