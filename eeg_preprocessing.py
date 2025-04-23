import argparse
import math
import numpy as np
import os
import scipy.io as sio


def td_granulation(x, window):
    """Time-domain granulation based on the minimum and maximum sample values of a granulation window."""
    n_trials, n_channels, n_samples = x.shape
    n_windows = n_samples // window  # assume that n_samples is multiples of n_window
    y = x.reshape(n_trials, n_channels, n_windows, window)
    return np.stack(
        (np.min(y, axis=3), np.max(y, axis=3)), axis=3)


def ps_granulation(x, window):
    """Phase-space granulation based on the mininum and maximum of normalized gradient angles of a granulation window."""
    n_trials, n_channels, n_samples = x.shape
    n_windows = n_samples // window  # assume that n_samples is multiples of n_window
    y = np.arctan(x.reshape((n_trials, n_channels, n_windows, window))[:, :, :, 1:] / np.tile(np.arange(1, window), (
    n_trials, n_channels, n_windows, 1)))  # gradient angles
    return (np.stack((np.min(y, axis=3), np.max(y, axis=3)),
                     axis=3) + 0.5 * math.pi) / math.pi  # normalize before return



def eeg_preprocessing(eeg_path, window):
    """Preprocess EEG data based on GrC."""
    eeg_dir = os.path.dirname(eeg_path)
    eeg_file = os.path.basename(eeg_path)

    eeg_data = sio.loadmat(eeg_path)
    datasets = dict.fromkeys(['c1', 'c2'])
    for i in range(1, 3):
        datasets['c' + str(i)] = eeg_data['C' + str(i)]
        np.save(f'c{str(i)}_time.npy', datasets['c' + str(i)])

    # time-domain granules (td_granules)
    td_granules = dict.fromkeys(['c1', 'c2'])
    for i in range(1, 3):
        fname = os.path.join(eeg_dir, 'c' + str(i) + '_td_granules.npy')
        if not os.path.isfile(fname):
            td_granules['c' + str(i)] = td_granulation(datasets['c' + str(i)], window)
            np.save(fname, td_granules['c' + str(i)])

    # phase-space granules (ps_granules)
    ps_granules = dict.fromkeys(['c1', 'c2'])
    for i in range(1, 3):
        fname = os.path.join(eeg_dir, 'c' + str(i) + '_ps_granules.npy')
        if not os.path.isfile(fname):
            ps_granules['c' + str(i)] = ps_granulation(datasets['c' + str(i)], window)
            np.save(fname, ps_granules['c' + str(i)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-E",
        "--eeg_path",
        help="path of a EEG data file; default is './data/BCI_data.mat'",
        default='./Data/BCI_data.mat',
        type=str)
    parser.add_argument(
        "-T",
        "--testing_split",
        help="ratio of testing dataset; default is 0.2",
        default=50,
        type=float)
    parser.add_argument(
        "-W",
        "--window",
        help="granulation window size; default is 50",
        default=50,
        type=int)
    args = parser.parse_args()

    # set variables using command-line arguments
    eeg_path = args.eeg_path
    testing_split = args.testing_split
    window = args.window

    # preprocess EEG data
    eeg_preprocessing(eeg_path, window)

    # print a summary information
    print('Completed the preprocessing of EEG data based on GrC.')
