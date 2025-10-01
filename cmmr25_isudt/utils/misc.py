import numpy as np


def midi2frequency(midi, base_frequency=440.0):
    return base_frequency * 2**((midi - 69) / 12)


def frequency2midi(frequency, base_frequency=440.0):
    return 69 + 12 * np.log2(frequency / base_frequency)