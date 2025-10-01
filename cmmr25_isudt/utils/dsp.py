import numpy as np


def amp2db(
    amp: float,
) -> float:
    """
    Convert amplitude to decibels.

    Args:
        amp (float): Amplitude.

    Returns:
        float: Decibels.
    """
    return 20 * np.log10(amp)