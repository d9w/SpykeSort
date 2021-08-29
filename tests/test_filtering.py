#!/usr/bin/env python3
from spykesort import filtering
from scipy import signal


def test_butter_bandpass():
    fs = 1000
    lowcut = 25 / (0.5 * fs)
    highcut = 100 / (0.5 * fs)
    b, a = filtering.butter_bandpass(25, 100, fs)
    sb, sa = signal.butter(5, [lowcut, highcut], btype='band')
    assert (b == sb).all()
    assert (a == sa).all()
