import random
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple, TypeVar, Union

import matplotlib.mlab as mlab
import numpy as np
from matplotlib.pyplot import Axes, Figure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure

SongID = TypeVar("SongID")


def rand_clip(digital: np.ndarray, new: float, fs: int = 44100) -> np.ndarray:
    """Produce a random "clip" of a digital signal

    Parameters
    ----------
    digital : numpy.ndarray, shape=(T, )
        digital signal to be clipped

    new : float
        The duration (seconds) of the resulting clip

    fs : int, optional (default=44100)
        The sampling rate of the digital signal

    Returns
    -------
    digital : numpy.ndarray, shape=(T_clipped, )
        Clipped digital signal, sampled from a random starting point"""
    if new is None:
        return digital

    old = len(digital)
    new = int(round(new * fs))
    assert 0 < new <= old
    start = random.randint(0, old - new - 1)
    return digital[start : start + new]


def digital_to_spec(
    digital: np.ndarray, fs: float, frac_cut: float, plot: bool = False
) -> Union[
    Tuple[np.ndarray, float], Tuple[np.ndarray, float, Figure, Axes, float, float]
]:
    """Produces a spectrogram and a cut-off intensity to yield the
    specified fraction of data.

    Parameters
    ----------
    digital : numpy.ndarray, shape=(Ts, )
        The sampled audio-signal.

    fs : float
        The sample-frequency used to create the digital signal.

    frac_cut : float
        The fractional portion of intensities for which the cutoff is selected.
        E.g. frac_cut=0.8 will produce a cutoff intensity such that the bottom 80%
        of intensities are excluded.

    plot : bool
        If True, produce a plot of the spectrogram and return the
        matplotlib fig & ax objects.

    Returns
    -------
    Union[Tuple[numpy.ndarray, float]]
        The spectrogram and the desired cutoff

        If plot=True, then: (spectrogram, cutoff, fig, ax, df, dt)
        is returned. Where (fig, ax) are the plot objects, and df
        and dt are the frequency and time units associated with the
        spectrogram bins."""
    if digital.max() <= 1:
        digital = digital * 2 ** 15
    assert 0.0 <= frac_cut <= 1.0

    kwargs = dict(NFFT=4096, Fs=fs, window=mlab.window_hanning, noverlap=int(4096 / 2))
    if not plot:
        S, freqs, times = mlab.specgram(digital, **kwargs)
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        S, freqs, times, im = ax.specgram(digital, **kwargs)
        fig.colorbar(im)

    # log-scaled Fourier amplitudes have a much more gradual distribution
    # for audio data.
    np.clip(S, a_min=1e-20, a_max=None, out=S)
    np.log(S, out=S)

    # Compute the cumulative distribution over Fourier component log-amplitudes.
    # Use this to identify the threshold amplitude below which `frac_cutoff`
    # proportion of amplitudes lie.

    a = np.sort(S.flatten())
    cutoff = a[int(len(a) * frac_cut)]

    if not plot:
        return S, cutoff
    else:
        df = freqs[1] - freqs[0]
        dt = times[1] - times[0]
        return S, cutoff, fig, ax, df, dt


def local_peaks(
    log_spectrogram: np.ndarray, amp_min: float, p_nn: int
) -> List[Tuple[float, float]]:
    """
    Parameters
    ----------
    log_spectrogram : numpy.ndarray, shape=(n_freq, n_time)
        Log-scaled spectrogram. Columns are the periodograms of successive segments of a
        frequency-time spectrum.

    amp_min : float
        Amplitude threshold applied to local maxima

    p_nn : int
        Number of cells around an amplitude peak in the spectrogram in order

    Returns
    -------
    List[Tuple[float, float]]
        Time and frequency values of local peaks in spectogram. Sorted by ascending
        frequency and then time."""
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, p_nn)

    # find local maxima using our filter shape
    local_max = (
        maximum_filter(log_spectrogram, footprint=neighborhood) == log_spectrogram
    )  # where spectrogram aligns with local maxes
    foreground = log_spectrogram >= amp_min
    # Boolean mask of S with True at peaks that are in foreground, and are above the threshold
    detected_peaks = local_max & foreground

    # Extract peaks; encoded in terms of time and freq bin indices.
    # dt and df are always the same size for the spectrogram that is produced,
    # so the bin indices consistently map to the same physical units:
    # t_n = n*dt, f_m = m*df (m and n are integer indices)
    # Thus we can codify our peaks with integer bin indices instead of their
    # physical (t, f) coordinates. This makes storage and compression of peak
    # locations much simpler.

    # take transpose so peaks are ordered by time then frequency
    ts, fs = (i.astype(np.int16) for i in np.where(detected_peaks.T))
    return list(zip(ts, fs))


def peaks_to_fingerprints(
    peaks: Sequence[Tuple[float, float]], fan_value: int
) -> Iterable[Tuple[Tuple[float, float, float], float]]:
    """Given the time-frequency locations of spectrogram peaks, generates
    'fingerprint' features.

    Parameters
    ----------
    peaks : Sequence[Tuple[float, float]]
        A sequence of time-frequency pairs

    fan_value : int
        Given a peak, `fan_value` indicates the number of subsequent peaks
        to be used to form fingerprint features.

    Yields
    ------
    Tuple[Tuple[float, float, float], float]
        ((f_{n}, f_{n+j}, t_{n+j} - t_{n}), t_{n})
        The frequency value of peak n, peak n+j, their time-offset, along with the
        time at which peak n occurred."""

    assert 1 <= fan_value
    for n, (t1, f1) in enumerate(peaks):
        for t2, f2 in peaks[n + 1 : n + fan_value + 1]:
            yield ((f1, f2, t2 - t1), t1)


def fingerprints_to_matches(
    sample_fingerprints: Iterable[Tuple[Tuple[float, float, float], float]],
    database: Dict[Tuple[float, float, float], List[Tuple[SongID, float]]],
) -> Tuple[SongID, float]:
    """Generates database matches from all of a sample's fingerprints.

    Parameters
    ----------
    sample_fingerprints : Iterable[Tuple[Tuple[float, float, float], float]]
        ((f_{n}, f_{n+j}, dt), t_{n})
        The frequency value of peak n and peak n+j, along with the time at which peak n occurred.

    database : Dict[Tuple[float, float, float], List[Tuple[Any, float]]
        (freq_{n}, freq_{n+j, dt} -> [(song_ID, t), ... ]
        A dictionary that maps frequency peak-pairs and their offset to a list of all the
        song IDs containing that signature, and the time at which the signature occurred
        in the song.

    Yields
    ------
    Tuple[song_ID, dt]
        A song ID that had a matching peak-pair signature, and the time offset between when
        the signature occurred in the song versus the sample."""
    for f1_f2_dt, t_sample in sample_fingerprints:
        o = database.get(f1_f2_dt)
        if o is not None:
            for s_id, t_song in o:
                yield (s_id, t_song - t_sample)


def matches_to_best_match(matches: Iterable[Tuple[SongID, float]]) -> SongID:
    """Determines the song-ID that has the most consistent fingerprint-offset

    Parameters
    ----------
    matches : Iterable[Tuple[song_ID, dt]]
        A song-ID that had a match with the sample, and the time-offset between their
        matching signatures.

    Returns
    -------
    SongID
        The song-ID with the most common time-offset with the sample."""
    cntr = Counter(matches)
    if not cntr:
        return None

    item, cnt = cntr.most_common(1)[0]
    return item[0]
