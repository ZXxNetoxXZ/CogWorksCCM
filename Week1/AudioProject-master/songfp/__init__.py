"""`songfp` is a Python package that performs song-fingerprint matching.
**This is a re-implementation/simplification of the [dejavu](https://github.com/worldveil/dejavu) project.**
Authorship should effectively be attributed to Will Drevo (GitHub user worldveil), who created dejavu.

In effect, `songfp` provides a service similar to popular song-recognition programs like Shazaam.
It can "listen" to a song, and match it against a database of song fingerprints, which is populated by the user.

`songfp` was created as a prototype for the CogWorks summer program in the
Beaver Works Summer Institute at MIT. It was developed by Ryan Soklaski."""

from pathlib import Path
from typing import Tuple, Union

import numpy as _np
from matplotlib.pyplot import Axes, Figure

import librosa as _librosa
from microphone import record_audio

from .database import list_songs, load_song_db
from .functions import digital_to_spec as _digital_to_spec
from .functions import fingerprints_to_matches as _fingerprints_to_matches
from .functions import local_peaks as _local_peaks
from .functions import matches_to_best_match as _matches_to_best_match
from .functions import peaks_to_fingerprints as _peaks_to_fingerprints

__all__ = [
    "list_songs",
    "match_sample",
    "match_recording",
    "plot_recording",
    "plot_song",
]

__version__ = "0.0"


@load_song_db
def match_sample(sample_digital: _np.ndarray, fs: int) -> str:
    """ Given a digital signal, produce the best match from the fingerprint database.

    Parameters
    ----------
    sample_digital : numpy.ndarray, shape=(T,)
        The digital signal

    fs : int
        The sampling rate for the signal

    Returns
    -------
    str
        The song-ID for the best match. `None` if no mat"""
    from .database import database

    if not database:
        print("No songs to match - your _database is empty!")
        return "no match... your database is empty!"
    peaks = _local_peaks(*_digital_to_spec(sample_digital, fs, frac_cut=0.77), p_nn=20)
    fingerprints = _peaks_to_fingerprints(peaks, fan_value=15)
    matches = _fingerprints_to_matches(fingerprints, database.pair_mapping)
    song_id = _matches_to_best_match(matches)

    if song_id is None:
        return "no match..."

    name, artist = database.song_list[song_id]
    return name + ("" if artist is None else " by {}".format(artist))


def match_recording(time: float) -> str:
    """ Record a song for the specified time, and return the best match from the fingerprint database.

    Parameters
    ----------
    time : float
        The time, in seconds, for which the microphone will record the sample.

    Returns
    -------
    str
        The song-ID for the best match"""

    frames, sample_rate = record_audio(time)
    digital_data = _np.hstack([_np.frombuffer(i, _np.int16) for i in frames])
    return match_sample(digital_data, sample_rate)


def plot_song(
    song: Union[str, Path, _np.ndarray], with_peaks: bool = True
) -> Tuple[Figure, Axes]:
    """ Plot a spectrogram and fingerprint features for a song.

    Parameters
    ----------
    song : Union[str, pathlib.Path, numpy.ndarray]
        The filepath to a song-file, or the digital signal itself.

    with_peaks : bool
        If True, include peak-value scatter-points

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes """
    from microphone.config import settings
    from pathlib import Path

    if isinstance(song, (str, Path)):
        digital, fs = _librosa.load(str(song), sr=44100, mono=True)
    elif isinstance(song, _np.ndarray):
        digital = song
        fs = settings.rate
    else:
        raise TypeError("`song` must be a path to a song or an audio signal array")
    S, cut, fig, ax, df, dt = _digital_to_spec(digital, fs, frac_cut=0.77, plot=True)

    if with_peaks:
        peaks = _local_peaks(S, cut, p_nn=20)
        t_loc, f_loc = zip(*peaks)
        ts = dt * _np.array(list(t_loc))
        fs = df * _np.array(list(f_loc))
        ax.scatter(ts, fs, s=4)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Frequency (Hz)")
    return fig, ax


def plot_recording(time: float, with_peaks: bool = True) -> Tuple[Figure, Axes]:
    """ Plot a spectrogram and fingerprint features for a live recording

    Parameters
    ----------
    time : float
        The time, in seconds, for which the microphone will record the sample.

    with_peaks : bool
        If True, include peak-value scatter-points

    Returns
    -------
    matplotlib.pyplot.Figure, matplotlib.pyplot.Axes """
    frames, sample_rate = record_audio(time)
    digital_data = _np.hstack([_np.frombuffer(i, _np.int16) for i in frames])
    return plot_song(digital_data, with_peaks=with_peaks)
