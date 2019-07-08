import pickle
from collections import abc, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import librosa
from songfp.functions import digital_to_spec, local_peaks, peaks_to_fingerprints


class Database:
    def __init__(self):
        self.default_path = Path(__file__).parent / "song_db.pkl"
        self.path: Path = self.default_path
        self.song_list_path = self.path / "_song_list.pkl"

        # Stores the mapping: (f1, f2, dt) -> [(song-ID, t1), ...]
        # where (f1, f2, dt) are the frequencies of two peaks and dt is
        # their separation in time. [(song-ID, t1), ...] is the list
        # of all song-IDs that contain this "fingerprint feature", along
        # with the time at which it occurs.
        self.pair_mapping: Dict[
            Tuple[int, int, int], List[Tuple[int, int]]
        ] = defaultdict(list)

        # A list of (song-name, artist)
        # Items should not be removed from this list! The song-ID in the
        # database corresponds to the index of that song's name in this list.
        # Instead, a song is removed by replacing its tuple with None
        self.song_list: List[Optional[Tuple[str, Optional[str]]], ...] = list()

        self._loaded = False

    def __len__(self):
        return len(self.song_list)

    def clear(self):
        """Clears the database"""
        self.pair_mapping.clear()
        self.song_list = list()
        self._loaded = False

    def switch_db(self, path=None):
        """ Switch the song database being used by specifying its load/save path. Calling this
        function with no argument will revert to the default database.

        Providing a name with no directories will assume songfp/database as the directory,
        otherwise the provided path is used. All databases will be saved as .pkl files.

        Parameters
        ----------
        path : PathLike"""

        _backup_db = self.pair_mapping
        _backup_path = self.path
        _loaded = self._loaded

        try:
            if path is not None:
                path = Path(path).resolve()
                parent = (
                    path.parent if str(path.parent) != "." else self.default_path.parent
                )
                self.path = parent / (path.stem + ".pkl")
                assert self.path.parent.exists(), f"{self.path.parent} doesn't exist"
            else:
                self.path = self.default_path
            self._loaded = False
            self.pair_mapping = defaultdict(list)
            self.load()

        except Exception as e:
            print("The following error occurred: {}".format(e))
            print(
                "\nReverting to your prior database state at: {}".format(
                    _backup_path.absolute()
                )
            )
            self.pair_mapping = _backup_db
            self.path = _backup_path
            self._loaded = _loaded
            raise e

    def load(self, force=False):
        """ Load the database from songfp/database/song_db.pkl if it isn't
        already loaded.

        Call this if you want to load the database up front. Otherwise,
        the other database methods will automatically load it.

        Parameters
        ----------
        force : bool, optional (default=False)
            If `True` the database will be loaded again, even if it
            is already in-memory."""
        if not force and self._loaded:
            return

        if not self.path.is_file():
            print(
                "No song database found. Creating empty database...\n"
                "\tSaving it will save to {}".format(self.path.absolute())
            )
            self.pair_mapping = defaultdict(list)
            self.song_list = list()
        else:
            with self.path.open(mode="rb") as f:
                data = pickle.load(f)

            assert isinstance(
                data, defaultdict
            ), f"the loaded database should be a defaultdict, got: {data}"

            self.pair_mapping = data

            with (self.path.parent / (self.path.stem + "_song_list.pkl")).open(
                mode="rb"
            ) as f:
                song_list = pickle.load(f)

            assert isinstance(
                song_list, list
            ), f"the loaded song_list should be a list, got: {song_list}"

            self.song_list = song_list
            print("song database loaded from: {}".format(self.path.absolute()))
        self._loaded = True

    def remove_song(self, name: str, artist: Optional[str] = None):
        try:
            # do not delete items from song list. song_id in database
            # is determined by song's position in song list. Removing
            # song will create offset in results.
            song_id = self.song_list.index((name, artist))
            self.song_list[song_id] = None

            for key, value in self.pair_mapping.items():
                self.pair_mapping[key] = [x for x in value if x[0] != song_id]

            print("{} removed from database. Be sure to save.".format((name, artist)))
        except ValueError:
            print("{} not in database".format((name, artist)))

    def save(self):
        if self.pair_mapping is None:
            print("No changes to face-database to save")
            return None

        with self.path.open(mode="wb") as f:
            pickle.dump(self.pair_mapping, f)

        with (self.path.parent / (self.path.stem + "_song_list.pkl")).open(
            mode="wb"
        ) as f:
            pickle.dump(self.song_list, f)

        print("Song database saved to: {}".format(self.path.absolute()))

    def add_songs(
        self,
        songs: Union[Path, Sequence[Path]],
        names: Optional[Union[Path, Sequence[Path]]] = None,
        artists: Optional[Union[Path, Sequence[Path]]] = None,
    ):
        """ Add songs to the fingerprinting database

        Parameters
        ----------
        songs : Union[str, Iterable[str]]
           File path(s) to .mp3, .wav, (and maybe other formats) file(s) to be added.

        names : Optional[Sequence[Union[str, None]]]
           Corresponding song names. If `None` is provided, the song name is inferred from
           the filename.

        artists : Optional[Sequence[Union[str, None]]]
           Corresponding song artists.

        Notes
        -----
        `add_songs('path/to/song/SongTitle.mp3')` will log this song
        in the database under the title 'SongTitle'. """

        if isinstance(songs, str):
            songs = [songs]

        if names is not None:
            assert isinstance(names, abc.Sequence) and len(names) == len(songs)
        else:
            names = [None] * len(songs)

        if artists is not None:
            assert isinstance(artists, abc.Sequence) and len(artists) == len(songs)
        else:
            artists = [None] * len(songs)

        old_num = len(self.song_list)

        for file_path, name, artist in zip(songs, names, artists):
            song_id = len(self.song_list)
            if name is None:
                name = Path(file_path).name

            if (name, artist) in self.song_list:
                print("{} already in song database. Skipping song.".format(name))
                continue
            print("adding {}..".format(name))

            digital, fs = librosa.load(file_path, sr=44100, mono=True)
            peaks = local_peaks(*digital_to_spec(digital, fs, frac_cut=0.77), p_nn=20)

            for f1_f2_dt, t1 in peaks_to_fingerprints(peaks, fan_value=15):
                self.pair_mapping[f1_f2_dt].append((song_id, t1))

            self.song_list.append((name, artist))

        if len(self.song_list) - old_num:
            print(
                "{} songs added to the database. "
                "\n\nBe sure to run `songfp.database.save()".format(
                    len(self.song_list) - old_num
                )
            )

    def list_songs(self):
        return sorted(x for x in self.song_list if x is not None)


database = Database()
