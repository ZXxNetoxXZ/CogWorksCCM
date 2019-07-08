"""
This module contains all of the functionality for managing song databases, including
adding/removing songs to/from a database.

Every song is ascribed a song_id based on its position in database._song_list.
If a song is removed, that song's fingerprints are removed from the database
and its entry in database._song_list is replaced with `None`.
"""


from ._database import database


__all__ = ["load_song_db",
           "clear",
           "add_songs",
           "switch_db",
           "list_songs",
           "remove_song"]


def load_song_db(func=None):
    """ This function can be invoked directly to lazy-load the song-recognition database, or it can
    be used as a decorator: the database is lazy-loaded prior to invoking the decorated function.

    See face_rec.face_db._load for more information.

    Parameters
    ----------
    func : Optional[Callable]

    Returns
    -------
    Union[None, Callable]"""
    if func is None:
        database.load()
        return None

    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        database.load()
        return func(*args, **kwargs)
    return wrapper


def switch_db(path=None):
    """ Switch the song database being used by specifying its load/save path. Calling this
    function with no argument will revert to the default database.

    Providing a name with no directories will assume songfp/database as the directory,
    otherwise the provided path is used. All databases will be saved as .pkl files.

    Parameters
    ----------
    path : PathLike"""
    database.switch_db(path)


@load_song_db
def clear(x: bool):
    """ Clear the song database.

    You must subsequently run `songfp.save_song_database()` to save this change.

    Parameters
    ----------
    x : bool
        Pass True explicitly to confirm that you want to clear the database."""
    database.clear(x)


def save():
    """ Save the database."""
    database.save()


@load_song_db
def add_songs(songs, names=None, artists=None):
    """ Add songs to the fingerprinting database

        Parameters
        ----------
        songs : Union[str, Iterable[str]]
           File path(s) to .mp3, .wav, (and maybe other formats) file(s) to be added.

        names : Optional[Sequence[Union[str, None]]]
           Corresponding song names. If `None` is provided, the dong name is inferred from
           the filename.

        artists : Optional[Sequence[Union[str, None]]]
           Corresponding song artists.

        Notes
        -----
        `add_songs_to_database('path/to/song/SongTitle.mp3')` will log this song in the database
        under the title 'SongTitle'. """
    database.add_songs(songs, names=names, artists=artists)


@load_song_db
def list_songs():
    return database.list_songs()


@load_song_db
def remove_song(name, artist=None):
    database.remove_song(name, artist)


