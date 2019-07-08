"""`songfp` is a Python package that performs song-fingerprint matching.
**This is a re-implementation/simplification of the [dejavu](https://github.com/worldveil/dejavu) project.**
Authorship should effectively be attributed to Will Drevo (GitHub user worldveil), who created dejavu.

In effect, `songfp` provides a service similar to popular song-recognition programs like Shazaam.
It can "listen" to a song, and match it against a database of song fingerprints, which is populated by the user.

`songfp` was created as a prototype for the CogWorks 2017 summer program, in the
Beaver Works Summer Institute at MIT. It was developed by Ryan Soklaski, the lead instructor of CogWorks 2017. """

from __future__ import absolute_import
from distutils.core import setup
from setuptools import find_packages
import songfp

ver = songfp.__version__
setup(name='songfp',
      version=ver,
      description='A reimplementation of the dejavu algorithm for CogWorks 2017',
      author='Ryan Soklaski (@LLrsokl)',
      author_email="ryan.soklaski@ll.mit.edu",
      packages=find_packages(),
      license="MIT",
      install_requires=['microphone', 'librosa', 'numpy', 'matplotlib', 'scipy']
      )
