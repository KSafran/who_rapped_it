import unittest
from . import download_data

class UnitTests(unittest.TestCase):
    def test_artist_id(self):
        kanye_id = download_data.get_artist_id('Kanye West')
        self.assertEqual(kanye_id, 72)

    def test_artist_paths(self):
        kanye_path = download_data.get_artist_paths(72)
        self.assertEqual(len(kanye_path), 50)

    def test_song_lyrics(self):
        kanye_song = download_data.get_song_lyrics('/songs/70324')
        self.assertTrue(isinstance(kanye_song, str))
