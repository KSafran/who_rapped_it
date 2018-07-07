import unittest
import pandas as pd
from . import download_data
from . import data_prep
from . import model

class UnitTests(unittest.TestCase):

    # Data Download
    def test_artist_id(self):
        kanye_id = download_data.get_artist_id('Kanye West')
        self.assertEqual(kanye_id, 72)

    def test_artist_paths(self):
        kanye_path = download_data.get_artist_paths(72)
        self.assertEqual(len(kanye_path), 50)

    def test_song_lyrics(self):
        kanye_song = download_data.get_song_lyrics('/songs/70324')
        self.assertTrue(isinstance(kanye_song, str))

    # Data Prep
    def test_extract_verse(self):
        song_data = pd.DataFrame({'rapper':['Drake', 'Kanye'],
                                 'lyrics':['/n/n[Verse 1: Drake]blah blah',
                                           '/n/n[Hook: Kanye]blah blah']})
        extract = [data_prep.extract_verses(song) for song in song_data.itertuples()]
        self.assertTrue(isinstance(extract[0], pd.core.frame.DataFrame))
        self.assertEqual(extract[0].iloc[0,0], 'blah blah')
        self.assertEqual(extract[0].iloc[0,2], 'Drake')

    def test_extract_verse(self):
        song_data = pd.DataFrame({'song_artist':['Drake', 'Kanye'],
                                 'verse_name':['[Intro]', '[Verse 1: Drake]']})
        extract = song_data.apply(lambda row: data_prep.get_verse_artist(row), axis=1)
        self.assertEqual(extract.tolist(), ['Drake'] * 2)

    def test_real_verse(self):
        verses = ['fake', 'Produced by Joe',  'Chorus', 'Verse', 'Hook']
        real = [data_prep.is_real_verse(verse) for verse in verses]
        self.assertEqual(real, [False , False, True, True, True])

    def test_real_verse(self):
        verses = ['fake', 'Produced by Joe',  'Chorus', 'Verse', 'Hook']
        real = [data_prep.is_real_verse(verse) for verse in verses]
        self.assertEqual(real, [False , False, True, True, True])

    def test_name_format(self):
        self.assertEqual(data_prep.format_name(' Icky-Izalea'), 'icky izalea')

    def test_primary_artist(self):
        test_names = ['Drake, Nobody', 'Drake & Nobody', 'Drake (Nobody)',
                      'Drake + Nobody', 'Drake with Nobody']
        primary_artist = [data_prep.get_primary_artist(name) for name in test_names]
        self.assertEqual(primary_artist, ['drake'] * 5)

    def test_main_artist(self):
        example = pd.Series(['Drake'] * 19 + ['Kanye'] * 20)
        self.assertEqual(data_prep.get_main_artists(example, 20), ['Kanye'])

    def test_artist_replacement(self):
        self.assertEqual(data_prep.replace_other_artists('Joe', ['Kanye', 'Drake']), 'other')
        self.assertEqual(data_prep.replace_other_artists('Drake', ['Kanye', 'Drake']), 'Drake')

    # Modeling
    def test_test_filter(self):
        test_data = pd.DataFrame({'song_id':4 * [0] + 4 * [1],
                                  'lyrics':8 * ['blah']})
        sample = model.create_test_filter(test_data, 0.5)
        self.assertEqual(sum(sample), 4)
        self.assertEqual(len(test_data['song_id'][sample].unique()), 1)

    def test_lyrics_format(self):
        lyrics = model.format_lyrics('Unformatted Lyrics. Woo!')
        self.assertEqual(lyrics, ['unformatted', 'lyrics', 'woo'])

    def test_lyric_to_int(self):
        tokenized_lyrics = [['tokenized', 'lyrics'], ['tokenized', 'words']]
        token_ints, word_dict = model.lyric_to_int(tokenized_lyrics)
        self.assertTrue(isinstance(token_ints[0][0], int))
        self.assertTrue(isinstance(word_dict, dict))
        self.assertEqual(token_ints[0][0], word_dict['tokenized'])
        self.assertEqual(token_ints[0][0], token_ints[1][0])

    def test_prep_data(self):
        sample_data = pd.DataFrame({'song_id':[0, 0, 1, 1],
                                    'verse_lyrics':['Old McDonald', 'had a Farm',
                                                    'eeyi eeyi oh', 'on that farm'],
                                    'primary_artist':['sue', 'sue', 'jim', 'bob']})
        train, test, words, artists = model.prep_data(sample_data, test_pct=0.5)
        self.assertTrue(isinstance(words, dict))
        self.assertTrue(isinstance(artists, dict))
        self.assertTrue(isinstance(words['farm'], int))
        self.assertTrue(isinstance(artists['sue'], int))
        self.assertEqual(len(train[0]), 2)
        self.assertEqual(len(test[0]), 2)
