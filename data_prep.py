import pandas as pd
import re


song_lyrics = pd.read_csv('data/rap_data.csv')

def extract_verses(row):
    song_lyrics = getattr(row, 'lyrics')
    verse_dict = {'verse_name':re.findall('\[.*\]', song_lyrics),
                  'verse_lyrics':re.split('\[.*\]', song_lyrics)[1:]}
    verse_df = pd.DataFrame(verse_dict)
    verse_df['song_artist'] = getattr(row, 'rapper')
    return verse_df

verse_data = [extract_verses(song) for song in song_lyrics.itertuples()]
verse_data = pd.concat(verse_data)

verse_data = verse_data.drop_duplicates()

def get_verse_artist(row):
    if ':' in row['verse_name']:
        return re.sub('^.+: |]','', row['verse_name'])
    return row['song_artist']

verse_data['verse_artist'] = verse_data.apply(lambda row: get_verse_artist(row), axis=1)
