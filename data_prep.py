import pandas as pd
import re

def extract_verses(row):
    song_lyrics = getattr(row, 'lyrics')
    verse_dict = {'verse_name':re.findall('\[.*\]', song_lyrics),
                  'verse_lyrics':re.split('\[.*\]', song_lyrics)[1:]}
    verse_df = pd.DataFrame(verse_dict)
    verse_df['song_artist'] = getattr(row, 'rapper')
    verse_df['song_id'] = row[0]
    return verse_df

def get_verse_artist(row):
    if ':' in row['verse_name']:
        return re.sub('^.+: |]','', row['verse_name'])
    return row['song_artist']

def is_real_verse(verse_name):
    return ('Verse' in verse_name) | \
    ('Hook' in verse_name) | \
    ('Chorus' in verse_name)

def format_name(name):
    return re.sub('-', ' ', name).strip().lower()

def get_primary_artist(verse_artist):
    seperators = re.findall(',|\(|&|\+|with', verse_artist)
    if len(seperators) == 0:
        return format_name(verse_artist)
    return format_name(verse_artist.split(seperators[0])[0])

def get_main_artists(artists, min_count):
    artist_counts = artists.value_counts()
    return artist_counts[artist_counts >= min_count].index.tolist()

def replace_other_artists(artist, main_artists):
    if artist not in main_artists:
        return 'other'
    return artist

def prep_data(song_data):

    verse_data = [extract_verses(song) for song in song_lyrics.itertuples()]
    verse_data = pd.concat(verse_data)

    verse_data['verse_artist'] = verse_data.apply(lambda row: get_verse_artist(row), axis=1)

    verse_data = verse_data.drop_duplicates(['song_id', 'verse_name'])
    verse_data = verse_data[verse_data['verse_name'].map(is_real_verse)]

    verse_data['primary_artist'] = verse_data['verse_artist'].map(get_primary_artist)
    keep_artists = get_main_artists(verse_data['primary_artist'], 50)
    verse_data['primary_artist'] = verse_data['primary_artist'].map(
                                    lambda x: replace_other_artists(x, keep_artists))

    return verse_data

if __name__ == '__main__':
    song_lyrics = pd.read_csv('data/rap_data.csv', encoding='ISO-8859-1')
    verse_data = prep_data(song_lyrics)
    verse_data.to_csv('data/prepped_rap_data.csv',  index=False)
