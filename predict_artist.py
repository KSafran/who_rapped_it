import download_data
import data_prep
import re


if __name__ == '__main__':
    test = download_data.get_song_path('nonstop')
    lyrics = download_data.get_song_lyrics(test)
    verses = re.split('\[.*\]', lyrics)
    
