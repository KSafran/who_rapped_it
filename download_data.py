import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

GENIUS_URL = "https://api.genius.com"
with open('data/rap_genius_api.txt', 'r') as token_file:
    TOKEN = token_file.read()
HEADERS = {'Authorization': 'Bearer {}'.format(TOKEN)}

def search_rap_genius(query):
    '''
    looks up an artists Rap Genius API path by name
    - artists: artist's name (str)
    returns: artist's API path
    '''
    search_url = GENIUS_URL + "/search"
    params = {'q': query}
    response = requests.get(search_url, params=params,
                            headers=HEADERS)
    return response.json()["response"]["hits"][0]['result']

def get_artist_id(artist):
    result = search_rap_genius(artist)
    return result['primary_artist']['id']

def get_song_path(song):
    result = search_rap_genius(song)
    return result['api_path']

def get_artist_paths(artist_id):
    '''
    returns API paths for 50 most popular songs for an artist
    - id: artist id (str or int)
    returns: list of song api paths
    '''
    page = requests.get('{}/artists/{}/songs/?sort=popularity&per_page=50'.format(GENIUS_URL,
                                                                                  artist_id),
                        headers=HEADERS)
    song_paths = [song['api_path'] for song in page.json()['response']['songs']]
    return song_paths

def get_song_lyrics(song_path):
    '''
    returns lyrics given a song api path
    - song_path: artist song api path
    returns: song lyrics (str)
    '''
    page = requests.get('http://genius.com{}'.format(song_path))
    html = BeautifulSoup(page.text, "html.parser")
    [scripts.extract() for scripts in html('script')]
    lyrics = html.find('div', class_='lyrics').get_text()
    re.sub('\[.*\]', '', lyrics) # rap genius has notes in brackets
    return lyrics

def get_artists_lyrics(artist):
    '''
    gets lyrics for an artists top 50 songs
    - artist: name of artist (str)
    returns: list of song lyrics
    '''
    print('Collecting data for {}'.format(artist))
    artist_id = get_artist_id(artist)
    songs_paths = get_artist_paths(artist_id)
    return pd.DataFrame([[artist, get_song_lyrics(song)] for song in songs_paths])

def download_rap_data(rappers, filename):
    '''
    downloads and saves each rappers' top 50 lyrics
    - rappers: list of rappers names (list of str)
    - filename: name of file to save (str)
    returns: True if saved
    '''
    rap_lyrics = [get_artists_lyrics(rap) for rap in rappers]
    rap_data = pd.concat(rap_lyrics)
    rap_data.columns = ['rapper', 'lyrics']
    return rap_data.to_csv('data/{}'.format(filename), index=False)

if __name__ == '__main__':
    rappers = ['Kanye West', 'Drake', '2 Chainz', 'Migos', 'Jay Z',
               'Lil Wayne', 'Eminem', 'Kendrick Lamar', 'Future', 'Nicki Minaj']
    download_rap_data(rappers, 'rap_data.csv')
