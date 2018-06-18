import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

GENIUS_URL = "http://api.genius.com"
with open('data/rap_genius_api.txt', 'r') as token_file:
    TOKEN = token_file.read()
HEADERS = {'Authorization': 'Bearer {}'.format(TOKEN)}

def get_artist_id(artist):
    '''
    looks up an artists Rap Genius API path by name
    - artists: artist's name (str)
    returns: artist's API path
    '''
    search_url = GENIUS_URL + "/search"
    params = {'q': artist}
    response = requests.get(search_url, params=params,
                            headers=HEADERS)
    return response.json()["response"]["hits"][0]['result']['primary_artist']['id']

def get_artist_paths(id):
    '''
    returns API paths for 50 most popular songs for an artist
    - id: artist id (str or int)
    returns: list of song api paths
    '''
    page = requests.get('{}/artists/{}/songs/?sort=popularity&per_page=50'.format(GENIUS_URL, yeezy), headers=HEADERS)
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


yeezy = get_artist_id('Kanye West')
paths = get_artist_paths(yeezy)
lyrics = [get_song_lyrics(path) for path in paths]
