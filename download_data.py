import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

GENIUS_URL = "http://api.genius.com"
with open('data/rap_genius_api.txt', 'r') as token_file:
    TOKEN = token_file.read()
HEADERS = {'Authorization': 'Bearer {}'.format(TOKEN)}

def get_artist_api(artist):
    '''
    looks up an artists Rap Genius API path by name
    - artists: artist's name (str)
    returns: artist's API path
    '''
    search_url = GENIUS_URL + "/search"
    params = {'q': artist}
    response = requests.get(search_url, params=params,
                            headers=HEADERS)
    return response.json()["response"]["hits"][0]['result']['api_path']

def get_artist_lyrics(api_path):
    page = requests.get("http://genius.com" + api_path)
    html = BeautifulSoup(page.text, "html.parser")
    #remove script tags that they put in the middle of the lyrics
    [h.extract() for h in html('script')]
    lyrics = html.find('div', class_='lyrics').get_text()
    re.sub('\[.*\]', '', lyrics) # rap genius has notes in brackets
    return lyrics

yeezy = get_artist_api('Kanye West')
lyrics = get_artist_lyrics(yeezy)
