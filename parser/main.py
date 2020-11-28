# -*- coding: utf-8 -*-
import keys

import vk_api
from vk_api import audio, tools
import requests
from tqdm.notebook import tqdm
import os
import time

REQUEST_STATUS_CODE = 200
base_path = '/home/lapltop/Рабочий стол/NSU_DP/parser/data/'
if not os.path.exists(base_path):
    os.makedirs(base_path)

vk_session = vk_api.VkApi(login=keys.USERNAME, password=keys.PASSWORD)
vk_session.auth()
vk = vk_session.get_api()
vk_audio = audio.VkAudio(vk_session)


def get_song(song_data):
    result = requests.get(song_data["url"])
    if result.status_code != REQUEST_STATUS_CODE:
        print('sleep = ', song_data["artist"], song_data["title"])
        time.sleep(3)
        return get_song(song_data)
    return result


def get_musicks(friend_name, ID, path):
    musicks_data = vk_audio.get(owner_id=ID)[:NUMBER_SONGS]
    for song_data in tqdm(musicks_data, total=len(musicks_data)):
        song = get_song(song_data)
        song_name = f'{song_data["artist"]}_{song_data["title"]}'[:50].replace('/', '')
        with open(path + song_name + '.mp3', 'wb') as file:
            file.write(song.content)


NUMBER_SONGS = 10
friends = vk.friends.get()
for friend in tqdm(friends['items']):
    time.sleep(1)
    user_info = vk.users.get(user_ids=str(friend))[0]
    first_name = user_info.get('first_name', None).lower()
    last_name = user_info.get('last_name', None).lower()
    
    friend_name = f'{first_name}_{last_name}'
    path = base_path + friend_name + '/'
    if os.path.exists(path):
        continue
    os.makedirs(path)
    
    try:
        get_musicks(friend_name, str(friend), path)
    except Exception as err:
        print(friend_name, err)





# +
# for i in os.listdir('data/'):
#     if len(os.listdir(f'data/{i}')) ==  0:
#         os.removedirs(f'data/{i}')
#         print(f'data/{i}')
# -




