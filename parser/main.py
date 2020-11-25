# -*- coding: utf-8 -*-
import keys

import vk_api
from vk_api import audio
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


def create_folder(friend_name):
    path = base_path + friend_name + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_musicks(friend_name, ID):
    path = create_folder(friend_name)
    musicks_data = vk_audio.get(owner_id=ID)
    for song_data in tqdm(musicks_data, total=len(musicks_data)):
        song = get_song(song_data)
        song_name = f'{song_data["artist"]}_{song_data["title"]}.mp3'
        with open(path + song_name, 'wb') as file:
            file.write(song.content)


get_musicks('angelica', '132403360')


