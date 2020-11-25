# -*- coding: utf-8 -*-
# https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
# Проход через ноль
# Центроид
from pydub import AudioSegment
name = 'Bloodborne: The Old Hunters OST_Lady Maria of the Astral Clocktower'
sound = AudioSegment.from_mp3(f"../parser/data/{name}.mp3")
sound.export(f"{name}.wav", format="wav")

import librosa
audio_path = f"{name}.wav"
x, sr = librosa.load(audio_path)

import IPython.display as ipd
ipd.Audio(audio_path)

# +
# %matplotlib inline
import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
# -

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()

# Частота пересечения нуля (zero crossing rate) – это частота изменения знака сигнала, т. е. частота, с которой сигнал меняется с положительного на отрицательный и обратно. Эта функция широко используется как для распознавания речи, так и для извлечения музыкальной информации. Для металла и рока этот параметр обычно выше, чем для других жанров, из-за большого количества ударных.

# Спектральный центроид указывает, где расположен "центр масс" звука, и рассчитывается как средневзвешенное значение всех частот.
#
# В блюзовых композициях частоты равномерно распределены, и центроид лежит где-то в середине спектра. В металле наблюдается выраженное смещение частот к концу композиции, поэтому и спектроид лежит ближе к концу спектра.
#
# Вычислим спектральный центроид для каждого фрейма с помощью 

# +
import sklearn

spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
# Вычисление времени для визуализации
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Нормализация спектрального центроида
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# Построение графика
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
# -

# Мел-частотные кепстральные коэффициенты
#
# Мел-частотные кепстральные коэффициенты (MFCC) сигнала – небольшой набор характеристик (обычно около 10-20) которые сжато описывают общую форму спектральной огибающей. Этот параметр моделирует характеристики человеческого голоса.

mfccs = librosa.feature.mfcc(x, sr=sr)
# Отображение
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

mfccs = sklearn.preprocessing.scale(mfccs.astype(float), axis=1)
librosa.display.specshow(mfccs, sr=sr, x_axis='time')

# Частота цветности
#
# Цветность (chroma features) – это интересное и мощное представление для музыкального звука, при котором весь спектр проецируется на 12 контейнеров, представляющих 12 различных полутонов музыкальной октавы.

hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')








