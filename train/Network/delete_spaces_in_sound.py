"""
This function delete space from sound. For example delete silence from voice audio from telegram.
To use this function you need to set this file in your directory where is your files (.wav files),
after run this file.
"""



import librosa
from os import listdir
from numpy import delete, flip
from scipy.io.wavfile import write as write_wav
sound_pathes = [i for i in listdir("/") if i[len(i) - 4:] == ".wav"]

for sound in sound_pathes:
    snd,sr = librosa.load(sound)
    index = 0
    for i in range(len(snd)):
        if index > len(snd):
            break
        if snd[index] > 0.01 or snd[index] < -0.01:
            break
        snd = delete(snd,index)
    snd = flip(snd)
    for i in range(len(snd)):
        if index > len(snd):
            break
        if snd[index] > 0.01 or snd[index] < -0.01:
            break
        snd = delete(snd,index)


    write_wav(sound+"_new.wav", sr, snd)