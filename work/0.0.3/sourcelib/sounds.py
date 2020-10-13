"""In this file functions for sounds"""
print("import library for sounds")
from librosa.feature import chroma_stft
from numpy import array

def split_sound(sound,count_split: int = 3000):
    splited = []
    for splited_index in range(len(sound)//count_split):
        splited.append(sound[:count_split])
        sound = sound[count_split:]
    return splited[:len(splited)-1]

def get_center_sound(splited_sound: list, count_split: int = 10000):
    if len(splited_sound) < count_split+count_split+1:#if len sound is too small
        return False
    list_in_center = []
    center_index = len(splited_sound)//2
    for index_in_sound in range(center_index-count_split,center_index+count_split):
        list_in_center.append(splited_sound[index_in_sound])
    return list_in_center

def get_center_split_sound(sound):
    return split_sound(get_center_sound(sound))

def tratment_sound(sound: bytes):
    new_sound = chroma_stft(array(sound))
    new_sound = new_sound.reshape(len(new_sound) * len(new_sound[0]))
    return new_sound

