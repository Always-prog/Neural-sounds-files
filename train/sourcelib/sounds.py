"""In this file functions for sounds"""
import librosa
import numpy as np
print("import library for sounds")
from scipy import stats
def split_sound(sound,count_split: int = 3000):
    splited = []
    for splited_index in range(len(sound)//count_split):
        splited.append(sound[:count_split])
        sound = sound[count_split:]
    if len(splited[:len(splited)-1]) == 0:
        return False
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
    ff_list = []

    chroma_stft_feature = librosa.feature.chroma_stft(sound)
    mfcc_feature = librosa.feature.mfcc(sound)
    chroma_cqt_feature = librosa.feature.chroma_cqt(sound)
    for stft in chroma_stft_feature:
        for stft2 in stft:
            ff_list.append(stft2)
    for mfcc in mfcc_feature:
        for mfcc2 in mfcc:
            ff_list.append(mfcc2)
    for cqt in chroma_cqt_feature:
        for cqt2 in cqt:
            ff_list.append(cqt2)
    return ff_list

