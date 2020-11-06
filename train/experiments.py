from matplotlib import pyplot as plt
from train.sourcelib.net import Network
from train.sourcelib.normalize_list import lists
import torch
import librosa.display
import numpy as np
from collections import Counter
from config import net_config
from scipy import stats

def tratment_sound(sound: bytes,sr):
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


def split_sound(sound,count_split: int = 500):
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




def make_result(result: list):
    return net_config["netout"][np.argmax(result.detach().numpy())]

def get_result(array_result: list):
    arr = Counter(array_result)

    arr_i = [j for j in arr.items()]
    for i in range(len(arr)):
        if arr_i[i][1] < arr_i[i-1][1]:
            arr_i[i-1],arr_i[i] = arr_i[i],arr_i[i-1]
    return arr_i[len(arr_i)-1][0]


first_network = Network([72,5000,2000,1000,100,3],activate="Tanh",optimizer_lr=0.0013)#this network is for get features in sound
second_network = Network([16,400,200,100,50,3],activate="Tanh",optimizer_lr=0.001)#this network is for get sound name by feaures first network
resizer = lists()
for repeat in range(1):
    for sounds_dict in net_config["training"]["train_data"]:
        snd_bytes,sr = librosa.load(sounds_dict["path"])
        sounds = split_sound(snd_bytes,count_split=1000)
        if not(sounds):
            continue
        outputs = []
        for sound in sounds:
            #print(tratment_sound(sound=sound, sr=sr))
            plt.plot(tratment_sound(sound=sound,sr=sr))
            plt.xlabel(sounds_dict["name"])
            plt.show()



