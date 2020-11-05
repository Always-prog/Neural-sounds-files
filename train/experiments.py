from matplotlib import pyplot as plt
from train.sourcelib.net import Network
from train.sourcelib.normalize_list import lists
import torch
import librosa.display
import numpy as np
from collections import Counter
from config import net_config


def tratment_sound(sound: bytes):
    spec = np.abs(librosa.stft(sound))
    spec = librosa.amplitude_to_db(spec)
    return spec.reshape(len(spec)*len(spec[0]))


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
        sounds = split_sound(librosa.load(sounds_dict["path"])[0], count_split=800)
        if not(sounds):
            continue
        outputs = []
        for sound in sounds:
            plt.plot(tratment_sound(sound=sound))
            plt.xlabel(sounds_dict["name"])
            plt.show()



