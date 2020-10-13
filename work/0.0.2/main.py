import librosa
from matplotlib import pyplot as plt
from os import listdir
from Network.net import Network
from Network.normalize_list import resize_list
import torch
import librosa.display
import numpy as np
from collections import Counter

from random import shuffle as random_list

result_dict = {
    0: "hmm",
    1: "firk",
    2: "space",
}
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
def filter_sound(sound):
    return split_sound(get_center_sound(sound))

def make_result(result: list):
    return result_dict[np.argmax(result.detach().numpy())]

def get_result(array_result: list):
    arr = Counter(array_result)

    arr_i = [j for j in arr.items()]
    for i in range(len(arr)):
        if arr_i[i][1] < arr_i[i-1][1]:
            arr_i[i-1],arr_i[i] = arr_i[i],arr_i[i-1]
    return arr_i[len(arr_i)-1][0]

def tratment_sound(sound: bytes):
    new_sound = librosa.feature.chroma_stft(np.array(sound))
    new_sound = new_sound.reshape(len(new_sound) * len(new_sound[0]))
    return new_sound


net_features_sound = Network([1,1],activate="Tanh",optimizer_lr=0.001)
net_features_sound.load("net_features_sound")
net_get_sound_by_features = Network([1,1],activate="Tanh",optimizer_lr=0.001)
net_get_sound_by_features.load("net_get_sound_by_features")
resizer = resize_list()
def think(path: str):
    try:
       sounds = split_sound(librosa.load(path)[0], count_split=3000)
    except Exception:
        return "dont such this catalog :| "
    outputs = []
    for sound in sounds:
        output = net_features_sound.forward(inputs=torch.FloatTensor(tratment_sound(sound=sound)))
        outputs.append(output)

    output = net_get_sound_by_features.forward(inputs=torch.FloatTensor(resizer.resize(lst=[j.detach().numpy().tolist()[0] for j in outputs],
                                                                                    resize_to=16)))
    return "sourcelib think this is: "+str(make_result(output))






while True:
    path_to_sound = input("Введите имя звука из этого каталога: ")
    print("Нейросеть думает что это: ",think(path=path_to_sound))






