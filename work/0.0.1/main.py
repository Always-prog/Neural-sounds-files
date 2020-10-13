import librosa
from matplotlib import pyplot as plt
from os import listdir
from Network.net import Network
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







SoundNet = Network([1,1],activate="PReLU",optimizer_lr=0.001)
SoundNet.load("config")
while True:
    path_to_sound = input("Введите имя звука из этого каталога: ")
    try:
        sounds = split_sound(librosa.load(path_to_sound)[0], count_split=3000)
    except Exception as e:
        print(e)
        print("В вашем каталоге нет такого файла :/")
        continue
    outputs = []
    for sound in sounds:
        sound_tratment = tratment_sound(sound=sound)
        output = SoundNet.forward(inputs=torch.FloatTensor(sound_tratment))
        outputs.append(make_result(output))
    print("Нейросеть думает что это: ",get_result(outputs))






