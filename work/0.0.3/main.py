from sourcelib.sounds import split_sound
from sourcelib.sounds import tratment_sound
from matplotlib import pyplot as plt
from train.sourcelib.net import Network
from train.sourcelib.normalize_list import resize_list
import torch
import librosa.display
import numpy as np
from collections import Counter
from config import net_config
from sourcelib.normalize_list import resize_list




def make_result(result: list):
    return net_config["netout"][np.argmax(result.detach().numpy())]

def get_result(array_result: list):
    arr = Counter(array_result)

    arr_i = [j for j in arr.items()]
    for i in range(len(arr)):
        if arr_i[i][1] < arr_i[i-1][1]:
            arr_i[i-1],arr_i[i] = arr_i[i],arr_i[i-1]
    return arr_i[len(arr_i)-1][0]


first_network = Network([72,5000,2000,1000,100,3],activate="Tanh",optimizer_lr=0.002)#this network is for get features in sound
second_network = Network([16,400,200,100,50,3],activate="Tanh",optimizer_lr=0.001)#this network is for get sound name by feaures first network
first_network.load("first_network")
second_network.load("second_network")
resizer = resize_list()

def think(path: str):
    sounds = split_sound(librosa.load(path)[0], count_split=3000)
    outputs = []
    for sound in sounds:
        output = first_network.forward(inputs=torch.FloatTensor(tratment_sound(sound=sound)))
        outputs.append(output)
    output = second_network.forward(
        inputs=torch.FloatTensor(resizer.resize(lst=[i.detach().numpy()[0] for i in outputs], resize_to=16)))
    return str(make_result(output))






while True:
    path_to_sound = input("Введите имя звука из этого каталога: ")
    print("Нейросеть думает что это: ",think(path=path_to_sound))






