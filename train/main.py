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





def make_result(result: list):
    return net_config["netout"][np.argmax(result.detach().numpy())]

def get_result(array_result: list):
    arr = Counter(array_result)

    arr_i = [j for j in arr.items()]
    for i in range(len(arr)):
        if arr_i[i][1] < arr_i[i-1][1]:
            arr_i[i-1],arr_i[i] = arr_i[i],arr_i[i-1]
    return arr_i[len(arr_i)-1][0]


first_network = Network([72,5000,2000,1000,100,3],activate="Tanh",optimizer_lr=0.0025)#this network is for get features in sound
second_network = Network([16,400,200,100,50,3],activate="Tanh",optimizer_lr=0.0025)#this network is for get sound name by feaures first network
resizer = resize_list()
for repeat in range(1):
    for sounds_dict in net_config["training"]["train_data"]:
        sounds = split_sound(librosa.load(sounds_dict["path"])[0], count_split=3000)
        outputs = []
        for sound in sounds:
            output = first_network.training_net(inputs=torch.FloatTensor(tratment_sound(sound=sound)),
                                           must_outputs=torch.FloatTensor(sounds_dict["must_output"]))
            outputs.append(output)

        print("Sound: ",sounds_dict["name"], "\n-----------------")
        for i in outputs:
            print(make_result(i))
plt.plot(first_network.errors)
plt.show()
first_network.save("2_nets_1_tanh")
second_network.save("2_nets_2_tanh")



for sounds_dict in net_config["test"]["test_data"]:
    sounds = split_sound(librosa.load(sounds_dict["path"])[0], count_split=3000)
    outputs = []
    for sound in sounds:
        sounds = split_sound(librosa.load(sounds_dict["path"])[0], count_split=3000)
        outputs = []
        for sound in sounds:
            output = first_network.forward(inputs=torch.FloatTensor(tratment_sound(sound=sound)))
            outputs.append(output)
    print("---")
    print("Sound: ",sounds_dict["name"])
    print("sourcelib think: ",get_result(outputs))





