from sourcelib.sounds import split_sound
from sourcelib.sounds import tratment_sound
from matplotlib import pyplot as plt
from train.sourcelib.net import Network
from train.sourcelib.normalize_list import lists
import torch
import librosa.display
import numpy as np
from collections import Counter
from config import net_config
from colorama import Fore, Style
from keyboard import is_pressed as key


def red_text(text):
    print(Fore.RED + str(text))
    print(Style.RESET_ALL)

def make_result(result: list):
    return net_config["netout"][np.argmax(result.detach().numpy())]

def get_array_from_str(array_str: list):
    arr = []
    for arr_str_elem in array_str:
        try:
            arr.append(net_config["convert_str_to_float"][arr_str_elem])
        except KeyError as e:
            red_text(e)
            return False
    return arr
def get_result(array_result: list):
    arr = Counter(array_result)
    arr_i = [j for j in arr.items()]
    for i in range(len(arr)):
        if arr_i[i][1] < arr_i[i-1][1]:
            arr_i[i-1],arr_i[i] = arr_i[i],arr_i[i-1]
    return arr_i[len(arr_i)-1][0]


first_network = Network([176,5000,2000,1000,100,3],activate="Tanh",optimizer_lr=0.001)#this network is for get features in sound
first_network.load("frst")
second_network = Network([32,800,200,100,50,3],activate="Tanh",optimizer_lr=0.01)#this network is for get sound name by feaures first network
resizer = lists()
for repeat in range(50):
    for sounds_dict in net_config["training"]["train_data"]:
        snd,sr = librosa.load(sounds_dict["path"])
        sounds = split_sound(snd, count_split=2000)
        if key("ctrl+s"):
            second_network.save("scnd")
            print(f"SAVE not of {repeat} epox")
        if not(sounds):
            continue
        outputs = []
        for sound in sounds:
            output = first_network.forward(
                inputs=torch.FloatTensor(tratment_sound(sound=sound)))
            outputs.append(output)

        output = second_network.training_net(inputs=torch.FloatTensor(resizer.resize(lst=get_array_from_str([make_result(i) for i in outputs]),resize_to=32)),
                                             must_outputs=torch.FloatTensor(sounds_dict["must_output"]))

        print("----|",sounds_dict["name"],"|----")
        print(make_result(output))
        print("- - - - - - - - - - - - - - - - - ")
second_network.save("scnd")
plt.plot(first_network.errors)
plt.show()



for sounds_dict in net_config["test"]["test_data"]:
    sounds = split_sound(librosa.load(sounds_dict["path"])[0], count_split=3000)
    outputs = []
    for sound in sounds:
        output = first_network.forward(inputs=torch.FloatTensor(tratment_sound(sound=sound)))
        outputs.append(output)
    print("Sound: ",sounds_dict["name"])
    print(get_result(outputs), "\n-----------------")





