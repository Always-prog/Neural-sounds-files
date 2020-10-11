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


firks = ["./sounds/train_sounds/firk/" + i for i in listdir("sounds/train_sounds/firk")]
firks_test = ["./sounds/test_sounds/firk/" + i for i in listdir("sounds/test_sounds/firk")]
hmms = ["./sounds/train_sounds/hmm/" + i for i in listdir("sounds/train_sounds/hmm")]
hmms_test = ["./sounds/test_sounds/hmm/" + i for i in listdir("sounds/test_sounds/hmm")]
spaces = ["./sounds/train_sounds/space/" + i for i in listdir("sounds/train_sounds/space")]
space_test = ["./sounds/test_sounds/space/" + i for i in listdir("sounds/test_sounds/space")]
others = ["./sounds/train_sounds/other/" + i for i in listdir("sounds/train_sounds/other")]
others_test = ["./sounds/test_sounds/other/" + i for i in listdir("sounds/test_sounds/other")]


training_sounds = []
for path_to_sound in range(len(firks)):
    try:
        training_sounds.append({
            "name":"firk",
            "path": firks[path_to_sound],
            "must_output":[0.0, 1.0, 0.0]
        })
    except IndexError:
        continue

for path_to_sound in range(len(hmms)):
    try:
        training_sounds.append({
            "name":"hmm",
            "path": hmms[path_to_sound],
            "must_output": [1.0, 0.0, 0.0]
        })
    except IndexError:
        continue

for path_to_sound in range(len(spaces)):
    try:
        training_sounds.append({
            "name": "space",
            "path": spaces[path_to_sound],
            "must_output": [0.0, 0.0, 1.0]
        })
    except IndexError:
        continue
test_sounds = []
for path_to_sound in range(len(firks_test)):
    try:
        test_sounds.append({
            "name":"firk",
            "path": firks_test[path_to_sound],
            "must_output":[0.0, 1.0, 0.0]
        })
    except IndexError:
        continue
for path_to_sound in range(len(firks_test)):
    try:
        test_sounds.append({
            "name":"hmm",
            "path": hmms_test[path_to_sound],
            "must_output":[1.0, 0.0, 0.0]
        })
    except IndexError:
        continue
for path_to_sound in range(len(space_test)):
    try:
        test_sounds.append({
            "name":"space",
            "path": space_test[path_to_sound],
            "must_output":[0.0, 0.0, 1.0]
        })
    except IndexError:
        continue
random_list(training_sounds)#rearrange lists
random_list(test_sounds)
random_list(training_sounds)
random_list(test_sounds)

SoundNet_errors = []
SoundNet = Network([72,5000,2000,500,100,3],activate="PReLU",optimizer_lr=0.001)
for i in range(30):
    for sounds_dict in training_sounds:
        sounds = split_sound(librosa.load(sounds_dict["path"])[0], count_split=3000)
        outputs = []
        for sound in sounds:
            sound_tratment = tratment_sound(sound=sound)
            output = SoundNet.training_net(inputs=torch.FloatTensor(sound_tratment),must_outputs=torch.FloatTensor(sounds_dict["must_output"]))
            print(output)
            outputs.append(output)
            SoundNet_errors.append(SoundNet.loss)
        print("--------",sounds_dict["name"],"--------")
        for j in outputs:
            print(make_result(j))
plt.plot(SoundNet_errors)
plt.show()
SoundNet.save("delete_other_new")



for sounds_dict in test_sounds:
    sounds = split_sound(librosa.load(sounds_dict["path"])[0], count_split=3000)
    outputs = []
    for sound in sounds:
        sound_tratment = tratment_sound(sound=sound)
        output = SoundNet.forward(inputs=torch.FloatTensor(sound_tratment))
        outputs.append(make_result(output))
    print("---")
    print("Sound: ",sounds_dict["name"])
    print("Network think: ",get_result(outputs))





