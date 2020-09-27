import librosa.display
from matplotlib import pyplot as plt
import torch
from Network.net import Network
wav_path = "C:/Program Files/JetBrains/projects/NeuralSounds/hmm_2.wav"
sr = librosa.load(wav_path)

"""
#######НЕЙРОСЕТЬ#######
Одна нейросеть берет 30 кусочков звуков, разделенных функцией split_sound, с параметром 500.
Обучается отвечать что это именно те звуки которые, обозначены

###которая принимает каждый раз новых 30 кусочков###


D-> O-> O\
D-> O-> O-> o-> 0 (Hmm)
D-> O-> O-> o-> 1 (Firk)?
D-> O-> O/



Вторая которая принимает ответы первой, 30 ответов, и делает вывод, какой был звук

D-> O-> O\
D-> O-> O-> o-> 0 (Hmm)
D-> O-> O-> o-> 1 (Firk)
D-> O-> O/


"""
def split_sound(sound,milisecounds: int = 500):
    splited = []
    for splited_index in range(len(sound)//milisecounds):
        splited.append(sound[:milisecounds])
        sound = sound[milisecounds:]
    return splited
def get_center_sound(splited_sound: list, count_split: int = 15):
    if len(splited_sound) < count_split+count_split+1:#if len sound is too small
        return False
    list_in_center = []
    center_index = len(splited_sound)//2
    for index_in_sound in range(center_index-count_split,center_index+count_split):
        list_in_center.append(splited_sound[index_in_sound])
    return list_in_center


# sound_20 = split_sound(sound=sr[0])
# plt.plot(get_center_sound(splited_sound=sound_20))
#
# plt.show()
print(torch.FloatTensor())