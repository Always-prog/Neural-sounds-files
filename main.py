import librosa
from matplotlib import pyplot as plt
from os import listdir
from Network.net import Network
import torch
import statistics
import librosa.display


def split_sound(sound,count_split: int = 1000):
    splited = []
    for splited_index in range(len(sound)//count_split):
        splited.append(sound[:count_split])
        sound = sound[count_split:]
    return splited
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


firks = ["./sounds/firk/"+i for i in listdir("./sounds/firk")]
hmms = ["./sounds/hmm/"+i for i in listdir("./sounds/hmm")]






training_sounds = []
for path_to_sound in range(len(firks)+len(hmms)):
    try:
        training_sounds.append({
            "name":"firk",
            "path": firks[path_to_sound],
            "must_output":[0.1]
        })
        training_sounds.append({
            "name":"hmm",
            "path": hmms[path_to_sound],
            "must_output": [0.9]
        })
    except IndexError:
        continue

"""This is diagrams"""
# for snd in training_sounds:
#     print()
#     sound = librosa.load(snd["path"])[0]
#     X = librosa.stft(sound)
#     Xdb = librosa.amplitude_to_db(abs(X))
#     plt.figure(figsize=(14, 5))
#     librosa.display.specshow(Xdb, x_axis='time', y_axis='hz')
#     plt.xlabel(snd["name"], fontsize=12, color='black')
#     plt.colorbar()
#     plt.show()

errors_Net_create_30 = []
Net_create_30 = Network([1000,2000,1000,500,200,100,1],activate="Sigmoid",optimizer_lr=0.01 ,optimizer_momentum=0.9)


for i in range(1):
    for sounds_dict in training_sounds:
        sound = librosa.load(sounds_dict["path"])[0]
        sounds = filter_sound(sound)

        if sounds != False:#if this not False
            outputs = []
            for splited_sound in sounds:
                print(statistics.mean(splited_sound))
                output = Net_create_30.training_net(inputs=torch.FloatTensor(splited_sound), must_outputs=torch.FloatTensor(sounds_dict["must_output"]))
                errors_Net_create_30.append(Net_create_30.loss)



plt.plot(errors_Net_create_30)
plt.show()
print("--------------------------")
for i in range(5):
    sound = librosa.load(hmms[i])[0]
    sounds = filter_sound(sound)
    if sounds != False:#if this not False
        all_out = []
        for splited_sound in sounds:
            plt.plot(splited_sound)
            output = Net_create_30.forward(inputs=torch.FloatTensor(splited_sound))
            all_out.append(output.detach().numpy()[0])
        print(statistics.mean(all_out))
        plt.show()
print("----------firk----------")
for i in range(5):
    sound = librosa.load(firks[i])[0]
    sounds = filter_sound(sound)

    if sounds != False:#if this not False
        all_out = []
        for splited_sound in sounds:
            plt.plot(splited_sound)
            output = Net_create_30.forward(inputs=torch.FloatTensor(splited_sound))
            all_out.append(output.detach().numpy()[0])
        print(statistics.mean(all_out))
        plt.show()
"""
#######НЕЙРОСЕТЬ#######
Одна нейросеть берет 30 кусочков звуков, разделенных функцией split_sound, с параметром 500.
Обучается отвечать что это именно те звуки которые, обозначены

###которая принимает каждый раз новых 30 кусочков###


D-> O-> O\
D-> O-> O-> o-> 1 (Hmm)
D-> O-> O-> o-> 1 (Firk)
D-> O-> O/



Вторая которая принимает ответы первой, 30 ответов, и делает вывод, какой был звук

D-> O-> O\
D-> O-> O-> o-> 1 (Hmm)
D-> O-> O-> o-> 1 (Firk)
D-> O-> O/
"""