import librosa
import scipy
from scipy import stats
from matplotlib import pyplot as plt
from os import listdir
from Network.net import Network
import torch
import statistics
import librosa.display
import numpy as np

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
    if result[0] > result[1]:
        return "hmm"
    else:
        return "firk"
def get_base_features(wav_path):
    ff_list = []

    y, sr = librosa.load(wav_path, sr=None)

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
    chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
    mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zrate = librosa.feature.zero_crossing_rate(y_harmonic)

    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    for i in range(0, 12):
        ff_list.append(chroma_mean[i])
    for i in range(0, 12):
        ff_list.append(chroma_std[i])

    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    for i in range(0, 13):
        ff_list.append(mfccs_mean[i])
    for i in range(0, 13):
        ff_list.append(mfccs_std[i])

    cent_mean = np.mean(cent)
    cent_std = np.std(cent)
    cent_skew = scipy.stats.skew(cent, axis=1)[0]

    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    data = np.concatenate(([cent_mean, cent_std, cent_skew],
                           contrast_mean, contrast_std,
                           [rolloff_mean, rolloff_std, rolloff_std]), axis=0)
    ff_list += list(data)

    zrate_mean = np.mean(zrate)
    zrate_std = np.std(zrate)
    zrate_skew = scipy.stats.skew(zrate, axis=1)[0]

    ff_list += [zrate_mean, zrate_std, zrate_skew]

    ff_list.append(tempo)

    return ff_list

firks = ["./sounds/train_sounds/firk/" + i for i in listdir("sounds/train_sounds/firk")]
hmms = ["./sounds/train_sounds/hmm/" + i for i in listdir("sounds/train_sounds/hmm")]






training_sounds = []
for path_to_sound in range(len(firks)+len(hmms)):
    try:
        training_sounds.append({
            "name":"firk",
            "path": firks[path_to_sound],
            "must_output":[0.0, 1.0]
        })
        training_sounds.append({
            "name":"hmm",
            "path": hmms[path_to_sound],
            "must_output": [1.0, 0.0]
        })
    except IndexError:
        continue

"""This is diagrams"""
# for snd in training_sounds:
#     sound = librosa.load(snd["path"])[0]
#     X = librosa.stft(np.array(get_center_sound(sound)))
#     Xdb = librosa.amplitude_to_db(abs(X))
#     plt.figure()
#     librosa.display.specshow(Xdb, x_axis='time', y_axis='log')
#     plt.colorbar()
#
#     plt.xlabel(snd["name"], fontsize=12, color='black')
#     plt.show()



errors_Net_create_30 = []
Net_create_30 = Network([6150,1300,5000,3000,1000,2],activate="Sigmoid",optimizer_lr=0.01)
# Net_create_30.load(path="10.0.0")
# for snd_now in split_sound(librosa.load(firks[0])[0]):
#     stft_sound = librosa.stft(np.array(snd_now))
#     temperature = librosa.amplitude_to_db(abs(stft_sound))
#     temperature = temperature.reshape(len(temperature)*len(temperature[0]))
#     output = Net_create_30.forward(inputs=torch.FloatTensor(temperature))
#     errors_Net_create_30.append(Net_create_30.loss)
#     print(output)
for i in range(40):
    for sounds_dict in training_sounds:
        for snd_now in split_sound(librosa.load(sounds_dict["path"])[0]):
            stft_sound = librosa.stft(np.array(snd_now))
            temperature = librosa.amplitude_to_db(abs(stft_sound))
            temperature = temperature.reshape(len(temperature)*len(temperature[0]))
            output = Net_create_30.training_net(inputs=torch.FloatTensor(temperature), must_outputs=torch.FloatTensor(sounds_dict["must_output"]))
            errors_Net_create_30.append(Net_create_30.loss)
            print(output)
Net_create_30.save("2_10.0.0_40")
plt.plot(errors_Net_create_30)
plt.show()

# for i in range(1):
#     for sounds_dict in training_sounds:
#         print("--------------------------")
#         print(sounds_dict["name"])
#         output = Net_create_30.forward(inputs=torch.FloatTensor(get_base_features(sounds_dict["path"])))
#         print(output)
#         print("=============================")
#
# print("+++++++++++++++++++++++++++++++++++++++")
# print("firk:")
# output = Net_create_30.forward(inputs=torch.FloatTensor(get_base_features(
#     "./sounds/test_sounds/firk/3e6fb03cf985866dc9f9f7070e2de3962a9125870fb9230c58297e4b68869332.wav")))
# print(output)
# print("hmm:")
# output = Net_create_30.forward(inputs=torch.FloatTensor(get_base_features(
#     "./sounds/test_sounds/hmm/2b30b03245cf21cfc202aeee47be752ad51b350761b7381ddb3c1def86adc5cf.wav")))
# print(output)