"""In this file makes config for netowrk"""
print("make config...")
from os import listdir
from random import shuffle as randomize_list
from sourcelib.normalize_list import lists;lists = lists()
print("create config value")
net_config = {
    "training": {
        "sounds": {
            "firk": {
                "name": "firk",
                "paths": [],
                "must_output": [0.0, 1.0, 0.0]
            },
            "hmm": {
                "name": "hmm",
                "paths": [],
                "must_output": [1.0, 0.0, 0.0]
            },
            "space": {
                "name": "space",
                "paths": [],
                "must_output": [0.0, 0.0, 1.0]
            }
        },
        "train_data": []

    },
    "convert_str_to_float": {
        "firk": 0.5,
        "hmm": 0.0,
        "space": 1.0
    },
    "netout": {}
}
"""Set need outputs in net"""
print("set network outputs")
for snd_now in net_config["training"]["sounds"]:
    description = net_config["training"]["sounds"][snd_now]
    for out_neural_index in range(len(description["must_output"])):
        if description["must_output"][out_neural_index] == 1.0:
            net_config["netout"].update({int(out_neural_index):description["name"]})

print("Successfully upload config")

