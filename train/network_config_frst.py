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
                "paths": ["./sounds/train_sounds/firk/" + i for i in listdir("sounds/train_sounds/firk")],
                "must_output": [0.0, 1.0, 0.0,0.0]
            },
            "hmm": {
                "name": "hmm",
                "paths": ["./sounds/train_sounds/hmm/" + i for i in listdir("sounds/train_sounds/hmm")],
                "must_output": [1.0, 0.0, 0.0,0.0]
            },
            "space": {
                "name": "space",
                "paths": ["./sounds/train_sounds/space/" + i for i in listdir("sounds/train_sounds/space")],
                "must_output": [0.0, 0.0, 1.0,0.0]
            },
            "other": {
                "name": "other",
                "paths": ["./sounds/train_sounds/other/" + i for i in listdir("sounds/train_sounds/other")],
                "must_output": [0.0, 0.0, 0.0,1.0]
            }
        },
        "train_data": []

    },

    "convert_str_to_float": {
        "firk": 0.3,
        "hmm": 0.0,
        "space": 0.6,
        "other":1.0
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
print("set network training data")
net_config["training"]["train_data"] = \
[
    [{"path": i,
     "name": net_config["training"]["sounds"]["firk"]["name"],
     "must_output": net_config["training"]["sounds"]["firk"]["must_output"]}
    for i in net_config["training"]["sounds"]["firk"]["paths"]]\
                           +\
    [{"path": i,
     "name": net_config["training"]["sounds"]["hmm"]["name"],
     "must_output": net_config["training"]["sounds"]["hmm"]["must_output"]}
    for i in net_config["training"]["sounds"]["hmm"]["paths"]]\
                          + \
    [
    {"path": i,
    "name": net_config["training"]["sounds"]["space"]["name"],
    "must_output": net_config["training"]["sounds"]["space"]["must_output"]}

     for i in net_config["training"]["sounds"]["space"]["paths"]]
    + \
    [{"path": i,
      "name": net_config["training"]["sounds"]["other"]["name"],
      "must_output": net_config["training"]["sounds"]["other"]["must_output"]}
     for i in net_config["training"]["sounds"]["other"]["paths"]]
    ][0]


print("set network test data")
"""Mix training data"""
net_config["training"]["train_data"] = lists.connect_lists([
    [i for i in net_config["training"]["train_data"] if i["name"] == "firk"],
    [i for i in net_config["training"]["train_data"] if i["name"] == "hmm"],
    [i for i in net_config["training"]["train_data"] if i["name"] == "space"],
    [i for i in net_config["training"]["train_data"] if i["name"] == "other"],
])

print("Successfully upload config")

