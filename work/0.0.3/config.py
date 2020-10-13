"""In this file makes config for netowrk"""
print("make config...")
print("create config value")
net_config = {
    "training": {
        "sounds": {
            "firk": {
                "name": "firk",
                "paths": None,
                "must_output": [0.0, 1.0, 0.0]
            },
            "hmm": {
                "name": "hmm",
                "paths": None,
                "must_output": [1.0, 0.0, 0.0]
            },
            "space": {
                "name": "space",
                "paths": None,
                "must_output": [0.0, 0.0, 1.0]
            }
        },
        "train_data": []

    },
    "test": {
        "sounds": {
            "firk": {
                "name": "firk",
                "paths": None,
                "must_output": [0.0, 1.0, 0.0]
            },
            "hmm": {
                "name": "hmm",
                "paths": None,
                "must_output": [1.0, 0.0, 0.0]
            },
            "space": {
                "name": "space",
                "paths": None,
                "must_output": [0.0, 0.0, 1.0]
            }
        },
        "test_data": []
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

