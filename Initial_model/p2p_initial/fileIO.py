"""
This file deals with all the functions related to file I/O 
"""

import os
import json
import json, codecs
import numpy as np 

# cwd = os.getcwd()
# path_to_file = cwd+"p1.json"
# new_file_path = cwd+"p2.json"

def make_json(data, file_path):
    jsonData = json.dumps(data)
    with open("test.json","w") as file:
        file.write(jsonData)


def convert_to_bytes(file_path):
    with open(file_path, "r") as file:
        read_data = file.read()
    return read_data.encode("utf-8")



def create_file(data):
    data = data.decode("utf-8")
    print("Writing to file")
    with open("test.hdf5", 'w') as file:
        file.write(data)
    return True

def np_to_json(data):
    # if not isinstance(data, np.ndarray):
    #data = data.tolist()
    json.dump(data, codecs.open(path_to_file, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
    return data
def json_to_np():
    data_text = codecs.open(new_file_path, 'r', encoding='utf-8').read()
    data = json.loads(data_text)
    data = np.array(data)
    return data



def main():
    data = convert_to_bytes()
    create_file(data)


if __name__ == "__main__":
    main()