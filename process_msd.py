import json

import fire
from tqdm import tqdm

from modules.dataset import get_MSD_train_dataset

def generate_trackid_list(local=True):
    dataset = get_MSD_train_dataset(local=True)

    track_ids = {}
    
    for data_example in tqdm(dataset):
        track_id = data_example['tid'].numpy()[0].decode('UTF-8')
        try:
            track_ids[track_id] += 1
        except KeyError:
            track_ids[track_id] = 1
        
    with open('msd/waveforms_track_ids.json', 'w') as fp:
        json.dump(track_ids, fp,  indent=4)


if __name__=="__main__":
    fire.Fire(generate_trackid_list)
