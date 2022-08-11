import os
import json

import fire
from tqdm import tqdm

from modules.dataset import get_MSD_train_dataset
from modules.utils import save_json

# WARNING will break eventually, 
# as get_MSD_train_dataset() will no longer return dicts  
def generate_trackid_json(local=True, 
                        overwrite = False, 
                        out_path = 'msd_data/waveforms_track_ids.json'):
    
    if os.path.exists(out_path) and not overwrite:
        print("/!\ /!\ /!\ Found a json file, cancelling process /!\ /!\ /!\ ")
        return
    
    else: 
        dataset = get_MSD_train_dataset(local=local)
        track_ids = {}

        for data_example in tqdm(dataset):
            track_id = data_example['tid'].numpy()[0].decode('UTF-8')
            try:
                track_ids[track_id] += 1
            except KeyError:
                track_ids[track_id] = 1

        list_of_track_ids = list(track_ids.keys())

        save_json(file=list_of_track_ids, json_path=out_path)
        
        return


if __name__=="__main__":
    fire.Fire(generate_trackid_json)
