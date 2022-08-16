from pathlib import Path

import fire
from tqdm import tqdm

from modules.dataset import _get_MSD_raw_dataset   
from modules.utils import save_json


def main(local=True, 
        overwrite = False, 
        out_dir = 'data_tfrecord'):
    """
    Process the tfrecords MSD dataset to extract the track ids.
    Save list as json to out_path. 
    """
    
    if not overwrite: 
        print("Use --overwrite True to overwrite existing output file.")
        return
    
    else: 
        dataset = _get_MSD_raw_dataset(local=local)
        track_ids = {}

        for data_example in tqdm(dataset):
            track_id = data_example['tid'].numpy()[0].decode('UTF-8')
            
            try:
                track_ids[track_id] += 1
            except KeyError:
                track_ids[track_id] = 1

        # Save list of track IDs
        list_of_track_ids = list(track_ids.keys())
        out_path = Path(out_dir) / "waveforms_track_ids.json"
        save_json(list_of_track_ids, out_path)

        return


if __name__=="__main__":
    fire.Fire(main)
