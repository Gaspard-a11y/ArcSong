import os
from pathlib import Path

from tqdm import tqdm
import fire

from modules.utils import save_json


def main(overwrite = False,
        out_path = "msd_data/track_id_to_artist_name.json"):
    """
    Process the Echo_Nest_track_ids.txt file 
    from http://millionsongdataset.com/pages/getting-dataset/
    into a dict[track_id] = artist_name,
    save as json to out_path. 
    """

    out_path = Path(out_path)
    if os.path.exists(out_path) and not overwrite:
        print("/!\ /!\ /!\ Found a json file, cancelling process /!\ /!\ /!\ ")
        return

    # Load msd_data\Echo_Nest_track_ids.txt
    f = open('msd_data\Echo_Nest_track_ids.txt','r', encoding="utf8")
    lines = f.readlines()

    def process_line(line):
        track_id, song_id, artist_name, song_title = line.split('<SEP>')
        return track_id, artist_name

    # Build trackID to artist name dict
    trackID_to_artistName = {}
    number_of_duplicates = 0

    for line in tqdm(lines):
        track_id, artist_name = process_line(line)
        
        try:
            _ = trackID_to_artistName[track_id]
            number_of_duplicates += 1
        except KeyError:
            trackID_to_artistName[track_id] = artist_name
    # TODO cleanup the artist names?

    save_json(file=trackID_to_artistName, json_path=out_path)

    return

if __name__=='__main__':
    fire.Fire(main)