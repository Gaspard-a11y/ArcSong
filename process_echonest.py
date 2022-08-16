from os.path import exists
from pathlib import Path

from tqdm import tqdm
import fire

from modules.utils import save_json


def process_echo_nest(echo_nest_metadata = 'echonest_data/Echo_Nest_metadata.txt'):
    echo_nest_path=Path(echo_nest_metadata)
    ### Load echonest_data/Echo_Nest_metadata.txt
    f = open(echo_nest_path,'r', encoding="utf8")
    lines = f.readlines()

    def process_line(line):
        track_id, _, artist_name, _ = line.split('<SEP>')
        return track_id, artist_name


    ### Build dict[track_id]=artist_name, dict[artist_name]=number_of_songs

    trackID_to_artistName = {}
    artistName_to_songCount = {}

    for line in tqdm(lines):
        track_id, artist_name = process_line(line)
        
        # First dict
        try:
            _ = trackID_to_artistName[track_id]
        except KeyError:
            trackID_to_artistName[track_id] = artist_name

        # Second dict
        try:
            artistName_to_songCount[artist_name] += 1
        except KeyError:
            artistName_to_songCount[artist_name] = 1


    ### Build artist_list, dict[artist_name]=artist_number

    artist_list = list(artistName_to_songCount.keys())
    # Sort by descending number of tracks
    artist_list.sort(key=artistName_to_songCount.__getitem__, reverse=True)

    artistName_to_artistNumber = {}
    for i, name in enumerate(artist_list):
        artistName_to_artistNumber[name] = i

    sorted_artistName_to_songCount = {}
    for name in artist_list: # which is sorted
        sorted_artistName_to_songCount[name] = artistName_to_songCount[name]

    return trackID_to_artistName, sorted_artistName_to_songCount, artist_list, artistName_to_artistNumber


def main(overwrite = False, out_path = "echonest_data"):
    """
    Process the Echo_Nest_track_ids.txt file 
    from http://millionsongdataset.com/pages/getting-dataset/.
    Saves four files: 
    - dict[track_id] = artist_name,
    - dict[artist_name] = number_of_songs,
    - artist_list,
    - dict[artist_name] = artist_number.
    These files are for the ENTIRE dataset (not the one we have access to with the tfrecords).
    For our subset, see process_tfrecords_x_echonest.py.
    """

    out_path1 = Path(out_path) / "track_id_to_artist_name.json"
    out_path2 = Path(out_path) / "artist_name_to_song_count.json"
    out_path3 = Path(out_path) / "artist_list.json"
    out_path4 = Path(out_path) / "artist_name_to_artist_number.json"

    exists_bool = exists(out_path1) or exists(out_path2) or exists(out_path3) or exists(out_path4)
    if exists_bool and not overwrite:
        print("============ Found a json file, cancelling process ============")
        print("============= to overwrite, use: --overwrite True =============")
        return
    
    print("Processing Echo Nest metadata ...")
    echo_nest = process_echo_nest()
    trackID_to_artistName = echo_nest[0] 
    artistName_to_songCount = echo_nest[1] 
    artist_list = echo_nest[2] 
    artistName_to_artistNumber = echo_nest[3]

    print("Saving ...")
    # Save dict[track_id] = artist_name
    save_json(trackID_to_artistName, out_path1)
    # Save dict[artist_name] = number_of_songs
    save_json(artistName_to_songCount, out_path2)
    
    # Save artist_list (ordered by number of )
    save_json(artist_list, out_path3)
    # Save dict[artist_name] = artist_number
    save_json(artistName_to_artistNumber, out_path4)

    return

if __name__=='__main__':
    fire.Fire(main)