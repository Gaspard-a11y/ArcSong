from os.path import exists
from pathlib import Path

from tqdm import tqdm
import fire

from modules.utils import load_json, save_json
from modules.dataset import _get_MSD_raw_dataset


def main(overwrite = False, local=True, out_path = "data_tfrecord_x_echonest"):
    if not overwrite:
        print("Use --overwrite True to overwrite previous files.")
        return

    # Load trackID_to_artistName dict
    trackID_to_artistName = load_json(Path("data_echonest/track_id_to_artist_name.json"))

    # Load dataset
    dataset = _get_MSD_raw_dataset(local=local)

    ### Build artist_name -> (song_count, total_audio_length) dict
    ### Build track length -> count dict
    artist_metadata = {}
    track_lengths_dict = {}

    for item in tqdm(dataset):
        track_id = item['tid'].numpy()[0].decode('UTF-8')
        artist_name = trackID_to_artistName[track_id]
        track_length = int(item['audio'].numpy().shape[0]/16000)
        # sampling rate is 16kHz

        # First dict
        try: 
            # Artist has been seen before
            song_count, total_audio_length = artist_metadata[artist_name]
            song_count += 1
            total_audio_length += track_length
        except KeyError: 
            # Previously unseen artist
            song_count = 1
            total_audio_length = track_length
        
        artist_metadata[artist_name] = song_count, total_audio_length

        # Second dict
        try:
            track_lengths_dict[track_length] += 1
        except KeyError:
            track_lengths_dict[track_length] = 1


    ### Build artist list order by descending song count or total audio length
    artist_list_by_count = list(artist_metadata.keys())
    artist_list_by_count.sort(key=(lambda item: artist_metadata[item][0]), reverse=True)

    ### Build reverse artist name -> artist number dict
    artist_name_to_artist_number_by_count = {}
    for k, name in enumerate(artist_list_by_count):
        artist_name_to_artist_number_by_count[name] = k


    ### Build artist list order by descending song count or total audio length
    artist_list_by_length = list(artist_metadata.keys())
    artist_list_by_length.sort(key=(lambda item: artist_metadata[item][1]), reverse=True)

    ### Build reverse artist name -> artist number dict
    artist_name_to_artist_number_by_length = {}
    for k, name in enumerate(artist_list_by_length):
        artist_name_to_artist_number_by_length[name] = k


    ### Save files
    out_file = Path(out_path) / "artist_discography_metadata.json"
    save_json(artist_metadata, out_file)

    out_file = Path(out_path) / "artist_list_by_count.json"
    save_json(artist_list_by_count, out_file)

    out_file = Path(out_path) / "artist_name_to_artist_number_by_count.json"
    save_json(artist_name_to_artist_number_by_count, out_file)

    out_file = Path(out_path) / "artist_list_by_length.json"
    save_json(artist_list_by_length, out_file)

    out_file = Path(out_path) / "artist_name_to_artist_number_by_length.json"
    save_json(artist_name_to_artist_number_by_length, out_file)

    out_file = Path(out_path) / "track_lengths.json"
    save_json(track_lengths_dict, out_file)
        
    return


if __name__=='__main__':
    fire.Fire(main)