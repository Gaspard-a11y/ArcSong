from calendar import c
from pathlib import Path

import fire
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules.utils import load_json
from modules.dataset import _get_MSD_raw_dataset
        

def main(out_dir="media", img=True, local=True, overwrite=False):
    """
    Load the msd and msd metadata, plot *
    and save exploratory data analysis. 
    """

    if not overwrite:
        return

    else:
        # Plot log histogram of soung counts per artist
        print("Plotting histogram of song counts per artists ...")

        available_track_ids = load_json(Path("msd_data/waveforms_track_ids.json"))
        trackId_to_artistName = load_json(Path("msd_data/track_id_to_artist_name.json"))

        artistName_to_songCount = {}
        for id in tqdm(available_track_ids):
            artist_name = trackId_to_artistName[id]
            try:
                artistName_to_songCount[artist_name]+=1
            except KeyError:
                artistName_to_songCount[artist_name]=1

        # This is the whole MSD and not the available audios
        # artistName_to_songCount = load_json(Path("msd_data/artist_name_to_song_count.json"))

        counts = list(artistName_to_songCount.values())
        total_song_count = np.sum(counts)
        total_artist_count = len(counts)

        plt.figure(figsize=(14,7))
        plt.hist(counts, bins = 50, color='midnightblue')
        plt.yscale('log')
        plt.title(f"Number of tracks per artist ({total_song_count} tracks, {total_artist_count} artists)")
        plt.xlabel("Number of tracks")
        plt.ylabel("Number of artists")
        plt.minorticks_on()
        plt.grid(which='major', color='r', linestyle='-', alpha=0.3)
        plt.grid(which='minor', color='r', linestyle='--', alpha=0.2)

        if img:
            out_path = Path(out_dir) / "song_count_histogram.png"
        else:
            out_path = Path(out_dir) / "song_count_histogram.pdf"
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

        
        # Plot song length distribution
        print("Plotting histogram of song lengths ...")
        
        msd_dataset = _get_MSD_raw_dataset(local=local)
        track_lengths = []

        for data_example in tqdm(msd_dataset):
            track_lengths.append(data_example['audio'].numpy().shape[0]/16000)
        plt.figure(figsize=(14,7))
        plt.hist(track_lengths, bins = 50, color='midnightblue')
        plt.yscale('log')
        plt.title("Tracks' lengths in the dataset")
        plt.xlabel("Track length (s)")
        plt.ylabel("Number of songs")
        plt.minorticks_on()
        plt.grid(which='major', color='b', linestyle='-', alpha=0.2)
        plt.grid(which='minor', color='r', linestyle='--', alpha=0.2)
        
        if img:
            out_path = Path(out_dir) / "songs_length_histogram.png"
        else:
            out_path = Path(out_dir) / "songs_length_histogram.pdf"
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

    return


if __name__=="__main__":
    fire.Fire(main)