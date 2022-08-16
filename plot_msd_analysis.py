from pathlib import Path

import fire
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules.utils import load_json
from modules.dataset import _get_MSD_raw_dataset


def plot_and_save_count_histogram(out_dir="media", img=True, total_msd=False):
 
    if not total_msd:
        # Just the available audios
        artist_metadata = load_json(Path("data_tfrecord_x_echonest/artist_discography_metadata.json"))
        artistName_to_songCount = {name: artist_metadata[name][0] for name in artist_metadata}
        specification = " in the subset."

    else:
        # The whole MSD
        artistName_to_songCount = load_json(Path("data_echonest/artist_name_to_song_count.json"))
        specification = " in the whole dataset."
        
    counts = list(artistName_to_songCount.values())
    total_song_count = np.sum(counts)
    total_artist_count = len(counts)

    plt.figure(figsize=(14,7))
    plt.hist(counts, bins = 50, color='midnightblue')
    plt.yscale('log')
    plt.title(f"Number of tracks per artist ({total_song_count} tracks, {total_artist_count} artists)"+specification)
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
    return
   

def plot_and_save_length_histogram(out_dir="media", img=True, local=True):
    msd_dataset = _get_MSD_raw_dataset(local=local)

    # Build track_length -> song count dict
    track_lengths_dict = {}
    for data_example in tqdm(msd_dataset):
        song_length = int(data_example['audio'].numpy().shape[0]/16000)
        try:
            track_lengths_dict[song_length] += 1
        except KeyError:
            track_lengths_dict[song_length] = 1

    # Transform the dict into an array    
    list_of_array = [song_length * np.ones(track_lengths_dict[song_length]) for song_length in track_lengths_dict]
    track_lengths = np.concatenate(list_of_array)
    
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
        out_path = Path(out_dir) / "song_length_histogram.png"
    else:
        out_path = Path(out_dir) / "song_length_histogram.pdf"
    
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    return


def plot_and_save_discography_length_histogram(out_dir="media", img=True, local=True):
    artist_metadata = load_json(Path("data_tfrecord_x_echonest/artist_discography_metadata.json"))

    artistName_to_discographyLength = {name: artist_metadata[name][1] for name in artist_metadata}
    counts = list(artistName_to_discographyLength.values())
    total_discography_length = np.sum(counts)
    total_artist_count = len(counts)
    
    plt.figure(figsize=(14,7))
    plt.hist(counts, bins = 50, color='midnightblue')
    plt.yscale('log')
    plt.title(f"Discographies' lengths in the dataset ({total_artist_count} artists, {total_discography_length} s of audio)")
    plt.xlabel("Artists' discography length (s)")
    plt.ylabel("Number of artists")
    plt.minorticks_on()
    plt.grid(which='major', color='b', linestyle='-', alpha=0.2)
    plt.grid(which='minor', color='r', linestyle='--', alpha=0.2)
    
    if img:
        out_path = Path(out_dir) / "song_length_histogram.png"
    else:
        out_path = Path(out_dir) / "song_length_histogram.pdf"
    
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    return


def main(out_dir="media", img=True, local=True, counts=True, lengths=False, total_length=True, total_msd=False):
    """
    Load the msd and msd metadata, plot *
    and save exploratory data analysis. 
    """
    if (not counts) and (not lengths):
        print("Use --counts False and/or --lengths False to overwrite.")

    if counts:
        # Plot log histogram of soung counts per artist
        print("Plotting histogram of song counts per artists ...")
        plot_and_save_count_histogram(out_dir=out_dir, img=img, total_msd=total_msd)

    if lengths:
        # Plot song length distribution
        # Warnin very long (~1h) operation
        print("Plotting histogram of song lengths ...")
        plot_and_save_length_histogram(out_dir=out_dir, img=img, local=local)

    if total_length:
        # Plot discography length distribution
        print("Plotting histogram of discography lengths ...")
        plot_and_save_discography_length_histogram(out_dir=out_dir, img=img, local=local)

    return


if __name__=="__main__":
    fire.Fire(main)