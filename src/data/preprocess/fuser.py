import pickle
from pathlib import Path
import numpy as np


def gen_npy(path_orig: Path, path_genreative: Path):
    # Load the numpy files
    file_orig_data = np.load(path_orig)

    file_gen_data = np.load(path_generative)

    # Assuming your original array is named `original_array`
    original_shape = (150000, 3, 64, 25, 1)
    new_shape = (150000, 3, 300, 25, 1)

    # Repeat along the second dimension to expand it to size 300
    repeated_array = np.repeat(file_gen_data, 300 // 64 + 1, axis=2)

    # If the original size isn't a multiple of 300, we may need to cut off the excess
    if original_shape[2] % 300 != 0:
        repeated_array = repeated_array[:, :, :300, :, :]

    # Repeat the last dimension
    repeated_array = np.repeat(repeated_array, 2, axis=-1)

    # Use np.repeat to duplicate the values along the last axis to create a new axis with size 2

    # Concatenate along the appropriate axis
    appended_data = np.concatenate((file_orig_data, repeated_array), axis=0)

    # Save the concatenated data to a new file
    np.save('../../../data/npy_files/processed/ntu60/xview60/appended_file.npy', appended_data)



def gen_pkl(path_orig: Path, path_genreative: Path):
    with open(path_orig, 'rb') as f:
        name, label, seq_len = pickle.load(f, encoding='latin1')
    file_orig_data = np.load(path_genreative)

    breakpoint()





if __name__ == '__main__':
    path_orig = Path(
        "/Users/felixtempel/PycharmProjects/AutoGCN/data/npy_files/processed/ntu60/xview60/xview60_train_data.npy")
    path_generative = Path(
        "/Users/felixtempel/PycharmProjects/Kinetic-GAN2/runs/kinetic-gan/exp6/actions/60_2500_trunc0.95_gen_data.npy")

    path_orig_pkl = Path(
        "/Users/felixtempel/PycharmProjects/AutoGCN/data/npy_files/processed/ntu60/xview60/xview60_train_label.pkl")

    # gen_npy(path_orig, path_generative)

    gen_pkl(path_orig_pkl, path_generative)
