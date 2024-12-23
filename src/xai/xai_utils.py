import logging
import glob
import re

import numpy as np
from torch.utils.data import Sampler


# Constants
NTU_JOINT_NAMES = ["base_of_the_spine", "middle_of_the_spine", "neck", "head", "left_shoulder",
                   "left_elbow", "left_wrist", "left_hand", "right_shoulder", "right_elbow", "right_wrist",
                   "right_hand", "left_hip", "left_knee", "left_ankle", "left_foot", "right_hip",
                   "right_knee", "right_ankle", "right_foot", "spine", "tip_of_the_left_hand",
                   "left_thumb", "tip_of_the_right_hand", "right_thumb"]

CP_JOINT_NAMES_29 = ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder',
                     'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist',
                     'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
                     'left_ankle', 'right_little_finger', 'right_index_finger', 'left_little_finger',
                     'left_index_finger', 'right_heel', 'right_little_toe', 'right_big_toe', 'left_heel',
                     'left_little_toe', 'left_big_toe']


class GradCamHandler:
    def __init__(self):
        pass

    def load_gradcams_cp(self, paths: str):
        pass

    def load_gradcams_ntu(self, paths: str):
        pass


class ShapHandler:
    def __init__(self):
        self.combined_shap_values = None
        self.combined_shap_names = None
        self.combined_shap_labels = None

        self.class_0_indices = None
        self.class_1_indices = None

    def load_shap_values_cp(self, paths) -> tuple:
        """
        Load and combine CP29 SHAP values from the provided paths.
        :param paths:
        :return:
        """
        weight = 1 / len(paths)
        logging.info(f'Weighting factor for final SHAPS: {weight}')
        for path in paths:
            if path is None:
                continue

            file_shap_values = glob.glob(f'{path}/shap_values_class_all*.npy')
            file_shap_labels = glob.glob(f'{path}/shap_values_labels_all*.npy')
            file_shap_names = glob.glob(f'{path}/shap_values_names_all*.npy')

            if file_shap_values:
                shap_values = np.load(file_shap_values[0], mmap_mode='r+')
                logging.info(f"Loaded file: {file_shap_values[0]}")
            else:
                logging.info(f"No files found for shap_values_class_all*.npy in {path}.")
                continue

            if file_shap_labels:
                shap_labels = np.load(file_shap_labels[0], allow_pickle=True)
                logging.info(f"Loaded file: {file_shap_labels[0]}")
            else:
                logging.info(f"No files found for shap_values_labels_all*.npy in {path}.")
                continue

            if file_shap_names:
                shap_names = np.load(file_shap_names[0], allow_pickle=True)
                logging.info(f"Loaded file: {file_shap_names[0]}")
            else:
                logging.info(f"No files found for shap_values_names_all*.npy in {path}.")
                continue

            if self.combined_shap_values is None:
                self.combined_shap_values = shap_values * weight
                self.combined_shap_labels = shap_labels
                self.combined_shap_names = shap_names
            else:
                assert len(self.combined_shap_labels) == len(shap_labels), "Mismatch in SHAP labels."
                assert len(self.combined_shap_names) == len(shap_names), "Mismatch in SHAP names."
                np.add(self.combined_shap_values, shap_values * weight, out=self.combined_shap_values)

            del shap_values

        if self.combined_shap_values is None:
            logging.error("No valid SHAP files found.")
            raise FileNotFoundError("No SHAP values files were found.")

        self.class_0_indices = np.where(self.combined_shap_labels == 0)[0]
        self.class_1_indices = np.where(self.combined_shap_labels == 1)[0]

        return self.combined_shap_values, self.combined_shap_names, self.combined_shap_labels

    def load_shap_values_ntu(self, paths, target_class) -> np.array:
        """
        Load and combine NTU SHAP values from the provided paths.
        :param paths:
        :param target_class:
        :return:
        """
        weight = 1 / len(paths)
        logging.info(f'Weighting factor for final SHAPS: {weight}')
        # Iterate through each path to load the SHAP files for the given class
        for path in paths:
            if path is None:
                continue

            # Use the specific class number in the filename
            file_shap_values = glob.glob(f'{path}/shap_values_class_{target_class}_ntu*.npy')

            if file_shap_values:
                shap_values = np.load(file_shap_values[0], mmap_mode='r+')
                logging.info(f"Loaded file: {file_shap_values[0]}")
            else:
                logging.info(f"No files found for shap_values_class_{target_class}_*.npy in {path}.")
                continue

            # Combine the SHAP values, labels, and names across multiple paths
            if self.combined_shap_values is None:
                self.combined_shap_values = shap_values * weight
            else:
                np.add(self.combined_shap_values, shap_values * weight, out=self.combined_shap_values)

            del shap_values

        # Raise error if no valid SHAP files were loaded
        if self.combined_shap_values is None:
            logging.error("No valid SHAP files found.")
            raise FileNotFoundError("No SHAP values files were found.")

        return self.combined_shap_values

    @staticmethod
    def extract_feature_values_ntu(data) -> dict:
        """
        Extract feature values from the provided data.
        :param data:
        :return:
        """
        return {
            "j_pos_x": data[:, 0, 0, ...],
            "j_pos_y": data[:, 0, 1, ...],
            "j_pos_z": data[:, 0, 2, ...],
            "j_pos_c_x": data[:, 0, 3, ...],
            "j_pos_c_y": data[:, 0, 4, ...],
            "j_pos_c_z": data[:, 0, 5, ...],
            "v_slow_x": data[:, 1, 0, ...],
            "v_slow_y": data[:, 1, 1, ...],
            "v_slow_z": data[:, 1, 2, ...],
            "v_fast_x": data[:, 1, 3, ...],
            "v_fast_y": data[:, 1, 4, ...],
            "v_fast_z": data[:, 1, 5, ...],
            "b_length_x": data[:, 2, 0, ...],
            "b_length_y": data[:, 2, 1, ...],
            "b_length_z": data[:, 2, 2, ...],
            "b_angle_x": data[:, 2, 3, ...],
            "b_angle_y": data[:, 2, 4, ...],
            "b_angle_z": data[:, 2, 5, ...],
            "a_slow_x": data[:, 3, 0, ...],
            "a_slow_y": data[:, 3, 1, ...],
            "a_slow_z": data[:, 3, 2, ...],
            "a_fast_x": data[:, 3, 3, ...],
            "a_fast_y": data[:, 3, 4, ...],
            "a_fast_z": data[:, 3, 5, ...]
        }

    @staticmethod
    def extract_feature_values_cp(data) -> dict:
        """
        Extract feature values from the provided data.
        :param data:
        :return:
        """
        return {
            "j_pos_x": data[:, 0, 0, ...],
            "j_pos_y": data[:, 0, 1, ...],
            "j_pos_c_x": data[:, 0, 2, ...],
            "j_pos_c_y": data[:, 0, 3, ...],
            "v_slow_x": data[:, 1, 0, ...],
            "v_slow_y": data[:, 1, 1, ...],
            "v_fast_x": data[:, 1, 2, ...],
            "v_fast_y": data[:, 1, 3, ...],
            "b_length_x": data[:, 2, 0, ...],
            "b_length_y": data[:, 2, 1, ...],
            "b_angle_x": data[:, 2, 2, ...],
            "b_angle_y": data[:, 2, 3, ...],
            "a_slow_x": data[:, 3, 0, ...],
            "a_slow_y": data[:, 3, 1, ...],
            "a_fast_x": data[:, 3, 2, ...],
            "a_fast_y": data[:, 3, 3, ...]
        }

    @staticmethod
    def average_values_time(shap_values, window_averaged=False):
        """
        Average SHAP values over time.
        :param shap_values:
        :param window_averaged:
        :return:
        """
        if window_averaged:
            averaged_parts = np.mean(shap_values, axis=0)
        else:
            averaged_parts = np.mean(shap_values, axis=1)
        return averaged_parts

    @staticmethod
    def average_values_body_groups(shap_values, parts):
        """
        Average SHAP values over body groups.
        :param shap_values:
        :param parts:
        :return:
        """
        index_joints = int(np.where(np.array(shap_values.shape) == 29)[0])

        if shap_values.ndim == 4:
            averaged_parts = [np.mean(shap_values[:, :, part, :], axis=index_joints) for part in parts]
            averaged_parts = np.stack(averaged_parts, axis=index_joints)
        elif shap_values.ndim == 3:
            averaged_parts = [np.mean(shap_values[:, part, :], axis=index_joints) for part in parts]
            averaged_parts = np.stack(averaged_parts, axis=index_joints)
        elif shap_values.ndim == 2:
            averaged_parts = [np.mean(shap_values[part, :], axis=index_joints) for part in parts]
            averaged_parts = np.stack(averaged_parts, axis=index_joints)
        else:
            raise ValueError("Shap values must have dimension 4, 3 or 2.")

        return averaged_parts

    @staticmethod
    def average_values_index(shap_values):
        """
        Average SHAP values over index.
        :param shap_values:
        :return:
        """
        return np.mean(shap_values, axis=0)

    def get_class_values(self, shap_values):
        """
        Get SHAP values for specific classes.
        :param shap_values:
        :return:
        """
        return shap_values[self.class_0_indices], shap_values[self.class_1_indices]

    @staticmethod
    def get_feature_data_at_index(dataloader, index):
        """
        Get feature data at specific index.
        :param dataloader:
        :param index:
        :return:
        """
        cumulative_sum = 0
        for x, _, _ in dataloader:
            batch_size = x.shape[0]
            if cumulative_sum + batch_size > index:
                batch_index = index - cumulative_sum
                return x[batch_index].cpu().detach().numpy()
            cumulative_sum += batch_size
        raise IndexError("Index out of range in DataLoader.")


class ShapSampler(Sampler):
    def __init__(self, indices, n_samples=50):
        super().__init__(indices)
        self.indices = indices
        self.n_samples = n_samples

    def __iter__(self):
        return iter(np.random.choice(self.indices, size=self.n_samples, replace=True))

    def __len__(self):
        return self.n_samples


def plot_skel_cp(val_loader, ax, window_idx, x, y):
    for joint_start, joint_end in val_loader.dataset.graph.neighbor_link:
        ax.plot([x[joint_start], x[joint_end]], [y[joint_start], y[joint_end]], color='gray')
    # Set axis limits and remove the coordinate system (no ticks or axis labels)
    ax.set_title(f'Window {window_idx}')
    ax.set_xlim([-0.25, 0.25])
    ax.set_ylim([-0.6, 0.5])
    ax.set_xticks([])
    ax.set_yticks([])


def plot_skel_ntu(val_loader, ax, window_idx, x, y, z):
    for joint_start, joint_end in val_loader.dataset.graph.neighbor_link:
        ax.plot([x[joint_start], x[joint_end]],
                [y[joint_start], y[joint_end]],
                [z[joint_start], z[joint_end]],
                color='gray')
    ax.set_title(f'Window {window_idx}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def load_gradcam_files(path):
    files_grad_values = sorted(glob.glob(f'{path}/gradcam_values_class_*.npy'))
    assert files_grad_values, "No GradCAM values were found!"
    # get class number and layer dynamically
    class_numbers = [int(re.search(r'class_(\d+)', s).group(1)) for s in files_grad_values]
    layer_types = [re.search(r'class_\d+_(.*?)_seed', path).group(1) for path in files_grad_values]
    logging.info(f"Found {len(files_grad_values)} GradCam values files for class {class_numbers} (+1)...")

    gradcam_values_ntu = []
    for file in files_grad_values:
        gradcam_values_file = np.load(file)
        gradcam_values_ntu.append(gradcam_values_file)
    logging.info("Loaded all the files...")

    return gradcam_values_ntu, class_numbers, layer_types


def load_gradcam_files_cp(path):
    """
    Expect CP data only one file.
    :param path:
    :return:
    """
    files_grad_values = sorted(glob.glob(f'{path}/gradcam_values_class_*.npy'))
    assert files_grad_values, "No GradCAM values were found!"

    class_numbers = []
    layer_types = []

    for file in files_grad_values:
        try:
            class_match = re.search(r'class_(all|\d+)', file)
            layer_match = re.search(r'class_(?:all|\d+)_(.*?)_seed', file)

            if class_match:
                class_numbers.append(class_match.group(1))  # 'all' or a specific class number
            else:
                raise ValueError(f"Class number not found in filename: {file}")

            if layer_match:
                layer_types.append(layer_match.group(1))
            else:
                raise ValueError(f"Layer type not found in filename: {file}")

        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")
            raise

    if len(files_grad_values) == 1:
        logging.info(f"Found 1 GradCam values file for class {class_numbers[0]}...")
    else:
        logging.info(f"Found {len(files_grad_values)} GradCam values files for classes {class_numbers}...")

    gradcam_values_cp = []
    for file in files_grad_values:
        gradcam_values_file = np.load(file)
        gradcam_values_cp.append(gradcam_values_file)

    logging.info("Loaded all the files...")

    return gradcam_values_cp, class_numbers, layer_types