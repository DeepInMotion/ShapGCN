import logging
import glob
import numpy as np
from torch.utils.data import Sampler


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
