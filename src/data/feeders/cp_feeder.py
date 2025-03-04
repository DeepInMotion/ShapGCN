import logging
import sys

import numpy as np
import math
import os

from torch.utils.data import Dataset
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt

from src.graph.graph import Graph
from src.data.feeders.feeder_helpers import BaseDataLoader


class CPDataloader(Dataset, BaseDataLoader):
    """
    Dataloader and preprocessor for CP-Skeleton dataset.
    """
    # TODO: fix this to be more dynamic
    DATA_ARGS = {
        'cp19': {'class': 2, 'folder': 'cp_19', 'shape': [4, 4, 150, 19, 1],
                 'train_set':   'train2',
                 'val_set':     'val2',
                 'test_set':    'test',
                 'layout':      'in-motion'},
        'cp29': {'class': 2, 'folder': 'cp_29', 'shape': [4, 4, 150, 29, 1],
                 'train_set':   'val13456',
                 'val_set':     'val27',
                 'test_set':    'test',
                 'layout':      'in-motion-2022'},
    }

    def __init__(self, phase, args):
        super().__init__()
        self.phase = phase
        self.args = args
        self.window_size = self.args.num_frame
        self.inputs = self.args.inputs
        self.debug = self.args.debug
        self.normalize = self.args.normalize
        self.mean_map = None
        self.std_map = None
        self.data_args = self.DATA_ARGS[self.args.dataset]
        logging.info(f"Data ARGS: {self.data_args}")
        self.shape = self.data_args['shape']
        self.num_classes = self.data_args['class']
        self.sub_folder = self.data_args['folder']

        # Values in check with Daniels code
        self.num_per_positive_sample = 30
        self.num_per_negative_sample = 5
        self.parts_distance = self.args.parts_distance

        # generate graph data depending on dataset
        self.args.layout = self.data_args['layout']
        self.graph = Graph(self.args.layout, strategy=self.args.strategy)
        self.A = self.graph.A
        self.parts = self.graph.parts
        self.conn = self.graph.connect_joint

        # get the paths
        if self.args.transform:
            # TODO make an effort to change those
            logging.info("Loading transformed data...")
            dataset_path = '{}/transformed/{}/'.format(self.args.root_folder, self.sub_folder)
        else:
            logging.info("Loading processed data...")
            dataset_path = '{}/processed/{}/'.format(self.args.root_folder, self.sub_folder)

        if self.phase == "train":
            self.path_data = os.path.join(dataset_path, '{}_coords.npy'.format(self.data_args['train_set']))
            self.path_labels = os.path.join(dataset_path, '{}_labels.npy'.format(self.data_args['train_set']))
            self.path_ids = os.path.join(dataset_path, '{}_ids.npy'.format(self.data_args['train_set']))
        elif self.phase == "eval":
            self.path_data = os.path.join(dataset_path, '{}_coords.npy'.format(self.data_args['val_set']))
            self.path_labels = os.path.join(dataset_path, '{}_labels.npy'.format(self.data_args['val_set']))
            self.path_ids = os.path.join(dataset_path, '{}_ids.npy'.format(self.data_args['val_set']))
        elif self.phase == "test":
            self.path_data = os.path.join(dataset_path, '{}_coords.npy'.format(self.data_args['test_set']))
            self.path_labels = os.path.join(dataset_path, '{}_labels.npy'.format(self.data_args['test_set']))
            self.path_ids = os.path.join(dataset_path, '{}_ids.npy'.format(self.data_args['test_set']))
        else:
            logging.error("Data mode [{}] not known!".format(self.phase))
            sys.exit()

        self.load_data()

        # if self.normalize:
        #     logging.info("Normalizing data...")
        #     self.mean_std(self.data)

        if self.args.augment and self.phase == "train":
            logging.info("Initializing the Augmenter...")

            from src.data.feeders.augmenter import Augmenter
            self.augmenter = Augmenter()
            logging.info("Applying following augmentations...")
            self.random_start = True
            self.random_perturbation = True
            self.roll_sequence = True
            if self.random_start:
                logging.info("Random start augmentation is enabled.")
            if self.random_perturbation:
                logging.info("Random perturbation augmentation is enabled.")
            if self.roll_sequence:
                logging.info("Roll sequence augmentation is enabled.")
        else:
            logging.info("NO augmentation on the dataset!")
            self.random_start = False
            self.random_perturbation = False
            self.roll_sequence = False

        if self.args.filter:
            self.butterworth_coeffs = signal.butter(self.args.filter_order, 0.5, output='sos')

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # Fetch part information
        data = np.array(self.data[index])
        name = self.names[index]
        label = self.labels[index]

        if self.phase == 'train':
            sequence_part, num_sequence_parts = self.seq_len[index]
            # Augment in train modus
            if self.random_start:
                data = self.augmenter.random_start(data, window_size=self.window_size,
                                                   roll_sequence=self.roll_sequence, sequence_part=sequence_part,
                                                   num_sequence_parts=num_sequence_parts)
            else:
                sequence_length = data.shape[1]
                if not self.roll_sequence:
                    try:
                        sequence_length = np.where(np.isnan(data[0, :, 0]))[0][0]
                    except IndexError:
                        pass
                part_size = math.floor(sequence_length / num_sequence_parts)
                window_start = min((sequence_part - 1) * part_size, sequence_length - self.window_size)
                data = data[:, window_start:window_start + self.window_size, :]

            # Perform data-augmentation by scaling, rotating and transforming the sequence
            if self.random_perturbation:
                data = self.augmenter.random_perturbation(data)

        C, T, V = data.shape
        data = np.reshape(data, (C, T, V, 1))

        # if self.normalize:
        #     data = (data - self.mean_map) / self.std_map
            # data = self.normalize_coordinates(data, method="z-score")

        # (C, max_frame, V, M) -> (I, C*2, T, V, M)
        joint, velocity, bone, accel = self.multi_input(data[:, :self.window_size, :, :])

        if self.normalize:
            joint = self.__normalize(joint, method="min-max")
            velocity = self.__normalize(velocity, method="min-max")
            bone = self.__normalize(bone, method="min-max")
            accel = self.__normalize(accel, method="min-max")

        data_new = []
        if 'J' in self.inputs:
            data_new.append(joint)
        if 'V' in self.inputs:
            data_new.append(velocity)
        if 'B' in self.inputs:
            data_new.append(bone)
        if 'A' in self.inputs:
            data_new.append(accel)
        data_new = np.stack(data_new, axis=0)

        return data_new, label, name

    def get_indices(self, index):
        assert index >= 0 and index <= 1, 'Class Index must be between 0 and 1.'
        all_indices = []
        for i in range(len(self.labels)):
            if self.labels[i] == index:
                all_indices.append(i)
        return all_indices

    def load_data(self):
        """
        Loads and processes data from specified paths.
        Balances dataset and handles sequence repetition if needed.
        """
        logging.info("Trying to load CP data...")
        try:
            self.data = np.load(self.path_data, mmap_mode='r')
            self.names = np.load(self.path_ids, mmap_mode='r')
            self.labels = np.load(self.path_labels, mmap_mode='r')
        except FileNotFoundError as e:
            missing_file = e.filename
            logging.error(f'Error: File not found - {missing_file}')
            raise ValueError(f'Error: Loading data files: {self.path_data} or {self.path_labels}') from e

        logging.info("Data loaded successfully. Processing samples...")

        samples = {}
        data, ids, labels, sequence_parts = [], [], [], []
        for i, (sample_id, sample_label) in enumerate(zip(self.names, self.labels)):
            samples[sample_id] = (self.data[i, ...], sample_label)

        if self.phase == 'train':
            logging.info("Processing train data...")
            data, ids, labels, sequence_parts = self.process_train(data, ids, labels, samples, sequence_parts)
        else:
            logging.info("Processing eval data...")
            data, ids, labels, sequence_parts = self.process_eval(data, ids, labels, samples, sequence_parts)

        if self.args.debug:
            logging.info('Loading SMALL dataset for debugging purposes.')
            data, ids, labels, seq_len = data[:200], ids[:200], labels[:200], sequence_parts[:200]

        self.data = np.array(data)
        self.names = np.array(ids)
        self.labels = np.array(labels)
        self.seq_len = sequence_parts

        logging.info('Done with loading data.')

    def process_train(self, data, ids, labels, samples, sequence_parts):
        for sample_id, (sample_data, sample_label) in samples.items():
            C, dataset_T, V = sample_data.shape

            try:
                sequence_T = np.where(np.isnan(sample_data[0, :, 0]))[0][0]
            except IndexError:
                sequence_T = dataset_T

            num_repetitions = math.ceil(dataset_T / sequence_T)
            sequence_data = sample_data[:, :sequence_T, :]
            repeated_sample_data = np.zeros((C, num_repetitions * sequence_T, V))

            for n in range(num_repetitions):
                repeated_sample_data[:, n * sequence_T:(n + 1) * sequence_T, :] = sequence_data

            sample_data = repeated_sample_data[:, :dataset_T, :]

            if sample_label == 1:
                for part in range(1, self.num_per_positive_sample + 1):
                    data.append(sample_data)
                    ids.append(sample_id)
                    labels.append(sample_label)
                    sequence_parts.append((part, self.num_per_positive_sample))
            else:
                for part in range(1, self.num_per_negative_sample + 1):
                    data.append(sample_data)
                    ids.append(sample_id)
                    labels.append(sample_label)
                    sequence_parts.append((part, self.num_per_negative_sample))
        return data, ids, labels, sequence_parts

    def process_eval(self, data, ids, labels, samples, sequence_parts):
        for sample_id, (sample_data, sample_label) in samples.items():
            try:
                sample_num_frames = np.where(np.isnan(sample_data[0, :, 0]))[0][0]
            except IndexError:
                sample_num_frames = sample_data.shape[1]

            # Construct sequence parts of self.window_size length
            # and self.parts_distance distance to consecutive parts
            for start_frame in range(0, sample_num_frames, self.parts_distance):
                if start_frame > sample_num_frames - self.window_size:
                    data.append(sample_data[:, sample_num_frames - self.window_size:sample_num_frames, :])
                    ids.append(sample_id)
                    labels.append(sample_label)
                    break
                else:
                    data.append(sample_data[:, start_frame:start_frame + self.window_size, :])
                    ids.append(sample_id)
                    labels.append(sample_label)
        return data, ids, labels, sequence_parts

    def multi_input(self, data):
        """
        Generate features from data.
        CP 29 data is already preprocessed within the numpy files. 
        This includes:
        - butterworth filter, 
        - resampling to 30Hz,
        - Frame-level trunk centralization and alignment,
        - Sequence-level scale normalization.
        :param data:
        :return: features
        """
        C, T, V, M = data.shape
        joint = np.zeros((C * 2, T, V, M), dtype=np.float32)
        velocity = np.zeros((C * 2, T, V, M), dtype=np.float32)
        accel = np.zeros((C * 2, T, V, M), dtype=np.float32)
        bone = np.zeros((C * 2, T, V, M), dtype=np.float32)
        joint[:C, :, :, :] = data
        for i in range(V):
            joint[C:, :, i, :] = data[:, :, i, :] - data[:, :, 1, :]
        for i in range(T - 2):
            velocity[:C, i, :, :] = (data[:, i + 1, :, :] - data[:, i, :, :]) / 1
            velocity[C:, i, :, :] = (data[:, i + 2, :, :] - data[:, i, :, :]) / 2
        for i in range(len(self.conn)):
            bone[:C, :, i, :] = data[:, :, i, :] - data[:, :, self.conn[i], :]
        bone_length = 0
        for i in range(C):
            bone_length += bone[i, :, :, :] ** 2
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C):
            bone[C + i, :, :, :] = np.arccos(bone[i, :, :, :] / bone_length)
        # acceleration
        if self.args.filter:
            filtered_signal = self.butterworth_filter(velocity, self.butterworth_coeffs)
            for i in range(T - 2):
                accel[:C, i, :, :] = (filtered_signal[:C, i + 1, :, :] - filtered_signal[:C, i, :, :]) / 1
                accel[C:, i, :, :] = (filtered_signal[C:, i + 2, :, :] - filtered_signal[C:, i, :, :]) / 2
        else:
            for i in range(T - 2):
                accel[:C, i, :, :] = (velocity[:C, i + 1, :, :] - velocity[:C, i, :, :]) / 1
                accel[C:, i, :, :] = (velocity[C:, i + 2, :, :] - velocity[C:, i, :, :]) / 2

        return joint, velocity, bone, accel

    def __normalize(self, feature, method="min-max"):

        f1_subset = feature[0:2, ...]
        f2_subset = feature[2:, ...]

        # Function to normalize a subset of data
        def normalize_subset(subset):
            combined = subset.reshape(2, -1)  # Flatten to (2, 150 * 29)
            old_min = np.min(combined)
            old_max = np.max(combined)
            normalized_combined = -1 + 2 * (combined - old_min) / (old_max - old_min)
            return normalized_combined.reshape(subset.shape)

        # Normalize each subset independently
        normalized_f1 = normalize_subset(f1_subset)
        normalized_f2 = normalize_subset(f2_subset)

        # Reassemble the data
        normalized_data = np.concatenate([normalized_f1, normalized_f2], axis=0)
        return normalized_data

    def normalize_coordinates(self, data, method="min-max"):
        """
        Normalize x and y coordinates separately.
        :param data: Input data of shape (C, T, V, M) where C holds coordinates like [x, y, x, y]
        :param method: Normalization method ("min-max", "z-score", or "mean")
        :return: Normalized data of the same shape
        """
        C, T, V, M = data.shape
        normalized_data = np.zeros_like(data)

        # Step by 2 to handle [x, y, x, y]
        normalized_data[0, :, :, :] = self.normalize_single(data[0, :, :, :], method, axis=(0, 1))
        normalized_data[1, :, :, :] = self.normalize_single(data[1, :, :, :], method, axis=(0, 1))

        return normalized_data

    def normalize_single(self, feature, method="z-score", axis=(0, 1)):
        """
        Apply normalization to a single feature.
        :param feature: Feature array to normalize
        :param method: Normalization method ("min-max", "z-score", or "mean")
        :param axis: Axes to apply normalization over
        :return: Normalized feature
        """
        if method == "min-max":
            min_val = np.min(feature, axis=axis, keepdims=True)
            max_val = np.max(feature, axis=axis, keepdims=True)
            normalized = (feature - min_val) / (max_val - min_val + 1e-5)
        elif method == "z-score":
            mean = np.mean(feature, axis=axis, keepdims=True)
            std = np.std(feature, axis=axis, keepdims=True) + 1e-5
            normalized = (feature - mean) / std
        elif method == "mean":
            mean = np.mean(feature, axis=axis, keepdims=True)
            min_val = np.min(feature, axis=axis, keepdims=True)
            max_val = np.max(feature, axis=axis, keepdims=True)
            normalized = (feature - mean) / (max_val - min_val + 1e-5)
        return normalized



