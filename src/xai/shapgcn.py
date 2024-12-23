import collections
import copy
import gc
import pickle
import sys
import os

import shap
import torch
import logging
import numpy as np
from tqdm import tqdm
from time import time
import pandas as pd
from captum.attr import DeepLiftShap
from torch.utils.data import DataLoader, RandomSampler, Subset

from src.initializer import Initializer
from src.model.trainer import Trainer
from src.model.student import Student
from src.xai.perturber import Perturber
from src.utils import utils
from src.xai.shap_vis import ShapVisualizer
from src.xai.xai_utils import ShapSampler, ShapHandler, NTU_JOINT_NAMES, CP_JOINT_NAMES_29


SEPARATOR = "-" * 50


class ShapGCN(Initializer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.trainer = None
        self.student_model = None
        self.best_state = None
        self.argmax_epochs = self.args.argmax_epochs

        self.eval_interval = self.args.eval_interval

        # XAI configs
        self.model_path = os.path.join(self.args.work_dir, self.args.xai_model_path)
        logging.info(f"Loading XAI model from: {self.model_path}")
        self.num_background = self.args.xai_num_back

        cp_suffix = f'{self.args.dataset}' if self.args.dataset in ['cp19', 'cp29'] else 'ntu'
        self.cp = self.args.dataset in ['cp19', 'cp29']

        self.xai_random_back = self.args.xai_random_back
        background_type = '_random_' if self.xai_random_back else '_'

        # Seed int if SHAP cannot be computed on small GPU; change in config to generate a new background
        self.xai_back_seed = str(self.args.xai_back_seed)

        # Update background_name
        self.background_name = f'back_shap{background_type}{cp_suffix}_seed_{self.xai_back_seed}.npy'
        self.shap_save_string = f'{cp_suffix}_seed_{self.xai_back_seed}.npy'

        self.background_folder = os.path.join(os.path.dirname(self.save_dir), 'background', self.xai_back_seed)
        self.background_path = os.path.join(self.background_folder, self.background_name)
        os.makedirs(self.background_folder, exist_ok=True)
        self.background_set, self.background_labels, self.background_names = None, None, None

        if not self.xai_random_back:
            self.xai_back_idx = self.args.xai_back_idx

        # body parts
        self.body_parts = self.val_loader.dataset.graph.parts

        # XAI explainers and perturber objects
        self.explainer_deep = None
        self.explainer_grad = None
        self.perturber = None
        self.perturb_handler = None

        self.gen_parts_names = ['left arm', 'right arm', 'left leg', 'right leg', 'torso']
        self.cp_joint_names_29 = CP_JOINT_NAMES_29
        self.ntu_joint_names = NTU_JOINT_NAMES


    def start(self):
        """
        Entry point for ShapGCN
        """
        torch.autograd.set_detect_anomaly(True)
        if self.cp:
            xai_methods_cp = {
                "shap": self.xai_shap_cp,
                "perturb": self.perturb_edge_experiment,
                "perturb_shap": self.perturb_shap_cp,
                "vis": self.visualize_shap_cp,
            }
            method = xai_methods_cp.get(self.args.xai_method)
        elif self.args.dataset in ["xsub60", "xview60", "xsub120", "xsetup120"]:
            xai_methods_ntu = {
                "shap": self.xai_shap_ntu,
                "perturb": self.perturb_edge_experiment,
                "perturb_shap": self.perturb_shap_ntu,
                "vis": self.visualize_shap_ntu,
            }
            method = xai_methods_ntu.get(self.args.xai_method)
        else:
            logging.error(f"Unknown dataset for SHAP: {self.args.dataset}")
            sys.exit(1)

        if method:
            method()
        else:
            logging.error(f"Unknown xai method: {self.args.xai_method}")
            sys.exit(1)

    def perturb_shap_ntu(self):
        """
        Perturbation experiments for the NTU data.
        :return:
        """
        # Experiment variables
        individual_joint = True
        config_perturb = {
            'thresh': 0.9,
            'dimension': 1,
            'percent': True,
            'zero_out': True,
            'change_towards_zero': False,
            'switch_sign': True,
            'top_joints': 5
        }
        # Flag for random perturbation experiment
        rand_per = False
        target_class = 15
        parts_range = range(len(self.ntu_joint_names))
        ntu_names = self.ntu_parts_names if not individual_joint else self.ntu_joint_names
        logging.info(f"Random perturbation: {rand_per}")
        logging.info(f"SHAP values are computed on class: {target_class}")

        # Define the list of paths for the NTU shap values
        paths = [
            "logs/xai/three_classes_xview60/2024-10-22_10-33-43",
            "logs/xai/three_classes_xview60/2024-10-22_10-37-59",
            "logs/xai/three_classes_xview60/2024-10-22_10-39-49",
            "logs/xai/three_classes_xview60/2024-10-22_10-48-22",
            "logs/xai/three_classes_xview60/2024-10-22_10-50-21",
        ]

        self.perturb_handler = ShapHandler()
        self.perturber = Perturber(body_parts=self.val_loader.dataset.graph.parts, cp=False,
                                   individual_joints=individual_joint)

        indices_class = self.val_loader.dataset.get_indices(target_class)

        shap_values_ntu = self.perturb_handler.load_shap_values_ntu(paths, target_class)

        shap_values_dict = self.perturb_handler.extract_feature_values_ntu(shap_values_ntu)

        sorted_shaps_ntu = self.compute_shap_ordering_ntu(shap_values_dict, individual_joint, ntu_names, parts_range,
                                                          target_class, indices_class)

        # loop over the parts perturb the model and check the result
        weights_original = self._load_weights()
        self.__build_student(weights_original, 0)
        self.assign_weights_opt_lrs(weights_original)
        # init trainer once and only replace the student
        self.trainer = Trainer(self.args, self.student_model, self.train_loader, self.val_loader,
                               self.loss_func, self.optimizer, self.scheduler, self.scheduler_warmup,
                               self.writer, self.gpu_id, self.num_classes, self.no_progress_bar,
                               self.new_batch_size)

        thresh_values = np.arange(0.5, 0.55, 0.05)
        for thresh in thresh_values:
            for top_joints in range(1, 11):
                # Update config with current parameters
                config_perturb['zero_out'] = False
                config_perturb['thresh'] = thresh
                config_perturb['top_joints'] = top_joints

                logging.info(f"Experiment config is: {config_perturb}")

                predictions_by_index = self.predict_perturbed(config_perturb, individual_joint, sorted_shaps_ntu,
                                                                  copy.deepcopy(weights_original), rand_per=rand_per)

                filename = (
                    f"preds_shap_class_ntu_{target_class}_"
                    f"t{int(config_perturb['thresh'] * 100)}_"
                    f"z{int(config_perturb['zero_out'])}_"
                    f"c{int(config_perturb['change_towards_zero'])}_"
                    f"d{config_perturb['dimension']}_"
                    f"p{int(config_perturb['percent'])}_"
                    f"s{int(config_perturb['switch_sign'])}_"
                    f"tj{int(config_perturb['top_joints'])}.pkl"
                )

                with open(os.path.join(self.save_dir, filename), 'wb') as f:
                    logging.info(f"Saving predictions...")
                    pickle.dump(predictions_by_index, f)

        logging.info("Done with perturbation experiment...")

    def perturb_shap_cp(self):
        """
        Perturb the SHAP values of the CP dataset depending on the highest FI derived from the SHAP values.
        Therefor, for the CP data the most important features from J,V,B and A are summed together, resulting in the
        most important body group for this window.
        Also, there is the possibility to aggregate the most important SHAP values based on the individual key points.
        :return:
        """
        # Experiment variables
        individual_joint = True
        config_perturb = {
            'thresh': 0.9,
            'dimension': 1,
            'percent': True,
            'zero_out': False,
            'change_towards_zero': False,
            'switch_sign': True,
            'top_joints': 5
        }
        # Flag for random perturbation experiment
        rand_per = True
        # put these to true if computed
        computed = False
        computed_path = ""
        logging.info(f"Random perturbation: {rand_per}")
        shap_class = 0
        logging.info(f"SHAP class focus {shap_class}")
        parts_range = range(len(self.gen_parts_names)) if not individual_joint else range(len(self.cp_joint_names_29))
        cp_names = self.gen_parts_names if not individual_joint else self.cp_joint_names_29
        logging.info(f"Shap class focus is: {shap_class}")

        path_shap_preds = ""

        # Define the list of paths
        paths = [
            "logs/xai/exp_new_model_cp29/2024-10-02_14-09-53",
            "logs/xai/exp_new_model_cp29/2024-10-07_12-03-36",
            "logs/xai/exp_new_model_cp29/2024-10-07_12-11-08",
            "logs/xai/exp_new_model_cp29/2024-10-07_14-39-46",
            "logs/xai/exp_new_model_cp29/2024-10-07_15-36-50",
            "logs/xai/exp_new_model_cp29/2024-10-08_14-13-09"
        ]

        self.perturb_handler = ShapHandler()
        self.perturber = Perturber(body_parts=self.val_loader.dataset.graph.parts, individual_joints=individual_joint)

        if not computed:
            shap_values_cp, shap_names_cp, shap_labels_cp = self.perturb_handler.load_shap_values_cp(paths)
        else:
            shap_values_cp, shap_names_cp, shap_labels_cp = None, None, None

        class_0_indices = []
        class_1_indices = []
        # Track the indices of Class 0 and Class 1
        for idx, label in enumerate(shap_labels_cp):
            if label == 0:
                class_0_indices.append(idx)
            elif label == 1:
                class_1_indices.append(idx)

        shap_values_dict = self.perturb_handler.extract_feature_values_cp(shap_values_cp)

        sorted_shaps = self.compute_shap_ordering_cp(class_0_indices, class_1_indices, cp_names, individual_joint,
                                                     parts_range, shap_values_dict, shap_class)

        with open(os.path.join(self.save_dir, "shap_importance.pkl"), 'wb') as f:
            logging.info(f"Saving sorted SHAPS...")
            pickle.dump(sorted_shaps, f)

        # loop over the parts perturb the model and check the result
        weights_original = self._load_weights()
        self.__build_student(weights_original, 0)
        self.assign_weights_opt_lrs(weights_original)
        # init trainer once and only replace the student
        self.trainer = Trainer(self.args, self.student_model, self.train_loader, self.val_loader,
                               self.loss_func, self.optimizer, self.scheduler, self.scheduler_warmup,
                               self.writer, self.gpu_id, self.num_classes, self.no_progress_bar,
                               self.new_batch_size)


        for zero_out in [True, False]:
            # Set thresh values depending on zero_out condition
            thresh_values = [None] # if zero_out else np.arange(0.7, 0.95, 0.05)
            # Loop through threshold values (or just [None] if zero_out is True)
            for thresh in thresh_values:
                # Loop through top_joints from 1 to 10
                for top_joints in range(1, 11):
                    # Update config with current parameters
                    if not zero_out:
                        config_perturb['thresh'] = thresh
                    config_perturb['zero_out'] = zero_out
                    config_perturb['top_joints'] = top_joints

                    logging.info(f"Experiment config is: {config_perturb}")

                    predictions_by_index = self.predict_perturbed(config_perturb, individual_joint, sorted_shaps,
                                                                      copy.deepcopy(weights_original), rand_per=rand_per)

                    filename = (
                        f"preds_shap_class_{shap_class}_"
                        f"t{int(config_perturb['thresh'] * 100)}_"
                        f"z{int(config_perturb['zero_out'])}_"
                        f"c{int(config_perturb['change_towards_zero'])}_"
                        f"d{config_perturb['dimension']}_"
                        f"p{int(config_perturb['percent'])}_"
                        f"s{int(config_perturb['switch_sign'])}_"
                        f"tj{int(config_perturb['top_joints'])}.pkl"
                    )

                    with open(os.path.join(self.save_dir, filename), 'wb') as f:
                        logging.info(f"Saving predictions...")
                        pickle.dump(predictions_by_index, f)

        logging.info("Done with perturbation experiment...")

    def predict_perturbed(self, config_perturb, individual_joint, sorted_shaps, weights_original, rand_per=False):
        """
        Predict the perturbed data...
        :param config_perturb:
        :param individual_joint:
        :param sorted_shaps:
        :param weights_original:
        :param rand_per:
        :return:
        """
        if config_perturb['top_joints']:
            assert individual_joint is True and config_perturb['top_joints'] > 0

        predictions_by_index = collections.OrderedDict()
        self.student_model.eval()

        for index, shap_importance in tqdm(enumerate(sorted_shaps), total=len(sorted_shaps), desc="SHAP perturbation"):
            if not self.cp:
                # get index from tuple depending on class for ntu
                index = shap_importance[1]
                shap_importance = shap_importance[0]
            data, label, name = self.val_loader.dataset[index]
            data = torch.from_numpy(data).to(self.gpu_id).unsqueeze(0)
            predictions_by_index[index] = {
                'label': label,
                'video_id': name,
                'preds': [],
                'shaps': shap_importance
            }

            weights_dummy = collections.OrderedDict()
            # Copy contents of sub_dicts into model_weights
            weights_dummy.update(copy.deepcopy(weights_original['model']['input_stream']))
            weights_dummy.update(copy.deepcopy(weights_original['model']['main_stream']))
            weights_dummy.update(copy.deepcopy(weights_original['model']['classifier']))

            # Group joints for perturbation
            if config_perturb["top_joints"]:
                if rand_per:
                    joints = list(self.perturber.parts_mapping.keys())
                    positive_joints = np.random.choice(joints, config_perturb['top_joints'], replace=False).tolist()
                    negative_joints = np.random.choice(joints, config_perturb['top_joints'], replace=False).tolist()
                else:
                    # Group joints by positive and negative values from shap_importance
                    positive_joints = [joint for joint in shap_importance if joint[1] > 0]
                    positive_joints = positive_joints[:config_perturb['top_joints']]
                    negative_joints = [joint for joint in shap_importance if joint[1] < 0]
                    negative_joints = negative_joints[-config_perturb['top_joints']:]
                joint_groups = [positive_joints, negative_joints]
            else:
                # process all joints
                joint_groups = shap_importance

            for joints in joint_groups:
                if isinstance(joints, (list, tuple)):
                    try:
                        # Attempt to unpack if it's a list of tuples
                        parts = [joint for joint, _ in joints]
                    except ValueError:
                        # If unpacking fails, use the whole list/tuple as parts
                        parts = joints if isinstance(joints, list) else joints[0]
                else:
                    raise ValueError("Recheck the parts variable...")

                self.perturber.edges = parts
                model_weights_perturbed = self.perturber.perturb_edge_matrix(copy.deepcopy(weights_dummy),
                                                                             config_perturb, log=False)
                # put the new weights to the model
                self.assign_weights_model(model_weights_perturbed)

                with torch.no_grad():
                    out = self.student_model(data)

                if not self.cp:
                    reco_top1 = out.max(1)[1]
                    predictions_by_index[index]['preds'].append(reco_top1.cpu().numpy())
                else:
                    predictions_by_index[index]['preds'].append(out.data.cpu().numpy())
                del model_weights_perturbed

            del weights_dummy

        return predictions_by_index

    def compute_shap_ordering_cp(self, class_0_indices, class_1_indices, cp_names, individual_joint, parts_range,
                                 shap_values_dict, shap_class):
        """
        Compute the SHAP ordering of the CP dataset for the perturbation.

        :param class_0_indices:
        :param class_1_indices:
        :param cp_names:
        :param individual_joint:
        :param parts_range:
        :param shap_values_dict:
        :param shap_class:
        :return:
        """
        shap_values_class_0 = {}
        shap_values_class_1 = {}
        averaged_shap_class_1 = {}
        averaged_shap_class_0 = {}
        # Loop through the SHAP values and get class-specific values
        for key, value in shap_values_dict.items():
            shap_values_class_0[key], shap_values_class_1[key] = self.perturb_handler.get_class_values(value)

        if not individual_joint:
            for key, value in shap_values_class_1.items():
                averaged_shap_class_1[key] = self.perturb_handler.average_values_body_groups(value, self.body_parts)

            for key, value in shap_values_class_0.items():
                averaged_shap_class_0[key] = self.perturb_handler.average_values_body_groups(value, self.body_parts)

        else:
            averaged_shap_class_0 = shap_values_class_0
            averaged_shap_class_1 = shap_values_class_1

        sorted_mean_shap_values_1 = {}
        sorted_mean_shap_values_0 = {}

        for feature, shap_values_data in averaged_shap_class_1.items():
            sorted_mean_shap_values_1[feature] = []
            num_instances = shap_values_data.shape[0]

            for i in range(num_instances):
                mean_shap_values = []
                shap_values = shap_values_data[i, :, :, shap_class]  # [samples, time, body parts/joints, shap class]

                for part_idx in parts_range:
                    mean_value_for_part = np.mean(shap_values[:, part_idx])  # Mean over time steps for this body part
                    mean_shap_values.append((cp_names[part_idx], mean_value_for_part))

                sorted_mean_shap_values_for_instance = sorted(mean_shap_values, key=lambda x: x[1], reverse=True)
                sorted_mean_shap_values_1[feature].append((i, sorted_mean_shap_values_for_instance))

        for feature, shap_values_data in averaged_shap_class_0.items():
            sorted_mean_shap_values_0[feature] = []
            num_instances = shap_values_data.shape[0]

            for i in range(num_instances):
                mean_shap_values = []
                shap_values = shap_values_data[i, :, :, shap_class]  # [time steps, body parts, shap class]

                for part_idx in parts_range:
                    mean_value_for_part = np.mean(shap_values[:, part_idx])  # Mean over time steps for this body part
                    mean_shap_values.append((cp_names[part_idx], mean_value_for_part))

                sorted_mean_shap_values_for_instance = sorted(mean_shap_values, key=lambda x: x[1], reverse=True)
                sorted_mean_shap_values_0[feature].append((i, sorted_mean_shap_values_for_instance))

        cumulative_shap_by_body_part_0 = collections.defaultdict(lambda: collections.defaultdict(float))
        cumulative_shap_by_body_part_1 = collections.defaultdict(lambda: collections.defaultdict(float))
        # Iterate over each feature in sorted_mean_shap_values
        for feature, instances in sorted_mean_shap_values_1.items():
            for index, mean_shap_values in instances:
                for body_part, mean_value in mean_shap_values:
                    # Accumulate the SHAP values for this body part and index
                    cumulative_shap_by_body_part_1[index][body_part] += mean_value
        # Iterate over each feature in sorted_mean_shap_values
        for feature, instances in sorted_mean_shap_values_0.items():
            for index, mean_shap_values in instances:
                for body_part, mean_value in mean_shap_values:
                    # Accumulate the SHAP values for this body part and index
                    cumulative_shap_by_body_part_0[index][body_part] += mean_value
        # Sort the body parts by their cumulative SHAP values for each index
        sorted_body_parts_by_index_1 = []
        sorted_body_parts_by_index_0 = []
        for index, body_part_shap_values in cumulative_shap_by_body_part_1.items():
            sorted_body_parts = sorted(body_part_shap_values.items(), key=lambda x: x[1], reverse=True)
            sorted_body_parts_by_index_1.append(sorted_body_parts)
        for index, body_part_shap_values in cumulative_shap_by_body_part_0.items():
            sorted_body_parts = sorted(body_part_shap_values.items(), key=lambda x: x[1], reverse=True)
            sorted_body_parts_by_index_0.append(sorted_body_parts)
        # Combine class 0 SHAP values with their original class 0 indices
        sorted_class_0_with_index = list(zip(sorted_body_parts_by_index_0, class_0_indices))
        # Combine class 1 SHAP values with their original class 1 indices
        sorted_class_1_with_index = list(zip(sorted_body_parts_by_index_1, class_1_indices))
        # Step 3: Merge both class 0 and class 1 values
        combined_shap_values_with_index = sorted_class_0_with_index + sorted_class_1_with_index
        # Step 4: Sort the combined values based on the original shap_labels_cp indices
        sorted_combined_list = sorted(combined_shap_values_with_index, key=lambda x: x[1])
        sorted_shaps = [value for value, index in sorted_combined_list]
        return sorted_shaps

    def compute_shap_ordering_ntu(self, shap_values_dict: dict, indivdual_joint: bool, ntu_names: list,
                                  parts_range: list, shap_class: int, indices_class: list):
        """
        Compute the ordering for one class of the NTU shaps.
        :return:
        """
        averaged_shap_class_ntu = {}
        sorted_mean_shap_ntu = {}

        if not indivdual_joint:
            logging.info("Averaging SHAPs over body parts")
            for key, value in shap_values_dict.items():
                averaged_shap_class_ntu[key] = self.perturb_handler.average_values_body_groups(value, self.body_parts)
        else:
            logging.info("Taking all SHAPs")
            averaged_shap_class_ntu = shap_values_dict

        for feature, shap_values_data in averaged_shap_class_ntu.items():
            sorted_mean_shap_ntu[feature] = []
            num_instances = shap_values_data.shape[0]

            for i in range(num_instances):
                mean_shap_values = []
                shap_values = shap_values_data[i, :, :, shap_class]  # [samples, time, body parts/joints, 60]

                for part_idx in parts_range:
                    mean_value_for_part = np.mean(shap_values[:, part_idx])  # Mean over time steps for this body part
                    mean_shap_values.append((ntu_names[part_idx], mean_value_for_part))

                sorted_mean_shap_values_for_instance = sorted(mean_shap_values, key=lambda x: x[1], reverse=True)
                sorted_mean_shap_ntu[feature].append((i, sorted_mean_shap_values_for_instance))

        cumulative_shap_by_body_part_ntu = collections.defaultdict(lambda: collections.defaultdict(float))
        # Iterate over each feature in sorted_mean_shap_values
        for feature, instances in sorted_mean_shap_ntu.items():
            for index, mean_shap_values in instances:
                for body_part, mean_value in mean_shap_values:
                    # Accumulate the SHAP values for this body part and index
                    cumulative_shap_by_body_part_ntu[index][body_part] += mean_value

        # Sort the body parts by their cumulative SHAP values for each index
        sorted_body_parts_by_index_ntu = []
        for index, body_part_shap_values in cumulative_shap_by_body_part_ntu.items():
            sorted_body_parts = sorted(body_part_shap_values.items(), key=lambda x: x[1], reverse=True)
            sorted_body_parts_by_index_ntu.append(sorted_body_parts)
        # Combine SHAP values with their original class 0 indices
        sorted_ntu_with_index = list(zip(sorted_body_parts_by_index_ntu, indices_class))
        return sorted_ntu_with_index

    def perturb_edge_experiment(self):
        """
        1. Build student model and change the edges
        2. Iterate over the experiment values
        3. Safe cm and model info for later use
        :return:
        """
        logging.info("Initializing Perturber...")
        self.perturber = Perturber()
        logging.info("Test perturbation for Threshold in body parts...")

        # Perturbation parameters
        # threshold = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        # for "normal" threshold --> 0.25 -> 10.0
        # threshold = [0.25 * i for i in range(1, 41)]
        # for percentage 0.025 --> 1.0
        threshold = [round(0.025 * i, 3) for i in range(1, 41)]
        change_tow_zero = False
        switch_sign = True
        perc = True
        dimension = [1]

        logging.info("Threshold values: {}".format(threshold))
        logging.info("Change towards zero: {}".format(change_tow_zero))
        logging.info("Switch sign: {}".format(switch_sign))
        logging.info("Percentage: {}".format(perc))
        logging.info("Dimension: {}".format(dimension))

        body_parts = self.val_loader.dataset.graph.parts
        experiment_idx = 0
        experiment_data = []
        max_value = len(self.feeder_val.graph.__str__())
        # dimension = [i for i in range(1, max_value + 1)]

        # get checkpoint file and load skeleton student
        checkpoint = self._load_weights()
        self.__build_student(checkpoint, experiment_idx)
        self.assign_weights_opt_lrs(checkpoint)

        logging.info(SEPARATOR)
        logging.info("Starting experiments...")
        for dim in dimension:
            logging.info("Experiment with dimension set to: 0-{}".format(dim))
            # for sign in sign_change:
            #     sign_bool = bool(sign)
            #     logging.info("Experiment with changing sign set to: {}".format(sign_bool))
            for thresh in threshold:
                logging.info("Experiment with threshold set to: {}".format(thresh))
                for parts in body_parts:
                    logging.info("Experiment with altered body parts: {}".format(parts))
                    logging.info("Experiment counter: {}...".format(experiment_idx))
                    self.perturber.edges = parts
                    # edge_mask = self.val_loader.dataset.graph.get_masking_matrix(parts)
                    # put the model weights separate
                    model_weights = collections.OrderedDict()
                    # Copy contents of sub_dicts into model_weights
                    model_weights.update(checkpoint['model']['input_stream'])
                    model_weights.update(checkpoint['model']['main_stream'])
                    model_weights.update(checkpoint['model']['classifier'])

                    model_weights_update = self.perturber.perturb_edge_matrix(dict, model_weights)

                    self.assign_weights_model(model_weights_update)
                    evaL_results = self.infer_student(experiment_idx)

                    # safe the stuff
                    experiment_data.append({
                        "index": experiment_idx,
                        "dimension": dim,
                        "towards_zero": change_tow_zero,
                        "switch_sign": switch_sign,
                        "threshold": thresh,
                        "body_parts": parts,
                        "acc": evaL_results.get('acc', 0),
                        "auc": evaL_results.get('auc', 0),
                        'precision': evaL_results.get('precision', 0),
                        'recall': evaL_results.get('recall', 0),
                        'F1': evaL_results.get('F1', 0),
                        'sens': evaL_results.get('sens', 0),
                        'spec': evaL_results.get('spec', 0),
                        'ppv': evaL_results.get('ppv', 0),
                        'npv': evaL_results.get('npv', 0),
                        'b_acc': evaL_results.get('b_acc', 0)
                    })

                    logging.info("Done with experiment {}...".format(experiment_idx))
                    logging.info(SEPARATOR)
                    experiment_idx += 1
                    del model_weights_update, model_weights

        save_dir_experiment = os.path.join(self.save_dir, "perturb_data.csv")
        logging.info("Saving experiments...")
        df = pd.DataFrame(experiment_data)
        df.to_csv(save_dir_experiment, index=False)
        logging.info("Done with perturbation experiment!")

    def xai_shap_cp(self, method="shap"):
        """
        Compute SHAP values for the CP dataset using either SHAP DeepExplainer or Captum DeepLiftShap.
        :param method:
        :return:
        """
        logging.info(f"Initializing SHAP model for CP dataset using {method}...")

        # Initialize the background set and load the model
        self.background_set, self.background_labels, self.background_names = self._shap_background()
        self.init_student_model()
        logging.info("Initialized model...")

        # Depending on the method, initialize the appropriate explainer
        if method == "shap":
            self.explainer = shap.DeepExplainer(self.student_model.to(self.gpu_id),
                                                self.background_set.to(self.gpu_id))
        elif method == "captum":
            self.explainer = DeepLiftShap(self.student_model.to(self.gpu_id))
            self.background_set = self.background_set.to(self.gpu_id)
            self.update_batch_size(2)
        else:
            raise ValueError("Invalid method specified. Use 'shap' or 'captum'.")

        shap_values_cp = []
        shap_labels_cp = []
        shap_names_cp = []
        shap_expected_value_cp = []
        logging.info(f"Computing SHAP values for CP using the {method} package...")
        shap_cp_iter = self.val_loader if self.no_progress_bar else tqdm(self.val_loader, leave=True, desc="SHAP iter")
        for num, (x, y, name) in enumerate(shap_cp_iter):
            x_batch = x.to(self.gpu_id)
            if method == "shap":
                shap_values_batch = self.explainer.shap_values(x_batch, check_additivity=False)
                shap_expected_value_cp.append(self.explainer.expected_value)
            elif method == "captum":
                shap_values_batch = self.explainer.attribute(inputs=x_batch, baselines=self.background_set, target=1)
                shap_values_batch = np.array(shap_values_batch.cpu().detach(), dtype=np.float32)
            else:
                raise ValueError("Invalid method specified. Use 'shap' or 'captum'.")

            shap_values_batch = np.array(shap_values_batch, dtype=np.float32)
            shap_values_cp.append(shap_values_batch)
            shap_labels_cp.append(y)
            shap_names_cp.append(name)

            del x_batch, shap_values_batch
            torch.cuda.empty_cache()
            gc.collect()

        shap_values_cp = np.array(shap_values_cp, dtype=np.float32)

        shap_values_cp = shap_values_cp.squeeze(axis=-2)
        logging.info(f"Shape of the compressed SHAP values: {shap_values_cp.shape}")

        new_shape = (shap_values_cp.shape[0] * shap_values_cp.shape[1],) + shap_values_cp.shape[2:]
        shap_values_cp = shap_values_cp.reshape(new_shape)
        logging.info(f"Shape of the reshaped SHAP values: {shap_values_cp.shape}")

        # squeeze here
        shap_labels_cp = np.concatenate(shap_labels_cp, axis=0)
        shap_names_cp = np.concatenate(shap_names_cp, axis=0)
        logging.info(f"Length of the SHAP ids: {shap_labels_cp.shape}")
        logging.info(f"Length of the SHAP names: {shap_names_cp.shape}")

        self.save_shap_values(shap_values_cp, "all")
        self.save_shap_values(shap_labels_cp, "all", "shap_values_labels")
        self.save_shap_values(shap_names_cp, "all", "shap_values_names")

        if method == "shap":
            self.save_shap_values(shap_expected_value_cp, "all", "shap_expected_value")

        torch.cuda.empty_cache()

    def xai_shap_ntu(self):
        """
        Compute SHAP values for the NTU data.
        :return:
        """
        logging.info("Initializing SHAP for the NTU dataset...")
        # initialize the model and load the weights etc.
        classes = [5, 10, 15]
        self.init_student_model()
        logging.info("Initialized model...")

        self.background_set, self.background_labels, self.background_names = self._shap_background()

        self.explainer_deep = shap.DeepExplainer(self.student_model.to(self.gpu_id),
                                                 self.background_set.to(self.gpu_id))
        # tqdm(range(self.val_loader.dataset.num_classes)
        for class_idx in tqdm(classes, desc="Classes", unit="class"):
            logging.info(f"Computing DeepExplainer SHAP values for class {class_idx}...")

            # Get the indices for the current class
            indices_class = self.val_loader.dataset.get_indices(class_idx)
            subset_loader = DataLoader(Subset(self.val_loader.dataset, indices_class),
                                       batch_size=self.args.xai_batch_size, shuffle=False, drop_last=False)
            logging.info("Initialized Subset loader...")

            shap_values_class = []
            # loop over the subset and compute SHAP values
            for batch_idx, (x_batch, y_batch, _) in enumerate(
                    tqdm(subset_loader, desc=f"Class {class_idx} Batches", unit="batch", leave=False)):
                x_batch = x_batch.to(self.gpu_id)

                shap_values_batch = self.explainer_deep.shap_values(x_batch, check_additivity=False)

                # reduce for NTU due to memory
                shap_values_batch = np.array(shap_values_batch, dtype=np.float32)
                # logging.info("Size of SHAP values in batch: {}".format(shap_values_batch.shape))
                shap_values_batch = shap_values_batch[:, :, :, :150, :, 0:1, :]

                # mean_arrays = []
                # Compute the mean for each part
                # for part_indices in self.body_parts:
                #     mean_part = np.mean(shap_values_batch[:, :, :, :, part_indices, :, :], axis=4)
                #     mean_arrays.append(mean_part)

                # shap_values_batch = np.stack(mean_arrays, axis=4)
                # cast to float32 and squeeze second skeleton
                shap_values_batch = np.squeeze(shap_values_batch, axis=5)
                shap_values_class.append(shap_values_batch)
                del shap_values_batch

            shap_values_class = np.concatenate(shap_values_class, axis=0)

            # shap_values_class_mean = np.mean(shap_values_class, axis=0)
            # shap_values_class_std = np.std(shap_values_class, axis=0)

            logging.info(f"The size of shap_values_class {class_idx}: {shap_values_class.shape}")

            self.save_shap_values(shap_values_class, class_idx)
            torch.cuda.empty_cache()
            del shap_values_class

    def save_shap_values(self, shap_values, class_idx, prefix="shap_values_class"):
        """
        Save the SHAP values.
        :param shap_values:
        :param class_idx:
        :param prefix:
        :return:
        """
        save_file = os.path.join(self.save_dir, f'{prefix}_{class_idx}_{self.shap_save_string}')
        np.save(save_file, shap_values)
        logging.info(f"{prefix} {class_idx} saved to: {save_file}")

    def visualize_shap_ntu(self):
        logging.info("Initializing SHAP visualizations for NTU...")
        paths = [
            "logs/xai/three_classes_xview60/2024-10-22_10-33-43",
            "logs/xai/three_classes_xview60/2024-10-22_10-37-59",
            "logs/xai/three_classes_xview60/2024-10-22_10-39-49",
            "logs/xai/three_classes_xview60/2024-10-22_10-48-22",
            "logs/xai/three_classes_xview60/2024-10-22_10-50-21",
        ]
        shap_visualizer = ShapVisualizer(self.val_loader, self.save_dir)
        shap_visualizer.visualize_ntu_shap(paths)

    def visualize_shap_cp(self):
        """
        Entry point for SHAP visualizations...
        """
        logging.info("Initializing SHAP visualizations...")
        paths = [
            "logs/xai/exp_new_model_cp29/2024-10-02_14-09-53",
            "logs/xai/exp_new_model_cp29/2024-10-07_12-03-36",
            "logs/xai/exp_new_model_cp29/2024-10-07_12-11-08",
            "logs/xai/exp_new_model_cp29/2024-10-07_14-39-46",
            "logs/xai/exp_new_model_cp29/2024-10-07_15-36-50",
            "logs/xai/exp_new_model_cp29/2024-10-08_14-13-09"
        ]
        shap_visualizer = ShapVisualizer(self.val_loader, self.save_dir)
        shap_visualizer.visualize_cp_shap(paths)

    def _shap_background(self) -> tuple[torch.Tensor, torch.Tensor, np.array]:
        """
        Generate background dataset with the training set if not yet generated -
        otherwise load it from given path in config.
        :return:
        """
        # check if background data is already loaded
        if self.background_path and os.path.exists(self.background_path):
            logging.info("Loading background data from .npy file: {}".format(self.background_path))
            try:
                background_data = np.load(self.background_path)

                labels_path = self.background_path.replace('.npy', '_labels.npy')
                background_labels = np.load(labels_path)
                logging.info("Loaded background labels with {} samples.".format(background_labels.shape[0]))

                names_path = self.background_path.replace('.npy', '_names.npy')
                background_names = np.load(names_path, allow_pickle=True)
                logging.info("Loaded background names with {} samples.".format(len(background_names)))

                # Check if background data size matches expected size
                if background_data.shape[0] != self.num_background:
                    logging.info(f"Loaded background data has different size {background_data.shape[0]} than config "
                                 f"{self.num_background}. Reloading background data...")
                else:
                    logging.info("Loaded background data with {} samples.".format(background_data.shape[0]))
                    # Convert data to torch tensors
                    background_data = torch.from_numpy(background_data)
                    background_labels = torch.from_numpy(background_labels) if background_labels is not None else None
                    # Return loaded data
                    return background_data, background_labels, background_names
            except (IOError, ValueError) as e:
                logging.warning("Failed to load background data from .npy file: {}".format(str(e)))

        logging.info("Generating background dataset with {} samples...".format(self.num_background))
        np.random.seed(int(self.xai_back_seed))
        torch.manual_seed(int(self.xai_back_seed))
        torch.cuda.manual_seed(int(self.xai_back_seed))
        logging.info(f"Numpy and PyTorch random seed is set to: {self.xai_back_seed}")
        if self.xai_random_back:
            logging.info("Generating random background dataset with possibly all classes...")
            shap_sampler = RandomSampler(self.train_loader, num_samples=self.num_background)
        else:
            #assert self.args.dataset in ["xsub60", "xview60"]
            logging.info("Generating background dataset with class idx {}...".format(self.xai_back_idx))
            indices = self.feeder_train.get_indices(self.xai_back_idx)
            shap_sampler = ShapSampler(indices, n_samples=self.num_background)

        background_loader = DataLoader(self.feeder_train, batch_size=self.args.batch_size,
                                       num_workers=self.args.num_workers, pin_memory=self.args.pin_memory,
                                       shuffle=False, drop_last=False, sampler=shap_sampler)

        background_data = []
        background_labels = []
        background_names = []
        with torch.no_grad():
            for num, (x, y, name) in enumerate(background_loader):
                x = x.to(self.gpu_id)
                y = y.to(self.gpu_id)
                background_data.append(x)
                background_labels.append(y)
                background_names.append(name)

            background_data = torch.cat(background_data, dim=0)
            background_labels = torch.cat(background_labels, dim=0)

        logging.info("Done generating background dataset!")
        logging.info(f"Background Labels: {background_labels}")
        logging.info(f"Background Names: {background_names}")

        try:
            np.save(self.background_path, background_data.cpu())
            logging.info("Background data saved to .npy file: {}".format(self.background_path))

            # Save background labels
            labels_path = self.background_path.replace('.npy', '_labels.npy')
            np.save(labels_path, background_labels.cpu())
            logging.info("Background labels saved to .npy file: {}".format(labels_path))

            # Save background names
            names_path = self.background_path.replace('.npy', '_names.npy')
            np.save(names_path, np.array(background_names))
            logging.info("Background names saved to .npy file: {}".format(names_path))

        except Exception as e:
            logging.warning("Failed to save background data, labels, or names to .npy files: {}".format(str(e)))

        return background_data, background_labels, background_names

    def __build_student(self, checkpoint, student_id=0) -> None:
        """
        Build the student for the xai method.
        May have to retrain the model due to old bug...
        :param checkpoint:
        :param student_id:
        :return:
        """
        try:
            # check if hyperparameters are stored in model otherwise get them from the config file
            actions_hyper = checkpoint['actions_hyper']
            assert len(actions_hyper) != 0
        except (AssertionError, KeyError):
            logging.info("Hyperparameter values not provided in checkpoint file!")
            logging.info("Trying to load them from config file...")
            if len(self.args.xai_hyper_param) != 0:
                actions_hyper = list(self.args.xai_hyper_param)
            else:
                logging.info("Hyperparameter values not valid!")
                logging.info("Please provide the hyperparameter values manually in the config file!")
                sys.exit()

        # get indices for student class -> should be a dict from model
        actions_arch_dict = checkpoint['actions']

        self.student_model = Student(actions_arch_dict, self.arch_choices, actions_hyper, self.hyper_choices,
                                     student_id, self.args, **self.kwargs).to(self.gpu_id)

        logging.info("Student AP: {}".format(self.student_model.action_arch_dict))
        logging.info("Student HP: {}".format(self.student_model.actions_hyper_dict))
        # flops, params = thop.profile(deepcopy(self.student_model), inputs=torch.rand([1, 1] + self.data_shape),
        #                             verbose=False)
        # logging.info('Model profile: {:.2f}G FLOPs and {:.2f}M Parameters'.format(flops / 1e9, params / 1e6))

        # update hyperparameters for sampled student
        optimizer_fn = None
        for optimizer_class in self.optim_list:
            if optimizer_class.__name__.lower() == self.student_model.hyper_info['optimizers'].lower():
                optimizer_fn = optimizer_class
                break

        assert optimizer_fn is not None

        # fixed optimizer args from config
        optimizer_args = self.args.optimizer_args[self.student_model.hyper_info['optimizers']]
        # sampled optimizer args
        optimizer_args['lr'] = self.student_model.actions_hyper_dict['lr']
        optimizer_args['weight_decay'] = self.student_model.hyper_info['weight_decay']

        if optimizer_fn.__name__.lower() not in ['adam', 'adamw']:
            optimizer_args['momentum'] = self.student_model.hyper_info['momentum']

        self.new_batch_size = int(self.student_model.hyper_info['batch_size'])
        self.update_batch_size(self.new_batch_size)

        self.optimizer = optimizer_fn(params=self.student_model.parameters(), **optimizer_args)

        if self.args.lr_scheduler:
            # self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)
            self.scheduler_warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                                      start_factor=self.args.sched_args.start_factor,
                                                                      total_iters=self.args.sched_args.warm_up)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=self.args.sched_args.step_lr,
                                                                  gamma=self.args.sched_args.gamma)
        else:
            self.scheduler = None
        logging.info('Learning rate scheduler: {}'.format(self.scheduler.__class__.__name__))
        logging.info("Done with initializing student skeleton...")

    def init_student_model(self):
        """
        Initialize the student model.
        :return:
        """
        checkpoint = self._load_weights()
        self.__build_student(checkpoint, 0)
        self.assign_weights_opt_lrs(checkpoint)
        model_weights = collections.OrderedDict()
        # Copy contents of sub_dicts into model_weights
        model_weights.update(checkpoint['model']['input_stream'])
        model_weights.update(checkpoint['model']['main_stream'])
        model_weights.update(checkpoint['model']['classifier'])
        self.student_model.load_state_dict(model_weights)
        logging.info("Loaded weights to model")
        self.assign_weights_model(model_weights)

    def assign_weights_model(self, model_weights) -> None:
        """
        Assign weights to the student model and put to GPU
        """
        self.student_model.load_state_dict(model_weights)
        self.student_model.to(self.gpu_id)
        # logging.info("Successfully loaded the state dict!")

    def assign_weights_opt_lrs(self, checkpoint):
        """
        Assign weights to the optimizer and rest...
        """
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epochs = checkpoint['epoch']
            self.best_state = checkpoint['best_state']
            logging.info("Successfully loaded optimizer and scheduler.")
            logging.info('Best Acc Top1: {:.2%}'.format(self.best_state['acc']))
            if self.cp:
                logging.info("Best AUC: {:.2}".format(self.best_state['auc']))

        except (RuntimeError or ValueError):
            logging.info("Error while loading state dict for student")
            logging.info("Starting the retrain...")
            self.retrain_student()

    def _load_weights(self):
        """
        Load weights into the model from the specified file.
        """
        try:
            # load trained student or argmax architecture and check if all streams are saved
            checkpoint = torch.load(self.model_path, map_location=self.gpu_id)
            model_state = ['input_stream', 'main_stream', 'classifier']
            # check if all streams are stored in the
            if not all(state_name in checkpoint['model'] for state_name in model_state):
                logging.info("Loaded checkpoint but not usable with this version!")
                logging.info("Please retrain the model!")
                sys.exit()
            return checkpoint
        except FileNotFoundError:
            logging.error("File not found at: {}".format(self.model_path))
            logging.info("Please check the path an try again!")
            sys.exit()

    def infer_student(self, student_id=0) -> dict:
        # initialize trainer
        self.trainer = Trainer(self.args, self.student_model, self.train_loader, self.val_loader,
                               self.loss_func, self.optimizer, self.scheduler, self.scheduler_warmup,
                               self.writer, self.gpu_id, self.num_classes, self.no_progress_bar,
                               self.new_batch_size)
        epoch = 0
        if self.cp_used:
            best_state_argmax = {'acc': 0, 'auc': 0, 'cm': 0, 'precision': 0, 'recall': 0, 'F1': 0, 'sens': 0,
                                 'spec': 0, 'ppv': 0, 'npv': 0, 'b_acc': 0}
        else:
            best_state_argmax = {'acc': 0, 'acc5': 0, 'auc': 0, 'cm': 0}

        start_infer_time = time()
        eval_stud = self.trainer.trainer_eval(0, student_id, self.cp)
        infer_time = time() - start_infer_time

        # save model for later
        best_state_argmax.update(eval_stud)
        utils.save_checkpoint(self.student_model, self.optimizer.state_dict(), self.scheduler.state_dict(),
                              epoch + 1, infer_time, self.student_model.arch_info, self.student_model.hyper_info,
                              best_state_argmax, True, self.save_dir, student_id, argmax=True)
        return best_state_argmax

    def retrain_student(self) -> None:
        """
        Retrain the student model.
        """
        logging.info("Retraining student...")
        argmax_time = 0
        argmax_id = 9999
        metric = 'auc'

        self.trainer = Trainer(self.args, self.student_model, self.train_loader, self.val_loader,
                               self.loss_func, self.optimizer, self.scheduler, self.scheduler_warmup,
                               self.writer, self.gpu_id, self.num_classes, self.no_progress_bar,
                               self.new_batch_size)

        if self.cp:
            best_state_argmax = {'acc': 0, 'auc': 0, 'cm': 0, 'precision': 0, 'recall': 0, 'F1': 0, 'sens': 0,
                                 'spec': 0, 'ppv': 0, 'npv': 0, 'b_acc': 0}
        else:
            best_state_argmax = {'acc': 0, 'acc5': 0, 'auc': 0, 'cm': 0}

        logging.info("Training ARGMAX student {} for {} epochs".format(argmax_id, self.argmax_epochs))
        try:
            for epoch in range(self.argmax_epochs):
                epoch_time, train_acc, train_loss = self.trainer.trainer_train(epoch, self.argmax_epochs,
                                                                               argmax_id)
                argmax_time += epoch_time
                is_best = False
                if (epoch + 1) % self.eval_interval == 0 or epoch >= self.argmax_epochs - 15:
                    # and (self.gpu_id == 0 if self.args.ddp else True):
                    # (self.gpu_id == 0 if self.args.ddp):
                    logging.info("Evaluating ARGMAX student quality in epoch {}/{}"
                                 .format(epoch + 1, self.argmax_epochs))
                    current_state_argmax = self.trainer.trainer_eval(epoch, argmax_id, self.cp)

                    # optimize towards AUC or Accuracy
                    is_best = current_state_argmax[metric] > best_state_argmax[metric]
                    if is_best:
                        best_state_argmax.update(current_state_argmax)

                # save model for later
                utils.save_checkpoint(self.student_model, self.optimizer.state_dict(),
                                      self.scheduler.state_dict(), epoch + 1, epoch_time,
                                      self.student_model.arch_info, self.student_model.hyper_info,
                                      best_state_argmax, is_best, self.save_dir, argmax_id,
                                      argmax=True)

        except torch.cuda.OutOfMemoryError:
            logging.info("ARGMAX student {} does not fit on GPU!".format(argmax_id))
            torch.cuda.empty_cache()

        logging.info("Done with retraining...")
