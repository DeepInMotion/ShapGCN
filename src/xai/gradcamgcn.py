import collections
import copy
import pickle
import sys
import os

import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from src.initializer import Initializer
from src.model.trainer import Trainer
from src.model.student import Student
from src.utils import utils
from src.xai.perturber import Perturber
from src.xai.gradcam_vis import GradCamVisualizer
from src.xai.xai_utils import load_gradcam_files, load_gradcam_files_cp, NTU_JOINT_NAMES, CP_JOINT_NAMES_29



class GradCamGCN(Initializer):
    def __init__(self, args):
        super().__init__(args)
        self.best_state = None
        self.epochs = None
        self.args = args
        self.trainer = None
        self.student_model = None
        self.argmax_epochs = self.args.argmax_epochs

        # XAI configs
        self.model_path = os.path.join(self.args.work_dir, self.args.xai_model_path)
        logging.info(f"Loading XAI model from: {self.model_path}")

        cp_suffix = f'{self.args.dataset}' if self.args.dataset in ['cp19', 'cp29'] else 'ntu'
        self.cp = self.args.dataset in ['cp19', 'cp29']
        self.gradcam_save_string = f'{cp_suffix}_seed_{self.args.seed}.npy'

        # body parts
        self.body_parts = self.val_loader.dataset.graph.parts
        self.cp_joint_names_29 = CP_JOINT_NAMES_29
        self.ntu_joint_names = NTU_JOINT_NAMES

        # perturber
        self.cp_target_class = 0
        self.perturber = None
        self.individual_joint = True
        self.rand_per = False
        self.config_perturb_cp = {
            'thresh': 0.9,
            'dimension': 1,
            'percent': True,
            'zero_out': True,
            'change_towards_zero': False,
            'switch_sign': True,
            'top_joints': 10
        }

        self.config_perturb_ntu = {
            'thresh': [0.35], #np.arange(0.35, 0.40, 0.05),
            'dimension': 1,
            'percent': True,
            'zero_out': True,
            'change_towards_zero': False,
            'switch_sign': True,
            'top_joints': 10
        }

        # GradCAM stuff
        self.target_layer = None
        self.gradients = None
        self.features = None
        self.negative_gradients = False

        # paths for computed ntu gradcam values
        # self.path_ntu = "logs/xai/gradcam_xview60/2024-11-20_11-21-16" # att bn
        self.path_ntu = "logs/xai/gradcam_xview60/2024-11-20_11-33-12" # tcn conv
        # self.path_ntu = "logs/xai/gradcam_xview60/2024-11-20_12-55-44"   # scn conv
        # self.path_ntu = "logs/xai/gradcam_xview60/2024-12-06_09-17-03" # attention with fixed features and normalization

        # ----- ntu negated gradients -----
        # self.path_ntu = "logs/xai/gradcam_xview60/2024-11-28_09-53-25" # tcn conv negated
        # self.path_ntu = "logs/xai/gradcam_xview60/2024-11-28_10-05-19" # scn conv negated
        # self.path_ntu = "logs/xai/gradcam_xview60/2024-12-03_15-31-56"  # att bn negated

        # self.path_ntu = "logs/xai/gradcam_xview60/2024-12-05_18-58-25" # tcn att with features
        # other computation
        # self.path_ntu = "logs/xai/gradcam_xview60/2024-11-29_13-09-27" # scn conv other

        # ----- CP paths ------
        # path for computed cp values
        # self.path_cp = "logs/xai/gradcam_cp29/2024-11-25_09-04-47" # -> tcn conv
        # self.path_cp = "logs/xai/gradcam_cp29/2024-12-06_11-50-11" # -> main att act
        # self.path_cp = "logs/xai/gradcam_cp29/2024-12-13_12-33-44" # -> main att new
        self.path_cp = "logs/xai/gradcam_cp29/2024-12-15_10-33-49" # -> tcn new

    def start(self):
        """
        Entry point for GradCAMGCN.
        :return:
        """
        torch.autograd.set_detect_anomaly(True)
        if self.cp:
            xai_methods_cp = {
                "gradcam": self.xai_gradcam_cp,
                "cam": self.xai_cam,
                "perturb": self.perturb_gradcam_cp,
                "vis": self.visualize_gradcam_cp
            }
            method = xai_methods_cp.get(self.args.xai_method)
        elif self.args.dataset in ["xsub60", "xview60", "xsub120", "xsetup120"]:
            xai_methods_ntu = {
                "gradcam": self.xai_gradcam_ntu,
                "cam": self.xai_cam,
                "perturb": self.perturb_gradcam_ntu,
                "vis": self.visualize_gradcam_ntu
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

    def perturb_gradcam_ntu(self):
        """
        Perturb gradcam values and infer student model on perturbed architecture for ntu.
        :return:
        """
        logging.info(f"Random perturbation -> {self.rand_per}")
        logging.info(f"Perturbation on individual joints -> {self.individual_joint}")

        gradcam_values_ntu, class_numbers, layer_types = load_gradcam_files(self.path_ntu)
        logging.info(f"GradCAM was computed at layer -> {layer_types}")

        # init perturber
        self.perturber = Perturber(body_parts=self.val_loader.dataset.graph.parts, cp=False,
                                   individual_joints=self.individual_joint)

        # init model
        weights_original = self._load_weights()
        self.__build_student(weights_original, 0)
        self.assign_weights_opt_lrs(weights_original)
        # init trainer once and only replace the student
        self.trainer = Trainer(self.args, self.student_model, self.train_loader, self.val_loader,
                               self.loss_func, self.optimizer, self.scheduler, self.scheduler_warmup,
                               self.writer, self.gpu_id, self.num_classes, self.no_progress_bar,
                               self.new_batch_size)

        # loop over classes, sort values and do perturbation
        for idx, gradcam_data in enumerate(gradcam_values_ntu):
            # get idx, layer etc.
            target_class = class_numbers[idx]
            layer_type = layer_types[idx]
            indices_class = self.val_loader.dataset.get_indices(target_class)

            logging.info(f"Perturbing GradCAMS for {target_class} -> {layer_type}")
            # sort the ntu values
            sorted_gradcams = self.compute_gradcam_order_ntu(gradcam_data, indices_class)

            for thresh in self.config_perturb_ntu['thresh']:
                for top_joints in range(1, self.config_perturb_ntu['top_joints'] + 1):
                    temp_config = copy.deepcopy(self.config_perturb_ntu)
                    temp_config['zero_out'] = False
                    temp_config['thresh'] = thresh
                    temp_config['top_joints'] = top_joints
                    logging.info(f"Experiment config is: {temp_config}")

                    predictions_by_index = self.predict_perturbed_gradcam(temp_config,
                                                                          self.individual_joint,
                                                                          sorted_gradcams,
                                                                          copy.deepcopy(weights_original),
                                                                          rand_per=self.rand_per)

                    filename = (
                        f"pred_gradcam_ntu_class_{target_class}_layer_{layer_type}_"
                        f"t{int(temp_config['thresh'] * 100)}_"
                        f"z{int(temp_config['zero_out'])}_"
                        f"c{int(temp_config['change_towards_zero'])}_"
                        f"d{temp_config['dimension']}_"
                        f"p{int(temp_config['percent'])}_"
                        f"s{int(temp_config['switch_sign'])}_"
                        f"tj{int(temp_config['top_joints'])}.pkl"
                    )

                    with open(os.path.join(self.save_dir, filename), 'wb') as f:
                        logging.info(f"Saving predictions...")
                        pickle.dump(predictions_by_index, f)

            logging.info("Done with perturbation experiment...")

    def perturb_gradcam_cp(self):
        """
        Perturb gradcam values and infer student model on perturbed architecture for CP.
        :return:
        """
        logging.info(f"Random perturbation -> {self.rand_per}")
        logging.info(f"Perturbation on individual joints -> {self.individual_joint}")
        logging.info(f"Target class is -> {self.cp_target_class}")
        gradcam_values_cp, class_numbers, layer_types = load_gradcam_files_cp(self.path_cp)

        logging.info(f"GradCAM was computed at layer -> {layer_types}")

        # init perturber
        self.perturber = Perturber(body_parts=self.val_loader.dataset.graph.parts, cp=True,
                                   individual_joints=self.individual_joint)

        # init model
        weights_original = self._load_weights()
        self.__build_student(weights_original, 0)
        self.assign_weights_opt_lrs(weights_original)
        # init trainer once and only replace the student
        self.trainer = Trainer(self.args, self.student_model, self.train_loader, self.val_loader,
                               self.loss_func, self.optimizer, self.scheduler, self.scheduler_warmup,
                               self.writer, self.gpu_id, self.num_classes, self.no_progress_bar,
                               self.new_batch_size)

        # get indices for both classes
        indices_class_0 = self.val_loader.dataset.get_indices(0)
        indices_class_1 = self.val_loader.dataset.get_indices(1)

        # loop over classes, sort values and do perturbation
        for idx, gradcam_data in enumerate(gradcam_values_cp):
            # get idx, layer etc.
            layer_type = layer_types[idx]
            logging.info(f"Perturbing GradCAMS for {self.cp_target_class} -> {layer_type}")
            # sort the ntu values
            # --------
            # gradcam_data = np.abs(gradcam_data)
            sorted_gradcams = self.compute_gradcam_order_cp(gradcam_data, indices_class_0, indices_class_1)

            # only one thresh for CP -> 0.9
            thresh_values = [None]
            for thresh in thresh_values:
                for top_joints in range(1, self.config_perturb_cp['top_joints'] + 1):
                    temp_config = copy.deepcopy(self.config_perturb_cp)
                    temp_config['zero_out'] = False
                    temp_config['top_joints'] = top_joints
                    logging.info(f"Experiment config is: {temp_config}")

                    predictions_by_index = self.predict_perturbed_gradcam(temp_config,
                                                                          self.individual_joint,
                                                                          sorted_gradcams,
                                                                          copy.deepcopy(weights_original),
                                                                          rand_per=self.rand_per)

                    filename = (
                        f"pred_gradcam_cp_class_{self.cp_target_class}_layer_{layer_type}_"
                        f"t{int(temp_config['thresh'] * 100)}_"
                        f"z{int(temp_config['zero_out'])}_"
                        f"c{int(temp_config['change_towards_zero'])}_"
                        f"d{temp_config['dimension']}_"
                        f"p{int(temp_config['percent'])}_"
                        f"s{int(temp_config['switch_sign'])}_"
                        f"tj{int(temp_config['top_joints'])}.pkl"
                    )

                    with open(os.path.join(self.save_dir, filename), 'wb') as f:
                        logging.info(f"Saving predictions...")
                        pickle.dump(predictions_by_index, f)

            logging.info("Done with perturbation experiment...")

    def predict_perturbed_gradcam(self, config_perturb, individual_joint, sorted_gradcams, weights_original,
                                  rand_per=False):
        """
        Perturb gradcam values and infer student model on most influential gradcam body key points.
        :param config_perturb:
        :param individual_joint:
        :param sorted_gradcams:
        :param weights_original:
        :param rand_per:
        :return:
        """
        if config_perturb['top_joints']:
            assert individual_joint is True and config_perturb['top_joints'] > 0

        predictions_by_index = collections.OrderedDict()
        self.student_model.eval()

        for index, gradcam_importance in tqdm(enumerate(sorted_gradcams), total=len(sorted_gradcams),
                                           desc="GradCAM perturbation"):
            data, label, name = self.val_loader.dataset[index]
            data = np.expand_dims(data, axis=0)
            data = torch.from_numpy(data).to(self.gpu_id)
            predictions_by_index[index] = {
                'label': label,
                'video_id': name,
                'preds': [],
                'gradcams': gradcam_importance
            }

            # just get the prediction
            if config_perturb['top_joints'] == 0:
                with torch.no_grad():
                    out = self.student_model(data)
                if not self.cp:
                    reco_top1 = out.max(1)[1]
                    predictions_by_index[index]['preds'].append(reco_top1.cpu().numpy())
                else:
                    predictions_by_index[index]['preds'].append(out.data.cpu().numpy())
                continue

            if not self.cp:
                # get index and gradcams based on the predicted class
                index = gradcam_importance[1]
                gradcam_importance = gradcam_importance[label]

            # Copy contents of sub_dicts into model_weights
            weights_dummy = collections.OrderedDict()
            weights_dummy.update(copy.deepcopy(weights_original['model']['input_stream']))
            weights_dummy.update(copy.deepcopy(weights_original['model']['main_stream']))
            weights_dummy.update(copy.deepcopy(weights_original['model']['classifier']))

            # Group joints for perturbation
            if config_perturb["top_joints"]:
                if rand_per:
                    joints = list(self.perturber.parts_mapping.keys())
                    positive_joints = np.random.choice(joints, config_perturb['top_joints'], replace=False).tolist()
                    # negative_joints = np.random.choice(joints, config_perturb['top_joints'], replace=False).tolist()
                else:
                    # Group joints by positive gradcam importance
                    positive_joints = [joint for joint in gradcam_importance if joint[1] > 0]
                    positive_joints = positive_joints[:config_perturb['top_joints']]
                    # negative_joints = [joint for joint in gradcam_importance if joint[1] < 0]
                    # negative_joints = negative_joints[-config_perturb['top_joints']:]
                joint_groups = [positive_joints]
            else:
                # process all joints
                joint_groups = gradcam_importance

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

    def xai_gradcam_ntu(self):
        """
        Calculate the GradCAM values for the NTU dataset
        :return:
        """

        classes = [5, 10, 15]
        target_strs = ["main_att_act", "main_tcn_conv", "main_block0_tcn_conv", "in_j_att_act", "in_v_att_act",
                       "in_b_att_act", "in_a_att_act", "in_j_tcn_p_conv", "in_v_tcn_p_conv", "in_b_tcn_p_conv",
                       "in_a_tcn_p_conv", "in_j_init_bn", "in_v_init_bn", "in_b_init_bn", "in_a_init_bn"]
        raw_gradcam = True #False

        for target_str in target_strs:
            
            self.init_student_model()
            
            target_layers = {
                "in_j_init_bn": self.student_model.input_stream[0].init_bn,
                "in_v_init_bn": self.student_model.input_stream[1].init_bn,
                "in_b_init_bn": self.student_model.input_stream[2].init_bn,
                "in_a_init_bn": self.student_model.input_stream[3].init_bn,
                "in_j_tcn_p_conv": self.student_model.input_stream[0][-2].point_conv_expand[0],
                "in_v_tcn_p_conv": self.student_model.input_stream[1][-2].point_conv_expand[0],
                "in_b_tcn_p_conv": self.student_model.input_stream[2][-2].point_conv_expand[0],
                "in_a_tcn_p_conv": self.student_model.input_stream[3][-2].point_conv_expand[0],
                "in_j_att_act": self.student_model.input_stream[0][-1].activation,
                "in_v_att_act": self.student_model.input_stream[1][-1].activation,
                "in_b_att_act": self.student_model.input_stream[2][-1].activation,
                "in_a_att_act": self.student_model.input_stream[3][-1].activation,
                "main_block0_tcn_conv": self.student_model.main_stream[1].conv,
                "main_tcn_conv": self.student_model.main_stream[-2].conv,
                "main_att_act": self.student_model.main_stream[-1].activation
            }
            
            self.target_layer = target_layers[target_str]
            
            logging.info(f"Getting gradients at layer: {self.target_layer}...")
            logging.info(f"Getting negative gradients: {self.negative_gradients}")
            # catch the gradients
            self.update_batch_size(1)
            self._register_hooks()
            logging.info(f"Computing GradCAM values for NTU dataset...")

            for class_idx in tqdm(classes, desc="Classes", unit="class"):
                logging.info(f"Computing GradCAM values for class {class_idx+1}...")

                # Get the indices for the current class
                indices_class = self.val_loader.dataset.get_indices(class_idx)
                subset_loader = DataLoader(Subset(self.val_loader.dataset, indices_class), batch_size=1, shuffle=False,
                                           drop_last=False)

                logging.info("Initialized Subset loader...")

                grad_values_class = []
                # loop over the subset and compute SHAP values
                for batch_idx, (x, y, _) in enumerate(
                        tqdm(subset_loader, desc=f"Class {class_idx} Batches", unit="batch", leave=False)):
                    with torch.enable_grad():
                        # gradcam_values_time = []
                        # for t in range(150):
                            # Slice the input at the specific time index (e.g., slice window of [t, t+1])
                            # x_time = x[:, :, :, t:t + 1, :, :]
                        x_batch = x.to(self.gpu_id)
                        out = self.student_model(x_batch)

                        # Zero gradients and compute the output for the target class
                        self.student_model.zero_grad()
                        target_output = out[:, class_idx]
                        target_output.backward(retain_graph=True)

                        # Only first skeleton
                        # (M, C, T, V)
                        self.gradients = self.gradients[:1]  # Shape: (1, 48, 75, 25)
                        if target_str == "main_att_act":
                            # get features directly from student model due to bug
                            self.features = self.student_model.feat_main[:1]
                        elif target_str == "in_j_att_act":
                            self.features = self.student_model.feat_j[:1]
                        elif target_str == "in_v_att_act":
                            self.features = self.student_model.feat_v[:1]
                        elif target_str == "in_b_att_act":
                            self.features = self.student_model.feat_b[:1]
                        elif target_str == "in_a_att_act":
                            self.features = self.student_model.feat_a[:1]
                        else:
                            # if conv. layer or other
                            self.features = self.features[:1]  # Shape: (1, 48, 75, 25)
                            
                        grad_cam, grad_cam_raw = self._compute_grad()
                
                        # Normalize the final Grad-CAM [0, 1]
                        # grad_cam_keypoints /= np.max(np.abs(grad_cam_keypoints) + 1e-8)

                        if raw_gradcam:
                            grad_values_class.append(grad_cam_raw.detach().cpu().numpy())
                        else:
                            grad_values_class.append(grad_cam.detach().cpu().numpy())
                            
                grad_values_class = np.array(grad_values_class, dtype=np.float32)
    
                # Save the Grad-CAM values for this sample (for all time slices)
                self.save_gradcam_values(grad_values_class, str(class_idx), target_str)

        torch.cuda.empty_cache()
        logging.info("Done")
            
    def xai_gradcam_cp(self):
        """
        Calculate the GradCAM values for the CP dataset.
        :return:
        """

        target_strs = ["main_att_act", "main_tcn_conv", "main_block0_tcn_conv", "in_j_att_act", "in_v_att_act",
                       "in_b_att_act", "in_a_att_act", "in_j_tcn_p_conv", "in_v_tcn_p_conv", "in_b_tcn_p_conv",
                       "in_a_tcn_p_conv", "in_j_init_bn", "in_v_init_bn", "in_b_init_bn", "in_a_init_bn"]

        raw_gradcam = False
        target_strs = ["main_tcn_conv"]
        for target_str in target_strs:
            
            self.init_student_model()
            
            target_layers = {
                "in_j_init_bn": self.student_model.input_stream[0].init_bn,
                "in_v_init_bn": self.student_model.input_stream[1].init_bn,
                "in_b_init_bn": self.student_model.input_stream[2].init_bn,
                "in_a_init_bn": self.student_model.input_stream[3].init_bn,
                "in_j_tcn_p_conv": self.student_model.input_stream[0][-2].point_conv[0],
                "in_v_tcn_p_conv": self.student_model.input_stream[1][-2].point_conv[0],
                "in_b_tcn_p_conv": self.student_model.input_stream[2][-2].point_conv[0],
                "in_a_tcn_p_conv": self.student_model.input_stream[3][-2].point_conv[0],
                "in_j_att_act": self.student_model.input_stream[0][-1].activation,
                "in_v_att_act": self.student_model.input_stream[1][-1].activation,
                "in_b_att_act": self.student_model.input_stream[2][-1].activation,
                "in_a_att_act": self.student_model.input_stream[3][-1].activation,
                "main_block0_tcn_conv": self.student_model.main_stream[1].conv,
                "main_tcn_conv": self.student_model.main_stream[-2].conv,
                "main_att_act": self.student_model.main_stream[-1].activation
            }
            
            self.target_layer = target_layers[target_str]
            
            logging.info(f"Getting gradients at layer: {self.target_layer}...")
            logging.info(f"Getting negative gradients: {self.negative_gradients}")
            # catch the gradients
            # target_class = 0 # --> 0==CP
            self.update_batch_size(1)
            self._register_hooks()
            # Register backward hook
            # target_layer.register_backward_hook(lambda module, input, output: save_grad(output[0]))
            all_cams = []
            logging.info(f"Computing GradCAM values for CP dataset...")
            shap_cp_iter = self.val_loader if self.no_progress_bar else (
                tqdm(self.val_loader, leave=True, desc="GradCAM iter"))
            for num, (x, y, name) in enumerate(shap_cp_iter):
                with torch.enable_grad():
                    x_batch = x.to(self.gpu_id)
                    out = self.student_model(x_batch)
                    
                    # get cams for both classes (CP and NO-CP)
                    # here we average over the CAMS
                    class_grad_cams = []
                    for target_class in range(0, 2):
                        self.student_model.zero_grad()
                        target_output = out[:, target_class]
                        target_output.backward(retain_graph=True)

                        self.gradients = self.gradients[:1]  # Shape: (1, 48, 75, 25)
                        if target_str == "main_att_act":
                            # get features directly from student model due to bug
                            self.features = self.student_model.feat_main[:1]
                        elif target_str == "in_j_att_act":
                            self.features = self.student_model.feat_j[:1]
                        elif target_str == "in_v_att_act":
                            self.features = self.student_model.feat_v[:1]
                        elif target_str == "in_b_att_act":
                            self.features = self.student_model.feat_b[:1]
                        elif target_str == "in_a_att_act":
                            self.features = self.student_model.feat_a[:1]
                        else:
                            # if conv. layer or other
                            self.features = self.features[:1]  # Shape: (1, 48, 75, 25)

                        grad_cam, grad_cam_raw = self._compute_grad()

                        if raw_gradcam:
                            class_grad_cams.append(grad_cam_raw.detach().cpu().numpy())
                        else:
                            class_grad_cams.append(grad_cam.detach().cpu().numpy())

                    # Stacking the Grad-CAMs of both classes
                    # grad_cams_combined = np.stack(class_grad_cams, axis=0)
                    # Shape: (48, 25, 29)
                    # grad_cam_nocp = grad_cams_combined[0]
                    # grad_cam_cp = grad_cams_combined[1]

                    # Stacking the Grad-CAMs of both classes
                    grad_cams_combined = np.stack(class_grad_cams, axis=0)
                    # Shape: (48, 25, 29)
                    # grad_cam_nocp = grad_cams_combined[0]
                    # grad_cam_cp = grad_cams_combined[1]
    
                    # grad_cam_nocp_keypoints = torch.median(grad_cam_nocp, dim=1)  # Shape: (48, 29)
                    # grad_cam_cp_keypoints = torch.median(grad_cam_cp, dim=1)  # Shape: (48, 29)
    
                    # Compute the difference between CP and NO-CP for each keypoint
                    # grad_cam_keypoints_diff = torch.sub(grad_cam_cp_keypoints, grad_cam_nocp_keypoints)
    
                    # Shape: (29)
                    # grad_cam_keypoints_diff_final = np.mean(grad_cam_keypoints_diff, axis=0)
                    # Normalize the final values
                    # Normalize to [-1, 1]
                    # grad_cam_keypoints_diff_final /= np.max(np.abs(grad_cam_keypoints_diff_final))
    
                    # Apply ReLU
                    # grad_cam_keypoints_diff_final = np.maximum(grad_cam_keypoints_diff_final, 0)
                    # grad_cam_keypoints_diff_final = grad_cam_keypoints_diff_final ** (1 / 4)
    
                    all_cams.append(grad_cams_combined)
    
            grad_values_cp = np.array(all_cams, dtype=np.float32)
    
            self.save_gradcam_values(grad_values_cp, class_idx="all", layer=target_str)

        torch.cuda.empty_cache()
        logging.info("Done")
        
    def _compute_grad(self):
        """
        Compute GradCAM.
        :return:
        """
        # gradient normalisation
        self.gradients /= (torch.sqrt(torch.mean(self.gradients ** 2)) + 1e-8)
        # global average pooling with counterfactual explanations
        if self.negative_gradients:
            pooled_gradients = torch.mean( - self.gradients, dim=(1, 2))
        else:
            pooled_gradients = torch.mean(self.gradients, dim=(1, 2))
        # weighted sum of feature maps
        grad_cam_raw = torch.sum(pooled_gradients.view(-1, 1, 1, 1) * self.features, dim=0)
        grad_cam = torch.mean(grad_cam_raw, dim=0)  # Shape: (75, 25) in main layer (25, 29) CP in input
        grad_cam = F.relu(grad_cam)
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() + 1e-8)
        return grad_cam, grad_cam_raw

    def xai_cam(self):
        """
        Calculate the CAM values for
        :return:
        """
        raise NotImplemented

    def save_gradcam_values(self, gradcam_values: np.array, class_idx: str, layer: str, prefix="gradcam_values_class") \
            -> None:
        """
        Save the GradCAM values.
        :param gradcam_values:
        :param class_idx:
        :param layer:
        :param prefix:
        :return:
        """
        save_file = os.path.join(self.save_dir, f'{prefix}_{class_idx}_{layer}_{self.gradcam_save_string}')
        np.save(save_file, gradcam_values)
        logging.info(f"{prefix} {class_idx} saved to: {self.save_dir}")

    def visualize_gradcam_ntu(self):
        """
        Visualize GradCAM values for ntu dataset.
        :return:
        """
        gradcam_visualizer = GradCamVisualizer(self.val_loader, self.save_dir)
        gradcam_visualizer.visualize_gradcam_ntu(self.path_ntu)

    def visualize_gradcam_cp(self):
        """
        Visualize GradCAM values for cp dataset.
        :return:
        """
        gradcam_visualizer = GradCamVisualizer(self.val_loader, self.save_dir)
        gradcam_visualizer.visualize_gradcam_cp(self.path_cp)

    def compute_gradcam_order_ntu(self, gradcam_data: np.ndarray, indices_class: list):
        """
        Compute GradCAM ordering of body key points for ntu dataset.
        This is called from the perturbation experiment.
        :return:
        """
        sorted_mean_gradcam_ntu = {}

        num_instances = gradcam_data.shape[0]  # 315 instances (idx axis)
        num_joints = gradcam_data.shape[2]  # 25 joints (v axis)

        for instance_idx in range(num_instances):
            mean_gradcam_values = []

            # For each joint, calculate the mean value across all time steps (axis 1)
            for joint_idx in range(num_joints):
                mean_value_for_joint = np.mean(
                    gradcam_data[instance_idx, :, joint_idx])  # Mean over time steps for this joint
                mean_gradcam_values.append((self.ntu_joint_names[joint_idx], mean_value_for_joint))

            # Sort the joints based on their mean Grad-CAM values for this instance
            sorted_mean_gradcam_values_for_instance = sorted(mean_gradcam_values, key=lambda x: x[1], reverse=True)
            sorted_mean_gradcam_ntu[instance_idx] = sorted_mean_gradcam_values_for_instance

        cumulative_gradcam_by_joint_ntu = collections.defaultdict(lambda: collections.defaultdict(float))

        # Accumulate the Grad-CAM values for each joint and index
        for instance_idx, mean_gradcam_values in sorted_mean_gradcam_ntu.items():
            for joint_name, mean_value in mean_gradcam_values:
                cumulative_gradcam_by_joint_ntu[instance_idx][joint_name] += mean_value

        # Sort the joints by their cumulative Grad-CAM values for each index
        sorted_joints_by_index_ntu = []
        for instance_idx, joint_gradcam_values in cumulative_gradcam_by_joint_ntu.items():
            sorted_joints = sorted(joint_gradcam_values.items(), key=lambda x: x[1], reverse=True)
            sorted_joints_by_index_ntu.append(sorted_joints)

        # Combine GradCAM values with their original class 0 indices
        sorted_ntu_with_index = list(zip(sorted_joints_by_index_ntu, indices_class))
        return sorted_ntu_with_index

    def compute_gradcam_order_cp(self, gradcam_data: np.ndarray, indices_class_0: list, indices_class_1: list):
        """
        Compute GradCAM ordering of body key points for CP dataset for two classes (class 0 and class 1).
        The GradCAM data has the shape (num_instances, 2, num_time_steps, num_joints).

        :param gradcam_data: GradCAM data, with shape (12099, 2, t, 29)
                             where 12099 is the number of instances form validation set,
                             2 is the number of classes (class 0 and 1),
                             t is the number of time steps, and 29 is the number of joints.
        :param indices_class_0: List of indices corresponding to class 0.
        :param indices_class_1: List of indices corresponding to class 1.

        :return: A list of sorted GradCAM values for both classes based on cumulative mean GradCAM values.
        """
        sorted_mean_gradcam_cp = {
            0: {},  # For class 0
            1: {}  # For class 1
        }

        num_instances = gradcam_data.shape[0]  # 12099 instances
        num_joints = gradcam_data.shape[3]  # 29 joints
        num_classes = gradcam_data.shape[1]  # 2 classes (class 0 and class 1)

        # Iterate over each instance
        for instance_idx in range(num_instances):
            for class_idx in range(num_classes):
                mean_gradcam_values = []

                # For each joint, calculate the mean value across all time steps (axis 2)
                for joint_idx in range(num_joints):
                    mean_value_for_joint = np.mean(gradcam_data[instance_idx, class_idx, :, joint_idx])  # Mean time
                    mean_gradcam_values.append((self.cp_joint_names_29[joint_idx], mean_value_for_joint))

                # Sort the joints based on their mean Grad-CAM values for this instance
                sorted_mean_gradcam_values_for_instance = sorted(mean_gradcam_values, key=lambda x: x[1], reverse=True)
                sorted_mean_gradcam_cp[class_idx][instance_idx] = sorted_mean_gradcam_values_for_instance

        # Accumulate the Grad-CAM values for each joint and class
        cumulative_gradcam_by_joint_cp = {
            0: collections.defaultdict(lambda: collections.defaultdict(float)),  # Class 0
            1: collections.defaultdict(lambda: collections.defaultdict(float))  # Class 1
        }

        for class_idx in range(num_classes):
            for instance_idx, mean_gradcam_values in sorted_mean_gradcam_cp[class_idx].items():
                for joint_name, mean_value in mean_gradcam_values:
                    cumulative_gradcam_by_joint_cp[class_idx][instance_idx][joint_name] += mean_value

        # Sort the joints by their cumulative Grad-CAM values for each index and class
        sorted_joints_by_index_cp = {
            0: [],  # Class 0
            1: []  # Class 1
        }

        for class_idx in range(num_classes):
            for instance_idx, joint_gradcam_values in cumulative_gradcam_by_joint_cp[class_idx].items():
                sorted_joints = sorted(joint_gradcam_values.items(), key=lambda x: x[1], reverse=True)
                sorted_joints_by_index_cp[class_idx].append(sorted_joints)

        # Combine GradCAM values with their original class indices
        sorted_class_0_with_index = list(zip(sorted_joints_by_index_cp[0], indices_class_0))
        sorted_class_1_with_index = list(zip(sorted_joints_by_index_cp[1], indices_class_1))

        # Combine both classes' sorted values and sort by indices
        # TODO save both values or diff???
        # grad_cam_keypoints_diff = grad_cam_keypoints_cp - grad_cam_keypoints_nocp
        combined_gradcam_values_with_index = sorted_class_0_with_index + sorted_class_1_with_index
        sorted_combined_list = sorted(combined_gradcam_values_with_index, key=lambda x: x[1])

        # Return only the sorted GradCAM values
        sorted_gradcams = [value for value, index in sorted_combined_list]

        return sorted_gradcams

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

    def _register_hooks(self):
        # Hook for the forward pass to capture the feature maps
        def forward_hook(module, input, output):
            self.features = output

        # Hook for the backward pass to capture the gradients
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Register hooks to the selected layer
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

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