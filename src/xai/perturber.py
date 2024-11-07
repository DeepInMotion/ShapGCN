import logging
import numpy as np
import torch


class Perturber:
    def __init__(self, body_parts, edges=None, cp=True, individual_joints=False):
        self.body_parts = body_parts
        if cp and not individual_joints:
            self.parts_mapping = {
                'left arm': np.array([9, 10, 11, 21, 22]),
                'right arm': np.array([5, 6, 7, 19, 20]),
                'left leg': np.array([16, 17, 18, 26, 27, 28]),
                'right leg': np.array([13, 14, 15, 23, 24, 25]),
                'torso': np.array([0, 1, 2, 3, 4, 8, 12])
            }
        if not cp and individual_joints:
            # for ntu data
            self.parts_mapping = {"base_of_the_spine": 0, "middle_of_the_spine": 1, "neck": 2, "head": 3,
                                  "left_shoulder": 4, "left_elbow": 5, "left_wrist": 6, "left_hand": 7,
                                  "right_shoulder": 8, "right_elbow": 9, "right_wrist": 10, "right_hand": 11,
                                  "left_hip": 12, "left_knee": 13, "left_ankle": 14, "left_foot": 15, "right_hip": 16,
                                  "right_knee": 17, "right_ankle": 18, "right_foot": 19, "spine": 20,
                                  "tip_of_the_left_hand": 21, "left_thumb": 22, "tip_of_the_right_hand": 23,
                                  "right_thumb": 24}
        if cp and individual_joints:
            # for CP 29 dataset
            self.parts_mapping = {'head_top': 0, 'nose': 1, 'right_ear': 2, 'left_ear': 3, 'upper_neck': 4,
                                  'right_shoulder': 5, 'right_elbow': 6, 'right_wrist': 7, 'thorax': 8,
                                  'left_shoulder': 9, 'left_elbow': 10, 'left_wrist': 11, 'pelvis': 12,
                                  'right_hip': 13, 'right_knee': 14, 'right_ankle': 15, 'left_hip': 16,
                                  'left_knee': 17, 'left_ankle': 18, 'right_little_finger': 19,
                                  'right_index_finger': 20, 'left_little_finger': 21, 'left_index_finger': 22,
                                  'right_heel': 23, 'right_little_toe': 24, 'right_big_toe': 25, 'left_heel': 26,
                                  'left_little_toe': 27, 'left_big_toe': 28}
        self._edges = edges
        if edges is not None:
            self.edges = edges

        self.edge_mask = None
        self.epsilon = 1e-5
        self.areas = ("input_stream", "main_stream", "stem_scn.conv.edge")

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, value):
        if isinstance(value, int):
            self._edges = [value]
        elif isinstance(value, list):
            if all(isinstance(v, str) for v in value):  # Check if it's a list of strings
                try:
                    # Map each string to its corresponding integer from parts_mapping
                    self._edges = np.array([self.parts_mapping[v] for v in value])
                except KeyError as e:
                    raise ValueError(
                        f"Invalid body part name: {str(e)}. Valid parts are: {list(self.parts_mapping.keys())}")
            else:
                self._edges = value
        elif isinstance(value, np.ndarray):
            self._edges = value
        elif isinstance(value, str):
            if value in self.parts_mapping:
                self._edges = self.parts_mapping[value]
            else:
                raise ValueError(f"Invalid body part name: {value}. Valid parts are: {self.body_parts}")
        else:
            raise ValueError("Edges must be an integer, list of strings, or a list of integers!")


    def perturb_edge_matrix(self, model_weights, config_perturb, log=True):
        """
        Extract tensors from the model's state dictionary that have keys ending with '.edge',
        convert them to numpy arrays, modify the values at specified indices, update the state dictionary,
        and move the updated tensors to the GPU if they were originally on the GPU.

        Parameters:
        checkpoint (dict): The model's state dictionary containing tensor values.
        change_value (float): The value to add to the specified indices in each edge tensor.
        change_sign (bool): If True, change the value to go towards zero at the specified indices - otherwise move the
        value
        dimension (int): The dimension of the edge matrix to perturb.

        Returns:
        dict: The updated state dictionary with modified edge tensors.
        """
        if log:
            logging.info(f"Perturbing edge tensor with: {config_perturb}")

        for key, value in model_weights.items():
            if key.endswith('.edge'):
                original_device = value.device
                # Check if tensor is on GPU
                if value.is_cuda:
                    value = value.cpu()
                edge_array = value.numpy()

                if edge_array.shape[0] <= config_perturb["dimension"]:
                    config_perturb["dimension"] = edge_array.shape[0]

                for edge in self.edges:
                    for dim in range(0, config_perturb["dimension"]):
                        current_value = edge_array[dim, edge, edge]

                        if config_perturb["percent"]:
                            # Calculate the absolute change based on the percentage of the ranges of edge_array
                            range_value = np.abs(np.max(edge_array[dim, :]) - np.min(edge_array[dim, :]))
                            absolute_change = config_perturb["thresh"] * range_value
                        else:
                            absolute_change = config_perturb["thresh"]

                        if config_perturb["change_towards_zero"]:
                            # Adjust values towards 0, ensuring no sign switch if required
                            if current_value > 0:
                                new_value = max(current_value - absolute_change,
                                                self.epsilon if not config_perturb["switch_sign"] else -np.inf)
                            else:
                                new_value = min(current_value + absolute_change,
                                                -self.epsilon if not config_perturb["switch_sign"] else np.inf)
                        else:
                            # Move values further from 0 in either direction
                            new_value = current_value + (absolute_change if current_value > 0 else -absolute_change)

                        if config_perturb["zero_out"]:
                            # put edge tensor to very small value
                            new_value = self.epsilon

                        # Update the array
                        edge_array[dim, edge, edge] = new_value

                # Convert back to tensor
                model_weights[key] = torch.from_numpy(edge_array).to(original_device)

        return model_weights

    def perturb_A(self, combined_state_dict):
        """
        Masking the input features with the mean value
        """
        raise NotImplementedError

    def perturb_noise(self, mean=0, std=0.1):
        """
        Perturbs the edge importance matrix with noise.
        """
        # TODO
        raise NotImplementedError
