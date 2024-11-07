import numpy as np
import math
import random


class Augmenter:
    def __init__(self):
        self.chance = random.random()
        self.angle_candidate = [i for i in range(-60, 60 + 1, 10)]
        self.scale_candidate = [i / 10 for i in range(int(0.7 * 10), int(1.3 * 10) + 1)]
        self.translation_candidate = [i / 10 for i in range(int(-0.4 * 10), int(0.4 * 10) + 1)]

    def random_perturbation(self, sample_data):
        sample_data = self._rotate(sample_data, random.choice(self.angle_candidate))
        sample_data = self._scale(sample_data, random.choice(self.scale_candidate))
        sample_data = self._translation(sample_data, random.choice(self.translation_candidate),
                                       random.choice(self.translation_candidate))
        return sample_data

    @staticmethod
    def random_start(sample_data, window_size=150, roll_sequence=False, sequence_part=1, num_sequence_parts=1):
        C, T, V = sample_data.shape
        if not roll_sequence:
            try:
                T = np.where(np.isnan(sample_data[0, :, 0]))[0][0]
            except:
                pass

        # Select sequence starting point
        part_size = math.floor(T / num_sequence_parts)
        window_start_minimum = (sequence_part - 1) * part_size if (sequence_part - 1) * part_size < (
                T - window_size) else T - window_size
        window_start_maximum = sequence_part * part_size if (sequence_part * part_size) < (
                T - window_size) else T - window_size
        window_start = random.randint(window_start_minimum, window_start_maximum)

        return sample_data[:, window_start:window_start + window_size, :]

    @staticmethod
    def random_local_position(global_position, bone_conns):
        local_position = np.zeros(global_position.shape)
        for i in range(len(bone_conns)):
            local_position[:, :, i, :] = global_position[:, :, i, :] - global_position[:, :, bone_conns[i], :]
        return local_position

    @staticmethod
    def random_angular_position(linear_position):
        angular_position = np.zeros(tuple([2] + list(linear_position.shape[1:])))
        rho = np.sqrt(linear_position[0, ...] ** 2 + linear_position[1, ...] ** 2)
        phi = np.arctan2(linear_position[1, ...], linear_position[0, ...])
        angular_position[0, ...] = (rho - 0.5) * 2
        angular_position[1, ...] = phi / math.pi
        return angular_position

    @staticmethod
    def _rotate(sample_data, angle):
        C, T, V = sample_data.shape
        rad = math.radians(-angle)
        a = np.full(shape=T, fill_value=rad)
        theta = np.array([[np.cos(a), -np.sin(a)],
                          [np.sin(a), np.cos(a)]])

        # Rotate joints for each frame
        for i_frame in range(T):
            xy = sample_data[0:2, i_frame, :]
            new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
            sample_data[0:2, i_frame, :] = new_xy.reshape(2, V)
        return sample_data

    @staticmethod
    def _scale(sample_data, scale_factor):
        sample_data[0:2, :, :] = sample_data[0:2, :, :] * scale_factor
        return sample_data

    @staticmethod
    def _translation(sample_data, delta_x, delta_y):
        sample_data[0, :, :] = sample_data[0, :, :] + delta_x
        sample_data[1, :, :] = sample_data[1, :, :] + delta_y
        return sample_data
