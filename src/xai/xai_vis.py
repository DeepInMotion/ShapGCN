import logging
import os.path

import shap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.animation as animation

# plt.switch_backend('TkAgg')
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'

from src.xai.xai_utils import ShapHandler

from shap.plots.colors import red_blue
from shap.plots._utils import convert_color


class XAIVis:
    def __init__(self, val_loader, save_dir):
        self.save_dir = save_dir
        self.vis_handler = ShapHandler()
        self.val_loader = val_loader

        # body parts
        self.colorbar = None
        self.body_parts = self.val_loader.dataset.graph.parts

        self.infant_class_1, self.infant_class_0 = None, None

        self.cp_parts_names = ['left arm', 'right arm', 'left leg', 'right leg', 'torso']

        self.cp_joint_names_29 = ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder',
                                  'right_elbow', 'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist',
                                  'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip', 'left_knee',
                                  'left_ankle', 'right_little_finger', 'right_index_finger', 'left_little_finger',
                                  'left_index_finger', 'right_heel', 'right_little_toe', 'right_big_toe', 'left_heel',
                                  'left_little_toe', 'left_big_toe']

        self.ntu_joint_names = ["base_of_the_spine", "middle_of_the_spine", "neck", "head", "left_shoulder",
                                "left_elbow", "left_wrist", "left_hand", "right_shoulder", "right_elbow", "right_wrist",
                                "right_hand", "left_hip", "left_knee", "left_ankle", "left_foot", "right_hip",
                                "right_knee", "right_ankle", "right_foot", "spine", "tip_of_the_left_hand",
                                "left_thumb", "tip_of_the_right_hand", "right_thumb"]

        self.feature_names_cp_plot = ["$J$ pos. absolute X", "$J$ pos. absolute Y", "$J$ pos. relative X",
                                      "$J$ pos. relative Y", "$V$ short X", "$V$ short Y", "$V$ long X", "$V$ long Y",
                                      "$B$ length X", "$B$ length Y", "$B$ angle X", "$B$ angle Y", "$A$ short X",
                                      "$A$ short Y", "$A$ long X", "$A$ long Y"]

        self.x_features_cp_plot = ["$J$ pos. absolute X", "$J$ pos. relative X", "$V$ short X", "$V$ long X",
                                   "$B$ length X", "$B$ angle X", "$A$ short X", "$A$ long X"]

        self.y_features_cp_plot = ["$J$ pos. absolute Y", "$J$ pos. relative Y", "$V$ short Y", "$V$ long Y",
                                   "$B$ length Y", "$B$ angle Y", "$A$ short Y", "$A$ long Y"]

        self.z_features_cp_plot = ["$J$ pos. absolute Z", "$J$ pos. relative Z", "$V$ short Z", "$V$ long Z",
                                   "$B$ length Z", "$B$ angle Z", "$A$ short Z", "$A$ long Z"]

        self.class_0_indices, self.class_1_indices = None, None

    def visualize_ntu(self, paths) -> None:
        """
        Visualize the NTU dataset.
        :param paths: list of shap values
        :return: None
        """
        target_class = 5
        index_to_vis = 200
        single_skeleton = True
        average_time = False
        average_windows = False
        average_body = False
        logging.info(f"Target class is {target_class}")

        shap_values_ntu = self.vis_handler.load_shap_values_ntu(paths, target_class)

        self.feature_vis_str = "jvba"
        self.n_windows = [1, 10, 20, 30, 40, 50]

        logging.info("Averaging SHAP values...")
        averaged_shap_ntu = self.average_ntu_shap_values(shap_values_ntu,time=average_time, body=average_body,
                                                         windows=average_windows)

        indices_class = self.val_loader.dataset.get_indices(target_class)
        logging.info("Averaging Feature values...")
        averaged_feature_ntu = self.average_ntu_feature_values(indices_class, time=average_time, body=average_body,
                                                               windows=average_windows)

        # visualize this stuff
        features = 25 if average_body is False else 5
        if single_skeleton:
            self.vis_shap_ntu_skeleton(averaged_shap_ntu, averaged_feature_ntu, target_class, index_to_vis)
            plot_1 = self.vis_shap_ntu_part_beeswarm_single(averaged_shap_ntu, averaged_feature_ntu, target_class,
                                                            index_to_vis, num_features=features, display=10)

            save_string = (
                f"shaps_single_class_{target_class + 1}_averaged_t{1 if average_time else 0}_b{1 if average_body else 0}"
                f"_w{1 if average_windows else 0}.pdf")
            plot_1.savefig(os.path.join(self.save_dir, save_string), dpi=300)
        else:
            # plot multiple
            plot_1 = self.vis_shap_ntu_part_beeswarm_multiple(averaged_shap_ntu, averaged_feature_ntu, target_class,
                                                             num_features=features, display=10)

            save_string = (f"shaps_class_{target_class+1}_averaged_t{1 if average_time else 0}_b{1 if average_body else 0}"
                           f"_w{1 if average_windows else 0}.pdf")
            plot_1.savefig(os.path.join(self.save_dir, save_string), dpi=300)
        logging.info("Done with NTU visualization!")

    def visualize_cp(self, paths) -> None:
        """
        Visualize the CP dataset.
        :param paths:
        :return:
        """
        # get the SHAP values from paths
        shap_values_cp, shap_names_cp, shap_labels_cp = self.vis_handler.load_shap_values_cp(paths)

        # class 1 --> CP
        # class 0 --> NO CP
        single_infant = True

        self.feature_vis_str = "v"
        self.n_windows = [1, 10, 20, 30, 40]
        average_time = True
        average_windows = False
        average_body = False

        if single_infant:
            # maybe random here
            class_0_names = shap_names_cp[self.vis_handler.class_0_indices]
            class_1_names = shap_names_cp[self.vis_handler.class_1_indices]
            unique_names_class_0 = np.unique(class_0_names)
            unique_names_class_1 = np.unique(class_1_names)

            # new indices focusing on one infant for each class
            self.infant_class_0 = "146_1_1_E_F"
            self.infant_class_1 = "VelloreTHIN_020_1_1_E"
            logging.info(f"Visualizing one infant - Class 0: {self.infant_class_0}, Class 1: {self.infant_class_1}")
            indices_class_0 = np.where(np.isin(shap_names_cp, self.infant_class_0))[0]
            indices_class_1 = np.where(np.isin(shap_names_cp, self.infant_class_1))[0]

            self.vis_handler.class_0_indices = np.arange(0, len(indices_class_0), 1, dtype=int)
            self.vis_handler.class_1_indices = np.arange(len(self.vis_handler.class_1_indices),
                                                         len(self.vis_handler.class_1_indices) + len(indices_class_1),
                                                         1, dtype=int)
            self.class_0_indices = indices_class_0
            self.class_1_indices = indices_class_1
        else:
            logging.info("Visualizing all infants...")
            self.class_0_indices = self.vis_handler.class_0_indices
            self.class_1_indices = self.vis_handler.class_1_indices
            self.infant_class_0 = "all"
            self.infant_class_1 = "all"

        logging.info("Averaging SHAP values...")
        averaged_shap_class_0, averaged_shap_class_1 = self.average_cp_shap_values(shap_values_cp,
                                                                                   time=average_time,
                                                                                   body=average_body,
                                                                                   windows=average_windows)

        logging.info("Averaging Feature values...")
        averaged_feature_class_0, averaged_feature_class_1 = self.average_cp_feature_values(time=average_time,
                                                                                            body=average_body,
                                                                                            windows=average_windows)

        # visualize this stuff
        features = 29 if average_body is False else 5
        if single_infant:
            self.vis_shap_cp_skeleton(averaged_shap_class_1, averaged_feature_class_1)

        plot_1 = self.vis_shap_cp_part_beeswarm_multiple(averaged_shap_class_1, averaged_feature_class_1, 1,
                                                num_features=features, display=10)

        save_string = (f"shaps_class_1_averaged_t{1 if average_time else 0}_b{1 if average_body else 0}"
                       f"_w{1 if average_windows else 0}.pdf")
        plot_1.savefig(os.path.join(self.save_dir, save_string), dpi=300)

        plot_0 = self.vis_shap_cp_part_beeswarm_multiple(averaged_shap_class_0, averaged_feature_class_0, 0,
                                                num_features=features, display=10)
        save_string = (f"shaps_class_0_averaged_t{1 if average_time else 0}_b{1 if average_body else 0}"
                       f"_w{1 if average_windows else 0}.pdf")
        plot_0.savefig(os.path.join(self.save_dir, save_string), dpi=300)

        # self.vis_shap_cp_part_beeswarm_multiple(averaged_shap_class_1, averaged_feature_class_1, 0)

        # self.vis_shap_cp_part_beeswarm_single(averaged_shap_class_1, averaged_feature_class_1, class_index, index)
        # self.vis_shap_cp_part_force_single(averaged_shap_class_1, averaged_feature_class_1, class_index, index)
        # self.vis_shap_cp_part_bar_single(averaged_shap_class_1, averaged_feature_class_1, 1)

    def vis_shap_ntu_skeleton(self, shap_values_cp, feature_values, target_class, index_to_vis):
        """
        Create plot for CP SHAPs mapped on the original skeleton.
        :param shap_values_cp:
        :param feature_values:
        :param target_class:
        :param index_to_vis:
        :return:
        """
        # sum the shaps on target class
        summed_shap_values = self.sum_shap_values_ntu(shap_values_cp, target_class)
        # get skeleton
        skeleton_shap_values = summed_shap_values[index_to_vis]
        # self.plot_skeleton_with_slider(summed_shap_values, feature_values)
        # self.plot_skeleton_gif_ntu(skeleton_shap_values, feature_values, index_to_vis, target_class)
        self.plot_n_windows_skeletons_ntu(skeleton_shap_values, feature_values, index_to_vis, target_class)

    def vis_shap_cp_skeleton(self, shap_values_cp, feature_values):
        """
        Create plot for CP SHAPs mapped on the original skeleton.
        :param shap_values_cp:
        :param feature_values:
        :return:
        """
        summed_shap_values = self.sum_shap_values_cp(shap_values_cp)
        # self.plot_skeleton_with_slider(summed_shap_values, feature_values)
        # self.plot_skeleton_gif_cp(summed_shap_values, feature_values)
        self.plot_n_windows_skeletons_cp(summed_shap_values, feature_values)

    def sum_shap_values_ntu(self, shap_values_ntu, target_class):
        """
        Sum SHAP values over relevant keys.
        :param shap_values_ntu:
        :param target_class:
        :return:
        """
        feature_vis_list = list(self.feature_vis_str)
        if self.feature_vis_str is None:
            keys_to_sum = [key for key in shap_values_ntu.keys() if key[0].lower() in feature_vis_list]
        else:
            keys_to_sum = [key for key in shap_values_ntu.keys() if key[0].lower() in feature_vis_list]
        logging.info(f"Visualizing summed features: {self.feature_vis_str}...")
        first_key = list(shap_values_ntu.keys())[0]
        summed_shap_values = np.zeros(shap_values_ntu[first_key].shape[:3])
        for key in keys_to_sum:
            shap_values = shap_values_ntu[key]  # Shape (i, 150, 25, 60)
            summed_shap_values += shap_values[:, :, :, target_class]  # target class to sum
        return summed_shap_values

    def sum_shap_values_cp(self, shap_values_cp):
        """
        Sum SHAP values over relevant keys.
        :param shap_values_cp:
        :return:
        """
        feature_vis_list = list(self.feature_vis_str)
        if self.feature_vis_str is None:
            keys_to_sum = [key for key in shap_values_cp.keys() if key[0].lower() in feature_vis_list]
        else:
            keys_to_sum = [key for key in shap_values_cp.keys() if key[0].lower() in feature_vis_list]
        logging.info(f"Visualizing summed features: {self.feature_vis_str}...")
        first_key = list(shap_values_cp.keys())[0]
        summed_shap_values = np.zeros(shap_values_cp[first_key].shape[:2])
        for key in keys_to_sum:
            shap_values = shap_values_cp[key]  # Shape (t, 29, 2)
            summed_shap_values += shap_values[:, :, 1]  # Class 1 SHAP values
        return summed_shap_values

    def plot_skeleton_ntu(self, ax, window_idx, x, y, z, summed_shap_values):
        """
        Plot skeleton ntu.
        :param ax:
        :param window_idx:
        :param x:
        :param y:
        :param z:
        :param summed_shap_values:
        :return:
        """
        vmin = np.min(summed_shap_values)
        vmax = np.max(summed_shap_values)
        abs_vmax = max(abs(vmin), abs(vmax))

        shap_values_class1 = summed_shap_values[window_idx]
        abs_shap_values = np.abs(shap_values_class1)  # Use absolute SHAP values
        joint_sizes = 25 + (abs_shap_values / abs_vmax) * 300  # Scale joint sizes globally

        # Create scatter plot
        color = convert_color(red_blue)

        scatter = ax.scatter(x, y, z, c=shap_values_class1, cmap=color, s=joint_sizes, edgecolor='black',
                             vmin=vmin, vmax=vmax)

        # Connect joints with lines based on neighbor_link
        for joint_start, joint_end in self.val_loader.dataset.graph.neighbor_link:
            ax.plot([x[joint_start], x[joint_end]],
                    [y[joint_start], y[joint_end]],
                    [z[joint_start], z[joint_end]],
                    color='gray')

        ax.set_title(f'NTU skeleton at Window {window_idx}')

        # Handle colorbar
        if self.colorbar:
            self.colorbar.update_normal(scatter)
        else:
            self.colorbar = plt.colorbar(scatter, ax=ax, label='Summed SHAP Values')

    def plot_skeleton(self, ax, window_idx, x, y, summed_shap_values):
        """
        Plot a single skeleton for the given window.
        :param ax:
        :param window_idx:
        :param x:
        :param y:
        :param summed_shap_values:
        :return:
        """
        vmin = np.min(summed_shap_values)
        vmax = np.max(summed_shap_values)
        abs_vmax = max(abs(vmin), abs(vmax))
        shap_values_class1 = summed_shap_values[window_idx]
        abs_shap_values = np.abs(shap_values_class1)  # Use absolute SHAP values
        joint_sizes = 25 + (abs_shap_values / abs_vmax) * 300  # Scale joint sizes globally

        # Create scatter plot
        color = convert_color(red_blue)

        scatter = ax.scatter(x, y, c=shap_values_class1, cmap=color, s=joint_sizes, edgecolor='black',
                             vmin=summed_shap_values.min(), vmax=summed_shap_values.max())

        # Connect joints with lines based on neighbor_link
        for joint_start, joint_end in self.val_loader.dataset.graph.neighbor_link:
            ax.plot([x[joint_start], x[joint_end]], [y[joint_start], y[joint_end]], color='gray')

        ax.set_xlim([-0.25, 0.25])
        ax.set_ylim([-0.6, 0.5])

        ax.set_title(f'Infant: [{self.infant_class_1}] at Window {window_idx}')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')

        # Handle colorbar
        if self.colorbar:
            self.colorbar.update_normal(scatter)
        else:
            self.colorbar = plt.colorbar(scatter, ax=ax, label='Summed SHAP Values')

    def plot_n_windows_skeletons_ntu(self, summed_shap_values, feature_values, index_to_vis, target_class):
        """
        Plot multiple in a row from the specified windows with a shared colorbar on the right.
        :param summed_shap_values:
        :param feature_values:
        :param index_to_vis:
        :param target_class:
        :return:
        """
        import matplotlib.gridspec as gridspec

        # fig = plt.figure(figsize=(11.69, 6))
        fig = plt.figure(figsize=(18, 10))
        #axes = [fig.add_subplot(1, len(self.n_windows), i + 1, projection='3d') for i in range(len(self.n_windows))]
        gs = gridspec.GridSpec(1, len(self.n_windows), figure=fig, wspace=0.00)

        vmin = np.min(summed_shap_values)  # -0.0060000001
        vmax = np.max(summed_shap_values)  # 0.0060000001
        abs_vmax = max(abs(vmin), abs(vmax))
        color = convert_color(red_blue)
        axes = []
        for i, window_idx in enumerate(self.n_windows):
            # ax = axes[i]
            ax = fig.add_subplot(gs[i], projection='3d')  # Use gridspec for consistent sizing
            axes.append(ax)

            x = np.squeeze(feature_values["j_pos_x"])[index_to_vis, window_idx, :, 0]
            y = np.squeeze(feature_values["j_pos_y"])[index_to_vis, window_idx, :, 0]
            z = np.squeeze(feature_values["j_pos_z"])[index_to_vis, window_idx, :, 0]
            shap_values_window = summed_shap_values[window_idx]
            abs_shap_values = np.abs(shap_values_window)  # Use absolute SHAP values for size
            joint_sizes = 25 + (abs_shap_values / abs_vmax) * 300  # Scale joint sizes globally

            scatter = ax.scatter(x, y, z, c=shap_values_window, cmap=color, s=joint_sizes, edgecolor='black',
                                 vmin=vmin, vmax=vmax)

            # Connect joints with lines based on neighbor_link
            for joint_start, joint_end in self.val_loader.dataset.graph.neighbor_link:
                ax.plot([x[joint_start], x[joint_end]],
                        [y[joint_start], y[joint_end]],
                        [z[joint_start], z[joint_end]],
                        color='gray')

            ax.set_title(f'Window {window_idx}')
            #ax.set_xlim([-0.25, 0.25])
            #ax.set_ylim([-0.6, 0.5])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            # ax.view_init(elev=0, azim=90)
            # ax.view_init(elev=100, azim=-90)
            ax.axis('off')  # Turn off the axis completely
            ax.set_aspect('equal')

        # Add the colorbar to the right of the figure
        # cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # Position for the colorbar
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar_ax = fig.add_axes([0.92, 0.35, 0.015, 0.35])

        # cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.4])
        plt.colorbar(scatter, cax=cbar_ax, label='Summed SHAP Values')

        fig_path = (f"{self.save_dir}/{len(self.n_windows)}"
                    f"_skeletons_class_{target_class+1}_feat_{self.feature_vis_str}.pdf")
        plt.savefig(fig_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Figure saved at {fig_path}")

    def plot_n_windows_skeletons_cp(self, summed_shap_values, feature_values):
        """
        Plot multiple in a row from the specified windows with a shared colorbar on the right.
        :param summed_shap_values:
        :param feature_values:
        :return:
        """
        fig, axes = plt.subplots(1, len(self.n_windows), figsize=(11.69, 6), layout='compressed')
        # fig.subplots_adjust(wspace=0.1, right=0.85)

        vmin = np.min(summed_shap_values) #-0.0060000001
        vmax = np.max(summed_shap_values) # 0.0060000001
        abs_vmax = max(abs(vmin), abs(vmax))
        color = convert_color(red_blue)
        for i, window_idx in enumerate(self.n_windows):
            ax = axes[i]
            x = np.squeeze(feature_values["j_pos_x"])[window_idx]
            y = np.squeeze(feature_values["j_pos_y"])[window_idx]
            shap_values_class1 = summed_shap_values[window_idx]
            abs_shap_values = np.abs(shap_values_class1)  # Use absolute SHAP values for size
            joint_sizes = 25 + (abs_shap_values / abs_vmax) * 300  # Scale joint sizes globally

            scatter = ax.scatter(x, y, c=shap_values_class1, cmap=color, s=joint_sizes, edgecolor='black',
                                 vmin=vmin, vmax=vmax)

            # Connect joints with lines based on neighbor_link
            for joint_start, joint_end in self.val_loader.dataset.graph.neighbor_link:
                ax.plot([x[joint_start], x[joint_end]], [y[joint_start], y[joint_end]], color='gray')

            # Set axis limits and remove the coordinate system (no ticks or axis labels)
            ax.set_title(f'Window {window_idx}')
            ax.set_xlim([-0.25, 0.25])
            ax.set_ylim([-0.6, 0.5])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')  # Turn off the axis completely
            ax.set_aspect('equal')

        # Add the colorbar to the right of the figure
        plt.colorbar(scatter, label='Summed SHAP Values')

        fig_path = (f"{self.save_dir}/{len(self.n_windows)}"
                    f"_skeletons_{self.infant_class_1}_feat_{self.feature_vis_str}.pdf")
        plt.savefig(fig_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Figure saved at {fig_path}")

    def plot_skeleton_with_slider(self, summed_shap_values, feature_values):
        """
        Plot multiple in a row from the specified windows with a shared colorbar on the right.
        :param summed_shap_values:
        :param feature_values:
        :return:
        """
        num_windows = summed_shap_values.shape[0]
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.25)

        self.colorbar = None

        window_idx = 0
        x = np.squeeze(feature_values["j_pos_x"])[window_idx]
        y = np.squeeze(feature_values["j_pos_y"])[window_idx]

        self.plot_skeleton(ax, window_idx, x, y, summed_shap_values)

        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        window_slider = Slider(ax_slider, 'Window', 0, num_windows - 1, valinit=0, valstep=1)

        def update(val):
            window_idx = int(window_slider.val)
            x = np.squeeze(feature_values["j_pos_x"])[window_idx]
            y = np.squeeze(feature_values["j_pos_y"])[window_idx]
            ax.clear()
            self.plot_skeleton(ax, window_idx, x, y, summed_shap_values)
            fig.canvas.draw_idle()

        window_slider.on_changed(update)
        plt.show()

    def plot_skeleton_gif_ntu(self, summed_shap_values, feature_values, index_to_vis, target_class):
        """
        Plot GIF in a row from the specified windows with a shared colorbar on the right.
        :param summed_shap_values:
        :param feature_values:
        :param index_to_vis:
        :param target_class:
        :return:
        """
        num_windows = summed_shap_values.shape[0]
        fig, ax = plt.subplots(figsize=(8, 8))

        self.colorbar = None

        def update_skeleton_ntu(window_idx):
            ax.clear()
            x = np.squeeze(feature_values["j_pos_x"])[index_to_vis, window_idx, :, 0]
            y = np.squeeze(feature_values["j_pos_y"])[index_to_vis, window_idx, :, 0]
            z = np.squeeze(feature_values["j_pos_z"])[index_to_vis, window_idx, :, 0]
            self.plot_skeleton_ntu(ax, window_idx, x, y, z, summed_shap_values)

        ani = animation.FuncAnimation(fig, update_skeleton_ntu, frames=num_windows, repeat=True)
        gif_path = f"{self.save_dir}/skeleton_animation_{target_class+1}_feat_{self.feature_vis_str}.gif"
        ani.save(gif_path, writer='imagemagick', fps=4)
        logging.info(f"GIF saved at: {gif_path}")

    def plot_skeleton_gif_cp(self, summed_shap_values, feature_values):
        """
        Plot GIF in a row from the specified windows with a shared colorbar on the right.
        :param summed_shap_values:
        :param feature_values:
        :return:
        """
        num_windows = summed_shap_values.shape[0]
        fig, ax = plt.subplots(figsize=(8, 8))

        self.colorbar = None

        def update_skeleton(window_idx):
            ax.clear()
            x = np.squeeze(feature_values["j_pos_x"])[window_idx]
            y = np.squeeze(feature_values["j_pos_y"])[window_idx]
            self.plot_skeleton(ax, window_idx, x, y, summed_shap_values)

        ani = animation.FuncAnimation(fig, update_skeleton, frames=num_windows, repeat=True)
        gif_path = f"{self.save_dir}/skeleton_animation_{self.infant_class_1}_feat_{self.feature_vis_str}.gif"
        ani.save(gif_path, writer='imagemagick', fps=4)
        logging.info(f"GIF saved at: {gif_path}")

    def plot_stacked_skeleton_with_shap(self, summed_shap_values, feature_values):
        """
        Plot all windows' skeletons stacked on the x-axis with summed SHAP values.
        :param summed_shap_values:
        :param feature_values:
        :return:
        """
        num_windows = summed_shap_values.shape[0]  # Total number of windows (53 in this case)
        x_offset = 0.5  # Adjust this to control the overlap between skeletons

        plt.figure(figsize=(num_windows * 0.5, 8))  # Adjust figure size for horizontal layout

        for window_idx in range(num_windows):
            # Extract joint positions for the current window
            x = np.squeeze(feature_values["j_pos_x"])[window_idx]  # Shape (29,)
            y = np.squeeze(feature_values["j_pos_y"])[window_idx]  # Shape (29,)

            x = (x - np.mean(x)) / (np.max(x) - np.min(x))
            y = (y - np.mean(y)) / (np.max(y) - np.min(y))

            x += window_idx * x_offset

            # Extract the summed SHAP values for class 1 for this window
            shap_values_class1 = summed_shap_values[window_idx]  # Shape (29,)

            joint_sizes = 50 + (shap_values_class1 - shap_values_class1.min()) / (
                    shap_values_class1.max() - shap_values_class1.min()) * 300

            scatter = plt.scatter(x, y, c=shap_values_class1, cmap='coolwarm', s=joint_sizes, edgecolor='black',
                                  vmin=summed_shap_values.min(), vmax=summed_shap_values.max())

            # Connect joints with lines based on neighbor_link
            for joint_start, joint_end in self.val_loader.dataset.graph.neighbor_link:
                plt.plot([x[joint_start], x[joint_end]], [y[joint_start], y[joint_end]], color='gray')

        # Add a colorbar to show the SHAP value scale
        plt.colorbar(scatter, label='Summed SHAP Value (Class 1)')

        plt.title('Horizontally Stacked Skeletons with SHAP Values (Class 1)')
        plt.xlabel('X Position (Stacked Windows)')
        plt.ylabel('Y Position')
        plt.gca().invert_yaxis()  # Invert Y axis for correct orientation if needed
        plt.show()

    def vis_shap_ntu_part_beeswarm_multiple(self, shap_values, feature_values, target_class, num_features=5, display=5):
        """
        Beeswarm plot.
        :param shap_values:
        :param feature_values:
        :param target_class:
        :param num_features:
        :param display:
        :return:
        """
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'Times New Roman'
        x_feature, x_shaps, y_feature, y_shaps, z_feature, z_shaps = self.split_shaps_features_ntu(feature_values,
                                                                                                   shap_values)

        # (316, 150, 25)
        # shap_values = self.sum_shap_values_ntu(shap_values, target_class)

        # num_features = len(x_shaps)
        nrows, ncols = 4, 6  # Adjusted for 4 rows and 6 columns (for x, y, z plots)
        figsize = (26, 16)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=figsize)

        for i, (x_feature_name, y_feature_name, z_feature_name) in enumerate(
                zip(x_shaps.keys(), y_shaps.keys(), z_shaps.keys())):
            row = i % 4  # Calculate row based on the new grid
            col_x = (i // 4) * 3  # Calculate column for x feature (every set of x, y, z occupies 3 columns)
            col_y = col_x + 1  # Column for y feature
            col_z = col_y + 1  # Column for z feature

            # Process x feature
            x_shap_data = x_shaps[x_feature_name]
            shap_values_x = x_shap_data[:, :, target_class].reshape(-1, num_features)
            x_feature_data = x_feature[x_feature_name][:, :, 0]
            shap_values_obj_x = shap.Explanation(
                values=shap_values_x,
                data=x_feature_data,
                feature_names=self.cp_parts_names if num_features == 5 else self.ntu_joint_names
            )
            shap.plots.beeswarm(shap_values_obj_x, max_display=display, show=False,
                                plot_size=None, ax=axes[row, col_x], color_bar=False)
            axes[row, col_x].set_title(f'{self.x_features_cp_plot[i]}')
            axes[row, col_x].tick_params(labelright=False)

            # Process y feature
            y_shap_data = y_shaps[y_feature_name]
            shap_values_y = y_shap_data[:, :, target_class].reshape(-1, num_features)
            y_feature_data = y_feature[y_feature_name][:, :, 0]
            shap_values_obj_y = shap.Explanation(
                values=shap_values_y,
                data=y_feature_data,
                feature_names=self.cp_parts_names if num_features == 5 else self.ntu_joint_names
            )
            shap.plots.beeswarm(shap_values_obj_y, max_display=display, show=False,
                                plot_size=None, ax=axes[row, col_y], color_bar=False)
            axes[row, col_y].set_title(f'{self.y_features_cp_plot[i]}')

            # Process z feature
            z_shap_data = z_shaps[z_feature_name]
            shap_values_z = z_shap_data[:, :, target_class].reshape(-1, num_features)
            z_feature_data = z_feature[z_feature_name][:, :, 0]
            shap_values_obj_z = shap.Explanation(
                values=shap_values_z,
                data=z_feature_data,
                feature_names=self.cp_parts_names if num_features == 5 else self.ntu_joint_names
            )
            # Display color bar only for the last column (z)
            show_color_bar = col_z == 5
            shap.plots.beeswarm(shap_values_obj_z, max_display=display, show=False,
                                plot_size=None, ax=axes[row, col_z], color_bar=show_color_bar)
            axes[row, col_z].set_title(f'{self.z_features_cp_plot[i]}')

            # Only show x-axis labels for the last row in each column
            if row != nrows - 1:
                axes[row, col_x].set_xlabel(None)
                axes[row, col_y].set_xlabel(None)
                axes[row, col_z].set_xlabel(None)

        return plt

    def vis_shap_ntu_part_beeswarm_single(self, shap_values, feature_values, target_class, index_to_vis, num_features=5,
                                          display=5):
        """
        Beeswarm plot.
        :param shap_values:
        :param feature_values:
        :param target_class:
        :param index_to_vis:
        :param num_features:
        :param display:
        :return:
        """
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'Times New Roman'
        x_feature, x_shaps, y_feature, y_shaps, z_feature, z_shaps = self.split_shaps_features_ntu(feature_values,
                                                                                                   shap_values)

        # (316, 150, 25)
        # shap_values = self.sum_shap_values_ntu(shap_values, target_class)

        # num_features = len(x_shaps)
        nrows, ncols = 4, 6  # Adjusted for 4 rows and 6 columns (for x, y, z plots)
        figsize = (26, 16)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=figsize)

        for i, (x_feature_name, y_feature_name, z_feature_name) in enumerate(
                zip(x_shaps.keys(), y_shaps.keys(), z_shaps.keys())):
            row = i % 4  # Calculate row based on the new grid
            col_x = (i // 4) * 3  # Calculate column for x feature (every set of x, y, z occupies 3 columns)
            col_y = col_x + 1  # Column for y feature
            col_z = col_y + 1  # Column for z feature

            # Process x feature
            x_shap_data = x_shaps[x_feature_name]
            shap_values_x = x_shap_data[index_to_vis, :, :, target_class].reshape(-1, num_features)
            x_feature_data = x_feature[x_feature_name][index_to_vis, :, :, 0]
            shap_values_obj_x = shap.Explanation(
                values=shap_values_x,
                data=x_feature_data,
                feature_names=self.cp_parts_names if num_features == 5 else self.ntu_joint_names
            )
            shap.plots.beeswarm(shap_values_obj_x, max_display=display, show=False,
                                plot_size=None, ax=axes[row, col_x], color_bar=False)
            axes[row, col_x].set_title(f'{self.x_features_cp_plot[i]}')
            axes[row, col_x].tick_params(labelright=False)

            # Process y feature
            y_shap_data = y_shaps[y_feature_name]
            shap_values_y = y_shap_data[index_to_vis, :, :, target_class].reshape(-1, num_features)
            y_feature_data = y_feature[y_feature_name][index_to_vis, :, :, 0]
            shap_values_obj_y = shap.Explanation(
                values=shap_values_y,
                data=y_feature_data,
                feature_names=self.cp_parts_names if num_features == 5 else self.ntu_joint_names
            )
            shap.plots.beeswarm(shap_values_obj_y, max_display=display, show=False,
                                plot_size=None, ax=axes[row, col_y], color_bar=False)
            axes[row, col_y].set_title(f'{self.y_features_cp_plot[i]}')

            # Process z feature
            z_shap_data = z_shaps[z_feature_name]
            shap_values_z = z_shap_data[index_to_vis, :, :, target_class].reshape(-1, num_features)
            z_feature_data = z_feature[z_feature_name][index_to_vis, :, :, 0]
            shap_values_obj_z = shap.Explanation(
                values=shap_values_z,
                data=z_feature_data,
                feature_names=self.cp_parts_names if num_features == 5 else self.ntu_joint_names
            )
            # Display color bar only for the last column (z)
            show_color_bar = col_z == 5
            shap.plots.beeswarm(shap_values_obj_z, max_display=display, show=False,
                                plot_size=None, ax=axes[row, col_z], color_bar=show_color_bar)
            axes[row, col_z].set_title(f'{self.z_features_cp_plot[i]}')

            # Only show x-axis labels for the last row in each column
            if row != nrows - 1:
                axes[row, col_x].set_xlabel(None)
                axes[row, col_y].set_xlabel(None)
                axes[row, col_z].set_xlabel(None)

        return plt

    def vis_shap_cp_part_beeswarm_multiple(self, shap_values, feature_values, class_index, num_features=5, display=5):

        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'Times New Roman'

        x_feature, x_shaps, y_feature, y_shaps = self.split_shaps_features_cp(feature_values, shap_values)

        # if class_index == 1:
        #     x_range = [-.5e-5, 1e-5]
        # else:
        #     x_range = [-1e-5, 1e-5]

        # num_features = len(x_shaps)
        nrows, ncols = 4, 4  # Adjusted for 4x4 grid
        figsize = (26, 16)  # Adjusted figure size for better visualization

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=figsize)

        for i, (x_feature_name, y_feature_name) in enumerate(zip(x_shaps.keys(), y_shaps.keys())):
            row = i % 4  # Calculate row based on the new grid
            col = (i // 4) * 2  # Calculate column for x feature
            col_y = col + 1  # Column for y feature

            # Process x feature
            x_shap_data = x_shaps[x_feature_name]
            shap_values_x = x_shap_data[:, :, class_index].reshape(-1, num_features)
            x_feature_data = x_feature[x_feature_name].squeeze(axis=-1)

            shap_values_obj_x = shap.Explanation(
                values=shap_values_x,
                data=x_feature_data,
                feature_names=self.cp_parts_names if num_features == 5 else self.cp_joint_names_29
            )
            shap.plots.beeswarm(shap_values_obj_x, max_display=display, show=False,
                                plot_size=None, ax=axes[row, col], color_bar=False)
            axes[row, col].set_title(f'{self.x_features_cp_plot[i]}')
            axes[row, col].tick_params(labelright=False)
            # axes[row, col].set_xlim(x_range)

            # Process y feature
            y_feature_data = y_shaps[y_feature_name]
            shap_values_y = y_feature_data[:, :, class_index].reshape(-1, num_features)
            y_feature_data = y_feature[y_feature_name].squeeze(axis=-1)
            shap_values_obj_y = shap.Explanation(
                values=shap_values_y,
                data=y_feature_data,
                feature_names=self.cp_parts_names if num_features == 5 else self.cp_joint_names_29
            )
            # Display color bar only in the last column
            show_color_bar = col_y == 3
            shap.plots.beeswarm(shap_values_obj_y, max_display=display, show=False,
                                plot_size=None, ax=axes[row, col_y], color_bar=show_color_bar)
            axes[row, col_y].set_title(f'{self.y_features_cp_plot[i]}')
            # axes[row, col_y].set_xlim(x_range)

            # Only show x-axis labels for the last row in each column
            if row != nrows - 1:
                axes[row, col].set_xlabel(None)
                axes[row, col_y].set_xlabel(None)

        # plt.savefig(f"shap_values_class_{class_index}.pdf", dpi=300)  # Save the figure first
        # plt.show()
        return plt

    def split_shaps_features_ntu(self, feature_values, shap_values):
        x_shaps = {k: v for k, v in shap_values.items() if '_x' in k}
        y_shaps = {k: v for k, v in shap_values.items() if '_y' in k}
        z_shaps = {k: v for k, v in shap_values.items() if '_z' in k}
        x_feature = {k: v for k, v in feature_values.items() if '_x' in k}
        y_feature = {k: v for k, v in feature_values.items() if '_y' in k}
        z_feature = {k: v for k, v in feature_values.items() if '_z' in k}
        assert len(x_shaps) == len(y_shaps), "The number of x and y shap values should be equal."
        assert len(x_feature) == len(y_feature), "The number of x and y features should be equal."
        assert len(z_feature) == len(z_feature), "The number of x and y features should be equal."
        return x_feature, x_shaps, y_feature, y_shaps, z_feature, z_shaps

    def split_shaps_features_cp(self, feature_values, shap_values):
        x_shaps = {k: v for k, v in shap_values.items() if '_x' in k}
        y_shaps = {k: v for k, v in shap_values.items() if '_y' in k}
        x_feature = {k: v for k, v in feature_values.items() if '_x' in k}
        y_feature = {k: v for k, v in feature_values.items() if '_y' in k}
        assert len(x_shaps) == len(y_shaps), "The number of x and y shap values should be equal."
        assert len(x_feature) == len(y_feature), "The number of x and y features should be equal."
        return x_feature, x_shaps, y_feature, y_shaps

    def vis_shap_cp_part_bar_single(self, shap_values, feature_values, class_index):
        # Filter SHAP values and features for x and y dimensions
        x_shaps = {k: v for k, v in shap_values.items() if '_x' in k}
        y_shaps = {k: v for k, v in shap_values.items() if '_y' in k}
        x_features = {k: v for k, v in feature_values.items() if '_x' in k}
        y_features = {k: v for k, v in feature_values.items() if '_y' in k}

        # Validate the consistency between x and y dimensions
        assert len(x_shaps) == len(y_shaps), "Mismatch in number of x and y SHAP values."
        assert len(x_features) == len(y_features), "Mismatch in number of x and y features."

        num_features = len(x_shaps)
        figsize = (17, 6 * num_features)

        x_range = [0, 1e-5]

        # Create subplots
        fig, axes = plt.subplots(nrows=num_features, ncols=2, figsize=figsize, squeeze=False)

        # Plot each feature pair
        for i, (x_feature_name, y_feature_name) in enumerate(zip(x_shaps.keys(), y_shaps.keys())):
            # Plot x feature
            x_shap_data = x_shaps[x_feature_name]
            x_shap_values = x_shap_data[:, :, class_index].reshape(-1, 5)
            x_feature_data = x_features[x_feature_name].squeeze(axis=-1)
            shap_values_x = shap.Explanation(values=x_shap_values, data=x_feature_data,
                                             feature_names=self.cp_parts_names)
            shap.plots.bar(shap_values_x, max_display=5, show=False, ax=axes[i, 0])
            axes[i, 0].set_title(f'SHAP Values for {x_feature_name} -> Class {class_index}')
            axes[i, 0].set_xlim(x_range)

            # Plot y feature
            y_shap_data = y_shaps[y_feature_name]
            y_shap_values = y_shap_data[:, :, class_index].reshape(-1, 5)
            y_feature_data = y_features[y_feature_name].squeeze(axis=-1)
            shap_values_y = shap.Explanation(values=y_shap_values, data=y_feature_data,
                                             feature_names=self.cp_parts_names)
            shap.plots.bar(shap_values_y, max_display=5, show=False, ax=axes[i, 1])
            axes[i, 1].set_title(f'SHAP Values for {y_feature_name} -> Class {class_index}')
            axes[i, 1].set_xlim(x_range)

            # Hide x-axis labels for all rows except the last
            if i != num_features - 1:
                axes[i, 0].set_xlabel(None)
                axes[i, 1].set_xlabel(None)

        plt.tight_layout()
        plt.show()

    def average_cp_feature_values(self, windows=False, time=False, body=False) -> tuple[dict, dict]:

        # Load features from dataloader based on index
        feature_data_1 = np.array([self.val_loader.dataset[idx][0] for idx in self.class_1_indices])
        feature_data_0 = np.array([self.val_loader.dataset[idx][0] for idx in self.class_0_indices])

        # sort based on features
        feature_values_class_1 = self.vis_handler.extract_feature_values_cp(feature_data_1)
        feature_values_class_0 = self.vis_handler.extract_feature_values_cp(feature_data_0)

        # Average feature values for each class
        averaged_feature_class_0 = self.average_helper(feature_values_class_0, windows, time, body)
        averaged_feature_class_1 = self.average_helper(feature_values_class_1, windows, time, body)

        return averaged_feature_class_0, averaged_feature_class_1

    def average_ntu_feature_values(self, indices, windows=False, time=False, body=False) -> dict:

        # Load features from dataloader based on index

        feature_data = np.array([self.val_loader.dataset[idx][0] for idx in indices])

        # sort based on features
        feature_values_class = self.vis_handler.extract_feature_values_ntu(feature_data)

        # Average feature values for each class
        averaged_feature_class = self.average_helper(feature_values_class, windows, time, body)

        return averaged_feature_class

    def average_ntu_shap_values(self, shap_values_cp, windows=False, time=False, body=False) -> dict:
        """
        Create a dictionary for all the SHAP values
        :param shap_values_cp:
        :param windows:
        :param time:
        :param body:
        :return:
        """
        shap_values_ntu = self.vis_handler.extract_feature_values_ntu(shap_values_cp)

        # Process SHAP values
        averaged_shap_ntu = self.average_helper(shap_values_ntu, windows, time, body)

        return averaged_shap_ntu

    def average_cp_shap_values(self, shap_values_cp, windows=False, time=False, body=False) -> tuple[dict, dict]:
        """
        Create a dictionary for all the SHAP values
        Parameters
        ----------
        shap_values_cp

        Returns
        -------
        :param shap_values_cp: (infants, time, joints, shap value)
        :param time: average over time
        :param body: average over body parts
        :param windows: average over windows

        """
        shap_values_class_0 = {}
        shap_values_class_1 = {}

        shap_values_dict = self.vis_handler.extract_feature_values_cp(shap_values_cp)

        # Loop through the SHAP values and get class-specific values
        for key, value in shap_values_dict.items():
            shap_values_class_0[key], shap_values_class_1[key] = self.vis_handler.get_class_values(value)

        # Process SHAP values for each class
        averaged_shap_class_0 = self.average_helper(shap_values_class_0, windows, time, body)
        averaged_shap_class_1 = self.average_helper(shap_values_class_1, windows, time, body)

        return averaged_shap_class_0, averaged_shap_class_1

    def average_helper(self, values: dict, windows: bool, time: bool, body: bool) -> dict:
        """
        Helper function for average SHAP values
        :param values:
        :param windows:
        :param time:
        :param body:
        :return:
        """
        if windows:
            logging.info("Averaged values over indexes....")
            values = {key: self.vis_handler.average_values_index(value) for key, value in values.items()}

        if time:
            logging.info("Averaged values over time....")
            values = {key: self.vis_handler.average_values_time(value, window_averaged=windows)
                      for key, value in values.items()}

        if body:
            logging.info("Averaged values over body groups.....")
            values = {key: self.vis_handler.average_values_body_groups(value, self.body_parts)
                      for key, value in values.items()}

        return values
