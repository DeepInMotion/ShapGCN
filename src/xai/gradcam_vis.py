import logging
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

from src.xai.xai_utils import (GradCamHandler, plot_skel_ntu, plot_skel_cp, load_gradcam_files, load_gradcam_files_cp,
                               NTU_JOINT_NAMES, CP_JOINT_NAMES_29)

# plt.switch_backend('TkAgg')
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'


class GradCamVisualizer:

    def __init__(self, val_loader, save_dir):
        self.save_dir = save_dir
        self.grad_handler = GradCamHandler()
        self.val_loader = val_loader

        self.n_windows = None
        self.infant_str_class_1 = None
        self.cp_joint_names_29 = CP_JOINT_NAMES_29
        self.ntu_joint_names = NTU_JOINT_NAMES

    def visualize_gradcam_ntu(self, path: str) -> None:
        """
        Visualize the GradCAM values on the NTU skeleton. functions mainly for the paper..
        :return: None
        """
        # load the GradCAM values
        # expected shape is (n, 29) --> individual windows
        # self.grad_handler
        idx_to_vis = 200 # paper class 5!
        self.n_windows = [1, 10, 20, 30, 40, 50]
        logging.info(f"Visualizing skeleton: {idx_to_vis}")
        gradcam_values_ntu, class_numbers, layer_types = load_gradcam_files(path)

        # vis gradcams on skeleton
        for idx, gradcam_data in enumerate(gradcam_values_ntu):
            target_class = class_numbers[idx]
            target_layer = layer_types[idx]
            logging.info(f"Visualizing for class {target_class}...")

            # get the features for this class
            idx_class = self.val_loader.dataset.get_indices(target_class)
            feature_data = np.array([self.val_loader.dataset[idx][0] for idx in idx_class])

            # self.vis_gradcam_skeleton_gif_ntu(gradcam_data, feature_data, idx_to_vis, target_class, target_layer)

            self.plot_n_windows_skeletons_ntu(gradcam_data, feature_data, idx_to_vis, target_class, target_layer)

        logging.info("Visualization done!")

    def vis_gradcam_skeleton_gif_ntu(self, gradcam_data: np.array, feature_data: np.array, idx_to_vis: int,
                                     target_class: int, target_layer: str) -> None:
        """
        Plot GIF in a row from the specified windows with a shared colorbar on the right.
        :param summed_shap_values:
        :param feature_values:
        :param index_to_vis:
        :param target_class:
        :return:
        """
        num_windows = gradcam_data.shape[1]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        gradcam_data = gradcam_data[idx_to_vis, :, :]

        self.colorbar = None

        def update_skeleton_ntu(window_idx):
            ax.clear()
            x = feature_data[idx_to_vis, 0, 0, window_idx, :, 0]
            y = feature_data[idx_to_vis, 0, 1, window_idx, :, 0]
            z = feature_data[idx_to_vis, 0, 2, window_idx, :, 0]
            self.plot_skeleton_ntu(ax, window_idx, x, y, z, gradcam_data)

        ani = animation.FuncAnimation(fig, update_skeleton_ntu, frames=num_windows, repeat=True)
        gif_path = f"{self.save_dir}/skeleton_ani_class_{target_class+1}_layer_{target_layer}.gif"
        ani.save(gif_path, writer='imagemagick', fps=4)
        logging.info(f"GIF saved at: {gif_path}")

    def plot_skeleton_ntu(self, ax, window_idx, x, y, z, gradcam_data):
        """
        Plot skeleton ntu.
        :param ax:
        :param window_idx:
        :param x:
        :param y:
        :param z:
        :param gradcam_data:
        :return:
        """
        gradcam_values_window = gradcam_data[window_idx]
        joint_sizes = 1 + (gradcam_values_window * 300)  # Scale between 1 and 300 based on value

        scatter = ax.scatter(x, y, z, c=gradcam_values_window, cmap='Reds', s=joint_sizes, edgecolors='black',
                             vmin=0, vmax=1)

        # Connect joints with lines based on neighbor_link
        plot_skel_ntu(self.val_loader, ax, window_idx, x, y, z)

        if self.colorbar:
            self.colorbar.update_normal(scatter)
        else:
            self.colorbar = plt.colorbar(scatter, ax=ax, label='GradCAM values')
            # self.colorbar.set_clim(vmin, vmax)

    def plot_n_windows_skeletons_ntu(self, gradcam_data: np.array, feature_data: np.array, idx_to_vis: int,
                                     target_class: int, target_layer: str) -> None:
        """
        Plot multiple in a row from the specified windows with a shared colorbar on the right.
        :param idx_to_vis:
        :param feature_data:
        :param gradcam_data:
        :param target_layer:
        :param target_class:
        :return:
        """
        fig = plt.figure(figsize=(18, 10))

        gs = gridspec.GridSpec(1, len(self.n_windows), figure=fig, wspace=0.00)

        vmin = 0
        vmax = 1
        abs_vmax = max(abs(vmin), abs(vmax))
        axes = []
        for i, window_idx in enumerate(self.n_windows):
            # ax = axes[i]
            ax = fig.add_subplot(gs[i], projection='3d')
            axes.append(ax)

            x = feature_data[idx_to_vis, 0, 0, window_idx, :, 0]
            y = feature_data[idx_to_vis, 0, 1, window_idx, :, 0]
            z = feature_data[idx_to_vis, 0, 2, window_idx, :, 0]

            gradcam_values_window = gradcam_data[idx_to_vis, math.ceil(window_idx/2)]
            # abs_shap_values = np.abs(gradcam_values_window)
            joint_sizes = 25 + (gradcam_values_window / abs_vmax) * 300  # Scale joint sizes globally

            scatter = ax.scatter(x, y, z, c=gradcam_values_window, cmap='Reds', s=joint_sizes, edgecolor='black',
                                 vmin=vmin, vmax=vmax)

            # Connect joints with lines based on neighbor_link
            plot_skel_ntu(self.val_loader, ax, window_idx, x, y, z)
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
        plt.colorbar(scatter, cax=cbar_ax, label='GradCAM Values')

        fig_path = (f"{self.save_dir}/{len(self.n_windows)}"
                    f"_skeletons_class_{target_class+1}_layer_{target_layer}.pdf")
        plt.savefig(fig_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Figure saved at {fig_path}")


    def vis_gradcam_skeleton_ntu(self, gradcam_data: np.array, feature_data: np.array, target_class: int) -> None:
        """
        Visualize the GradCAM values mapped on the NTU skeleton.
        :return:
        """
        pass

    def visualize_gradcam_cp(self, path) -> None:
        """
        Visualize the GradCAM values on the CP skeleton.
        :return: None
        """
        self.infant_str_class_1 = "VelloreTHIN_020_1_1_E"
        target_class = 0
        logging.info(f"Visualizing one infant Class 1: {self.infant_str_class_1}")
        self.n_windows = [1, 10, 20, 30, 40]

        gradcam_values_cp, class_numbers, layer_types = load_gradcam_files_cp(path)


        idx_class_1 = np.where(np.isin(self.val_loader.dataset.names, self.infant_str_class_1))[0]
        feature_data = np.array([self.val_loader.dataset[idx][0] for idx in idx_class_1])
        for idx, gradcam_data in enumerate(gradcam_values_cp):
            target_layer = layer_types[idx]
            gradcam_data = gradcam_data[idx_class_1]
            self.vis_gradcam_skeleton_gif_cp(gradcam_data, feature_data, target_class, target_layer)
            self.plot_n_windows_skeletons_cp(gradcam_data, feature_data, target_class, target_layer)

        logging.info("Done.")

    def plot_n_windows_skeletons_cp(self, gradcam_data: np.array, feature_data: np.array, target_class: int,
                                    target_layer: str) -> None:
        """
        Plot multiple in a row from the specified windows with a shared colorbar on the right.
        :param gradcam_data:
        :param feature_data:
        :param target_class:
        :param target_layer:
        :return:
        """
        fig, axes = plt.subplots(1, len(self.n_windows), figsize=(11.69, 6), layout='compressed')
        # fig.subplots_adjust(wspace=0.1, right=0.85)

        vmin = 0
        vmax = 1
        abs_vmax = max(abs(vmin), abs(vmax))
        for i, window_idx in enumerate(self.n_windows):
            ax = axes[i]
            x = feature_data[window_idx, 0, 0, :, :, 0]
            y = feature_data[window_idx, 0, 1, :, :, 0]

            x = np.mean(x, axis=0)
            y = np.mean(y, axis=0)

            gradcam_values_window = gradcam_data[window_idx, target_class, ...]
            gradcam_values_window = np.mean(gradcam_values_window, axis=0)
            joint_sizes = 25 + (gradcam_values_window / abs_vmax) * 300  # Scale joint sizes globally

            scatter = ax.scatter(x, y, c=gradcam_values_window, cmap='Reds', s=joint_sizes, edgecolor='black',
                                 vmin=vmin, vmax=vmax)

            # Connect joints with lines based on neighbor_link
            plot_skel_cp(self.val_loader, ax, window_idx, x, y)

            # Set axis limits and remove the coordinate system (no ticks or axis labels)
            ax.set_title(f'Window {window_idx}')
            ax.set_xlim([-0.25, 0.25])
            ax.set_ylim([-0.6, 0.5])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')  # Turn off the axis completely
            ax.set_aspect('equal')

        # Add the colorbar to the right of the figure
        plt.colorbar(scatter, label='GradCAM Values')

        fig_path = (f"{self.save_dir}/{len(self.n_windows)}"
                    f"_skeleton_{self.infant_str_class_1}_layer_{target_layer}.pdf")
        plt.savefig(fig_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Figure saved at {fig_path}")



    def vis_gradcam_skeleton_gif_cp(self, gradcam_data: np.array, feature_data: np.array, target_class: int,
                                    target_layer: str) -> None:
        """
        Plot GIF in a row from the specified windows with a shared colorbar on the right.
        :param target_layer:
        :param feature_data:
        :param gradcam_data:
        :param target_class:
        :return:
        """
        num_windows = gradcam_data.shape[0]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax = fig.add_subplot(111)
        self.colorbar = None

        gradcam_data = gradcam_data[:, target_class, ...]
        gradcam_data = np.mean(gradcam_data, axis=1)

        def update_skeleton_ntu(window_idx):
            ax.clear()
            x = feature_data[window_idx, 0, 0, :, :, 0]
            y = feature_data[window_idx, 0, 1, :, :, 0]
            x = np.mean(x, axis=0)
            y = np.mean(y, axis=0)
            self.plot_skeleton_cp(ax, window_idx, x, y, gradcam_data)

        ani = animation.FuncAnimation(fig, update_skeleton_ntu, frames=num_windows, repeat=True)
        gif_path = f"{self.save_dir}/skeleton_ani_class_{target_class+1}_layer_{target_layer}.gif"
        ani.save(gif_path, writer='imagemagick', fps=4)
        logging.info(f"GIF saved at: {gif_path}")

    def plot_skeleton_cp(self, ax, window_idx, x, y, gradcam_data):
        """
        Plot skeleton ntu.
        :param ax:
        :param window_idx:
        :param x:
        :param y:
        :param gradcam_data:
        :return:
        """
        gradcam_values_window = gradcam_data[window_idx]
        joint_sizes = 1 + (gradcam_values_window * 300)  # Scale between 1 and 300 based on value

        scatter = ax.scatter(x, y, c=gradcam_values_window, cmap='Reds', s=joint_sizes, edgecolors='black',
                             vmin=0, vmax=1)

        # Connect joints with lines based on neighbor_link
        plot_skel_cp(self.val_loader, ax, window_idx, x, y)

        if self.colorbar:
            self.colorbar.update_normal(scatter)
        else:
            self.colorbar = plt.colorbar(scatter, ax=ax, label='GradCAM values')
            # self.colorbar.set_clim(vmin, vmax)
