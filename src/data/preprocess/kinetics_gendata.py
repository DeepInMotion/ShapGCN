import os
import pickle

import numpy as np
import logging
import os.path as osp

from src.data.feeders.kinetics_feeder import KineticsDataloader
from src.data.preprocess.transform import TransformKinetics
from src.utils.visualization import vis


class KineticsGendata:
    NUM_PERSON_IN = 5
    NUM_PERSON_OUT = 1
    NUM_JOINTS = 18
    NUM_CHANNELS = 3
    MAX_FRAME = 300

    def __init__(self, choice, root_path, raw_path, transform=False, debug=False):

        self.choice = choice
        self.raw_path = raw_path
        self.label_path = None
        self.transform = transform
        self.debug = debug

        # alter the save path
        if self.transform:
            self.out_path = '{}/transformed/{}'.format(root_path, self.choice)
        else:
            self.out_path = '{}/processed/{}'.format(root_path, self.choice)

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        if self.transform:
            self.transformer = TransformKinetics(self.out_path)

        # logging - store log in save path
        self.data_corrupt_logger = logging.getLogger('kinetics_corrupt_log')
        self.data_corrupt_logger.setLevel(logging.INFO)
        self.data_corrupt_logger.addHandler(logging.FileHandler(osp.join(self.out_path, 'kinetics_corrupt_log.log')))

    def gendata_kinetics(self):
        part = ['train', 'val']
        for p in part:

            self.label_path = '{}/kinetics_{}_label.json'.format(self.raw_path, p)
            raw_path_part = '{}/kinetics_{}'.format(self.raw_path, p)

            crawler = KineticsDataloader(phase=part,
                                         data_path=raw_path_part,
                                         label_path=self.label_path,
                                         num_person_in=self.NUM_PERSON_IN,
                                         num_person_out=self.NUM_PERSON_OUT,
                                         window_size=self.MAX_FRAME,
                                         debug=self.debug)

            sample_name = crawler.sample_name
            skeleton_name = []
            total_frames = []
            results = []

            raw_skeleton = np.zeros((len(sample_name), self.NUM_CHANNELS, self.MAX_FRAME, self.NUM_JOINTS,
                                     self.NUM_PERSON_OUT), dtype=np.float32)

            # prog_bar = mmcv.ProgressBar(len(sample_name))
            for i, s in enumerate(tqdm(sample_name)):
                data, label = crawler[i]
                raw_skeleton[i, :, 0:data.shape[1], :, :] = data
                skeleton_name.append(label)
                total_frames.append(data.shape[1])
                results.append(raw_skeleton)
                # prog_bar.update()

            # TODO: check kinetic transformation - drop the kinetics files database?!?
            option_1 = ['pad', 'parallel_s', 'parallel_h', 'sub']

            if self.transform:
                # disentangle transformation and reading skeleton data
                transformed_skeleton, pop_list = self.transformer.transform(raw_skeleton.copy(), option_1,
                                                                            skeleton_name, total_frames)

            for i, skel_name in enumerate(sample_name):
                vis(transformed_skeleton[i, ...])
                vis(raw_skeleton[i, ...])

            if self.transform:
                # disentangle transformation and reading skeleton data
                transformed_skeleton, pop_list = self.transformer.transform(raw_skeleton.copy(), option_1,
                                                                            sample_name, total_frames)
            if not self.debug:
                with open('{}/{}_label.pkl'.format(self.out_path, p), 'wb') as f:
                    pickle.dump((sample_name, list(skeleton_name)), f)
            print('{} finished!'.format(p))

