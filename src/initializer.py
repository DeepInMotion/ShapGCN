import logging
import numpy as np
import os
import pynvml
import random
import sys
import torch
import copy
import warnings
from datetime import datetime
from time import strftime

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group, barrier
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.utils.utils import import_class
from . import scheduler


class Initializer:
    def __init__(self, args):
        self.args = args
        self.train_loader = None
        self.val_loader = None
        now = datetime.now()
        if not self.args.cont_training:
            if not self.args.debug:
                args.logs.dir += now.strftime("%d_%m_%Y_%H:%M:%S")
            else:
                args.logs.dir += "debug"
        self.kwargs = None
        self.init_logger()

        if self.args.ddp:
            logging.info("Using DDP!")
            self.init_ddp()
            self.gpu_id = int(os.environ["LOCAL_RANK"])
            logging.info("GPUs used are: {}".format(self.gpu_id))

        logging.info('')
        logging.info('Starting preparing {}'.format(args.mode))
        self.init_environment()
        self.init_device()
        self.init_data()
        self.init_optimizers()
        self.init_search_space()
        self.init_controller_hyper()
        self.init_loss_func()
        logging.info('Successful!')
        logging.info('')

    def init_logger(self):
        # only init on rank == 0 if in DDP mode
        self.save_dir = self.init_logging()
        with open(f'{self.save_dir}/config_{self.args.dataset}.yaml', 'w') as f:
            OmegaConf.save(self.args, f)
        logging.info('Experiment folder path is: {}'.format(self.save_dir))

    def init_environment(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logging.info("Seed set to: {}".format(self.args.seed))

        if self.args.debug:
            logging.info("Setting DEBUG mode...")
            self.no_progress_bar = False
            self.writer = SummaryWriter(log_dir=self.save_dir)
            self.args.argmax_epochs = self.args.debug_argmax_epoch
            self.args.warmup_rollouts = self.args.debug_warmup_rollouts
            self.args.rollouts = self.args.debug_rollouts
            self.args.train_epochs = self.args.debug_train_epochs
        else:
            self.no_progress_bar = self.args.no_progress_bar
            self.writer = SummaryWriter(log_dir=self.save_dir)
            warnings.filterwarnings('ignore')

    def init_device(self):
        if self.args.ddp:
            world_size = torch.cuda.device_count()
            assert world_size > 1, 'More than 1 GPU need to be accessible to use DDP training'
            return
        logging.info("Cuda: {}".format(torch.cuda.is_available()))
        logging.info("Device count {}".format(torch.cuda.device_count()))
        if len(self.args.gpus) > 0 and torch.cuda.is_available():
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.args.gpus[0])
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memused = meminfo.used / 1024 / 1024
            logging.info('GPU-{} used: {}MB'.format(self.args.gpus[0], memused))
            if memused > 2200:
                pynvml.nvmlShutdown()
                logging.info('')
                logging.error('GPU-{} is occupied!'.format(self.args.gpus[0]))
                raise ValueError()
            pynvml.nvmlShutdown()
            self.output_device = self.args.gpus[0]
            self.gpu_id = torch.device('cuda:{}'.format(self.output_device))
            logging.info(f"Using GPU: {self.gpu_id}")
            torch.cuda.set_device(self.output_device)
        # elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and len(self.args.gpus) > 0:
        #       logging.info('Using MPS!')
        #       self.output_device = torch.device("mps")
        #       self.gpu_id = torch.device("mps")
        else:
            logging.info('Using CPU!')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.output_device = None
            self.gpu_id = torch.device('cpu')

        # change num threads for pytorch depending on device
        num_cpus = os.cpu_count()
        if self.args.num_threads > num_cpus:
            new_num_threads = num_cpus - 4
            logging.info("Decreasing the number of PyTorch threads from {} -> {}".format(self.args.num_workers,
                                                                                         new_num_threads))
            self.args.num_workers = new_num_threads
        else:
            logging.info("Number of PyTorch threads: {}".format(self.args.num_threads))
        torch.set_num_threads(self.args.num_threads)

    def init_data(self):
        dataset_args = self.args.dataset_args
        if self.args.debug and self.args.debug_load_small_set:
            dataset_args['debug'] = True
        else:
            dataset_args['debug'] = False
        dataset_args['dataset'] = self.args.dataset

        if self.args.dataset == "cp29":
            from src.data.feeders.cp_feeder import CPDataloader
            self.feeder_train = CPDataloader(phase="train", args=dataset_args)
            self.feeder_val = CPDataloader(phase="eval", args=dataset_args)
            self.feeder_test = CPDataloader(phase="test", args=dataset_args)
        elif self.args.dataset == "cp19":
            from src.data.feeders.cp_feeder import CPDataloader
            self.feeder_train = CPDataloader(phase="train", args=dataset_args)
            self.feeder_val = CPDataloader(phase="eval", args=dataset_args)
        elif self.args.dataset in ["xsub60", "xview60", "xsub120", "xsetup120"]:
            from src.data.feeders.ntu_feeder import NTUDataloader
            self.feeder_train = NTUDataloader(phase="train", args=dataset_args)
            self.feeder_val = NTUDataloader(phase="eval", args=dataset_args)
        else:
            logging.info("Dataset {} not supported!".format(self.args.dataset))
            raise NotImplementedError

        self.parts = self.feeder_train.parts
        self.data_shape = self.feeder_train.shape
        self.num_classes = self.feeder_train.num_classes

        assert self.data_shape[0] == len(self.args.dataset_args.inputs)

        kwargs = {
            'data_shape': self.data_shape,
            'num_class': self.num_classes,
            'A': torch.Tensor(self.feeder_train.graph.A),
            'parts': self.parts,
            'bias': True,
            'edge': True,
            'residual': True,
        }

        self.kwargs = kwargs
        if self.args.ddp:
            sampler_train = DistributedSampler(self.feeder_train, shuffle=False, drop_last=False)
            shuffle_ = False
        else:
            sampler_train = None
            shuffle_ = True

        # count chips and change num workers
        num_cpus = os.cpu_count()
        if self.args.num_workers > num_cpus:
            new_num_workers = num_cpus - 2
            logging.info(f"Decreasing the number of DataLoader workers from {self.args.num_workers} "
                         f"-> {new_num_workers}")
            self.args.num_workers = new_num_workers
        else:
            logging.info(f"Number of DataLoader workers: {self.args.num_workers}")

        if self.args.mode == "xai":
            self.args.shuffle = False
            logging.info(f"Setting shuffle parameter in validation loader to {self.args.shuffle} for XAI mode.")

        self.train_loader = DataLoader(self.feeder_train,
                                       batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                       pin_memory=self.args.pin_memory, shuffle=shuffle_, drop_last=True,
                                       sampler=sampler_train)
        self.val_loader = DataLoader(self.feeder_val,
                                     batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                     pin_memory=self.args.pin_memory, shuffle=self.args.shuffle, drop_last=True)

        if self.args.dataset in ['cp19', 'cp29']:
            logging.info("Initializing test loader...")
            self.test_loader = DataLoader(self.feeder_test,
                                          batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                          pin_memory=self.args.pin_memory, shuffle=False, drop_last=False)

        # self.location_loader = self.feeders['location'] if dataset_name == 'ntu' else None
        logging.info('Dataset: {}'.format(self.args.dataset))
        logging.info('Batch size: train-{}, eval-{}'.format(self.args.batch_size, self.args.batch_size))
        logging.info('Data shape (branch, channel, frame, joint, person): {}'.format(self.feeder_train.shape))
        logging.info('Number of action classes: {}'.format(self.num_classes))

    def update_batch_size(self, new_batch_size):
        old_batch_size = self.args.batch_size
        self.args.batch_size = new_batch_size

        if self.args.ddp:
            sampler_train = DistributedSampler(self.feeder_train, shuffle=False, drop_last=False)
            shuffle_ = False
        else:
            sampler_train = None
            shuffle_ = True

        # Update the batch size
        self.train_loader = DataLoader(self.feeder_train,
                                       batch_size=new_batch_size, num_workers=self.args.num_workers,
                                       pin_memory=self.args.pin_memory, shuffle=shuffle_, drop_last=True,
                                       sampler=sampler_train)

        # Update the batch size
        self.val_loader = DataLoader(self.feeder_val,
                                     batch_size=new_batch_size, num_workers=self.args.num_workers,
                                     pin_memory=self.args.pin_memory, shuffle=self.args.shuffle, drop_last=True)

        logging.info('Updated Batch size from: old-{} -> new-{}'.format(old_batch_size, new_batch_size))

    def init_controller_hyper(self):
        from src.model.controller.reinforce_controller import ReinforceController
        self.controller = ReinforceController(self.arch_choices, self.hyper_choices, self.gpu_id, self.save_dir,
                                              self.args)

    def init_optimizers(self):
        # cast to list -.-
        optim_choice = OmegaConf.to_container(self.args.hyper.optimizers)
        self.optim_list = []
        for item in optim_choice:
            try:
                optimizer = import_class('torch.optim.{}'.format(item))
                self.optim_list.append(optimizer)
            except:
                logging.warning('This optimizer: {} is not known!'.format(self.args.optimizer_controller))

        logging.info('Optimizers: {}'.format(self.optim_list))

    def init_loss_func(self):
        self.loss_func = torch.nn.CrossEntropyLoss().to(self.gpu_id)
        logging.info('Loss function: {}'.format(self.loss_func.__class__.__name__))

    def init_lr_scheduler(self):
        scheduler_args = self.args.scheduler_args[self.args.lr_scheduler]
        self.max_epoch = scheduler_args['max_epoch']
        lr_scheduler = scheduler.create(self.args.lr_scheduler, len(self.train_loader), **scheduler_args)
        self.eval_interval, self.lr_lambda = lr_scheduler.get_lambda()
        self.scheduler = None

    def init_search_space(self):
        """
        Load search space fur current run.
        """
        logging.info("Load Search Space...")

        if self.args.debug:
            logging.info("Smaller Search Space for DEBUG mode")
            if self.args.old_sp:
                logging.info("Loading DEBUG and OLD Search Space")
                search_space = self.args.dev
                search_hyper = self.args.hyper_dev
            else:
                logging.info("Loading DEBUG and NEW Search Space")
                search_space = self.args.dev_2
                search_hyper = self.args.hyper_dev

        else:
            if self.args.old_sp:
                logging.info("Loading whole and OLD Search Space")
                search_space = self.args.arch
                search_hyper = self.args.hyper
            else:
                logging.info("Loading whole and NEW Search Space")
                search_space = self.args.arch_2
                search_hyper = self.args.hyper

        # hack for omega config
        from collections import OrderedDict
        arch_space = OmegaConf.to_container(search_space, resolve=True)
        self.arch_choices = OrderedDict(arch_space)
        self.arch_choices_copy = copy.deepcopy(self.arch_choices)
        self.arch_computations = len(arch_space)
        self.size_search = sum([len(x) for x in arch_space.values()])

        hyper_space = OmegaConf.to_container(search_hyper, resolve=True)
        self.hyper_choices = OrderedDict(hyper_space)
        self.hyper_choices_copy = copy.deepcopy(self.hyper_choices)
        self.hyper_computations = len(hyper_space)
        self.hyper_size = sum([len(x) for x in hyper_space.values()])

        self.arch_names = []
        self.arch_values = []
        for items in self.arch_choices.items():
            self.arch_names.append(items[0])
            self.arch_values.append(items[1])

        self.hyper_names = []
        self.hyper_values = []
        for items in self.hyper_choices.items():
            self.hyper_names.append(items[0])
            self.hyper_values.append(items[1])

        logging.info("Architecture Search Space is: {}".format(self.arch_choices))
        logging.info("Hyperparameter Search Space is: {}".format(self.hyper_choices))
        logging.info("Search Space size: {}".format(self.size_search + self.hyper_size))

    def init_logging(self):
        if not self.args.cont_training:
            if self.args.debug:
                save_dir = '{}/{}/{}_{}/{}'.format(self.args.work_dir, self.args.mode, self.args.experiment,
                                                   self.args.dataset, "debug")
            else:
                ct = strftime('%Y-%m-%d_%H-%M-%S')
                save_dir = '{}/{}/{}_{}/{}'.format(self.args.work_dir, self.args.mode, self.args.experiment,
                                                   self.args.dataset, ct)
        else:
            save_dir = self.args.cont_dir
        self.make_log_folder(save_dir)
        log_format = '[ %(asctime)s ] %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
        handler = logging.FileHandler('{}/logfile.txt'.format(save_dir), mode='a', encoding='UTF-8')
        handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(handler)
        return save_dir

    @staticmethod
    def init_ddp():
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        barrier()

    @staticmethod
    def make_log_folder(folder):
        print(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
