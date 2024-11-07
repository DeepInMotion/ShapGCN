import torch
import logging

class ControllerSaver:
    def __init__(self, controller, writer=None, arch_names=None, hyper_names=None, save_dir=None, args=None):
        self.controller = controller
        self.writer = writer
        self.arch_names = arch_names
        self.hyper_names = hyper_names
        self.save_dir = save_dir
        self.args = args

    def save_controller(self, iteration: int) -> None:
        """
        Save histograms of controller policies
        """
        self._save_policies(self.controller.policies['archspace'], self.arch_names, '/Parameters/Arch/', iteration)
        self._save_policies(self.controller.policies['hpspace'], self.hyper_names, '/Parameters/Hyper/', iteration)
        logging.info("Saving controller policies...")
        save_dir = self.save_dir + self.args.controller_dir
        self.controller.save_policies(save_dir)

    def _save_policies(self, policies, policy_names, prefix, iteration):
        for idx, p in enumerate(policies):
            params = policies[p].state_dict()['params']
            params /= torch.sum(params)

            if self.writer:
                curr_policy = policy_names[idx]
                save_dict = {}
                for i in range(len(params)):
                    param_value = getattr(self.controller, f"{p}_space").get(curr_policy)[i]
                    dict_name = '{}_{}'.format(curr_policy, param_value)
                    save_dict[dict_name] = params[i]

                self.writer.add_scalars(f'{prefix}{curr_policy}', save_dict, iteration)
