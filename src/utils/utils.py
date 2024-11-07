import json
import os
import shutil
import time
import torch
from thop import clever_format, profile
from scipy.stats import beta

def save_checkpoint(model, optimizer, scheduler, epoch: int, epoch_time: float, actions_arch: dict, actions_hyper: dict,
                    best_state: dict, is_best, save_dir: str, student_id: int, argmax=False):
    input_stream_state_dict = model.input_stream.state_dict(prefix="input_stream.")
    main_stream_state_dict = model.main_stream.state_dict(prefix="main_stream.")
    classifier_state_dict = model.classifier.state_dict(prefix="classifier.")

    combined_state_dict = {
        'input_stream': input_stream_state_dict,
        'main_stream': main_stream_state_dict,
        'classifier': classifier_state_dict
    }

    checkpoint = {
        'model': combined_state_dict, 'optimizer': optimizer, 'scheduler': scheduler, 'best_state': best_state,
        'actions': actions_arch, 'actions_hyper': actions_hyper, 'epoch': epoch, 'epoch_time': epoch_time,
    }

    if argmax:
        student = 'argmax'
    else:
        student = 'student'

    save_student_dir = '{}/{}_{}'.format(save_dir, student, student_id)
    check_dir(save_student_dir)
    cp_name = '{}/checkpoint.pth.tar'.format(save_student_dir)
    torch.save(checkpoint, cp_name)
    if is_best:
        model_name = 'student_model_' + str(student_id)
        shutil.copy(cp_name, '{}/{}.pth.tar'.format(save_student_dir, model_name))
        with open('{}/reco_results_student_{}.json'.format(save_student_dir, student_id), 'w') as f:
            if 'cm' in best_state:
                del best_state['cm']
            json.dump(best_state, f)

    model_file_name = '{}/arch_student_{}.txt'.format(save_student_dir, student_id)
    if os.path.isfile(model_file_name) is False:
        with open(model_file_name, 'w') as f:
            print(model, file=f)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_current_timestamp():
    current_time = time.time()
    ms = int((current_time - int(current_time)) * 1000)
    return '[ {},{:0>3d} ] '.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)), ms)


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_macs_and_params(model, input: torch.tensor):
    mac, params = profile(model, inputs=(input, ), verbose=False)
    macs, params = clever_format([mac, params], '%.3f')
    return mac, params

def cooper_pearson_confidence_interval(successes, trials, confidence_level=0.95):
    alpha = 1 - confidence_level

    # If successes are zero or trials are zero, return [0, 0] for lower bound
    if successes == 0:
        lower_bound = 0.0
    else:
        lower_bound = beta.ppf(alpha / 2, successes, trials - successes + 1)

    # If all successes, upper bound is 1
    if successes == trials:
        upper_bound = 1.0
    else:
        upper_bound = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)

    return lower_bound, upper_bound
