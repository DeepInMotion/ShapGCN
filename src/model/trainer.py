import gc
from typing import Union, Tuple

import torch
import logging
import numpy as np
from time import time

from numpy import ndarray
from tqdm import tqdm
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from src.utils.utils import cooper_pearson_confidence_interval
from scipy.special import softmax


class Trainer:
    def __init__(self, args, student_model, train_loader, val_loader, loss_func, optimizer, scheduler, scheduler_warmup,
                 writer, gpu_id, num_classes: int, no_progress_bar: bool, new_batch_size: int):
        self.args = args
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.gpu_id = gpu_id
        self.num_classes = num_classes
        self.no_progress_bar = no_progress_bar
        self.new_batch_size = new_batch_size
        self.scheduler_warmup = scheduler_warmup

        # check if CP data is used --> evaluation based on windows
        if self.args.dataset == 'cp19' or self.args.dataset == 'cp29':
            self.cp_used = True
        else:
            self.cp_used = False

    def trainer_train(self, epoch: int, max_epoch: int, student_id: int):
        """
        Train student network.
        :param epoch:
        :param max_epoch:
        :param student_id:
        """
        if self.args.ddp:
            self.student_model.to(self.gpu_id)
            self.student_model = DDP(self.student_model, device_ids=[self.gpu_id])
            self.train_loader.sampler.set_epoch(epoch)
        elif self.gpu_id:
            self.student_model.to(self.gpu_id)

        self.student_model.train()

        start_train_time = time()
        train_loss_list = []
        # train_y_true = [[] for _ in range(self.num_classes)]
        # train_probs = [[] for _ in range(self.num_classes)]
        # train_all_preds = []
        train_num_top1, train_num_sample = 0, 0

        train_iter = self.train_loader if self.no_progress_bar else \
            tqdm(self.train_loader, leave=True, desc="Train student {}".format(student_id))
        for num, (x, y, _) in enumerate(train_iter):
            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)

            out = self.student_model(x)
            loss = self.loss_func(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accuracies Top 1 & 5
            train_num_sample += x.size(0)
            reco_top1 = out.max(1)[1]
            train_num_top1 += reco_top1.eq(y).sum().item()

            # Logits to probability with Softmax
            # out_probs_train = torch.softmax(out, dim=1)
            # for c in range(self.num_classes):
            #     train_y_true[c].extend((y == c).cpu().numpy())
            #     train_probs[c].extend(out_probs_train[:, c].cpu().detach().numpy())

            # preds for auc
            # train_all_preds.append(out.data.cpu().numpy())

            # Progress
            lr = self.optimizer.param_groups[0]['lr']
            # if self.writer:
            #     self.writer.add_scalar('learning_rate/student_{}'.format(self.student_model.student_id), lr, num)
            #     self.writer.add_scalar('train_loss', loss.item(), num)
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch + 1, max_epoch, num + 1, len(self.train_loader), loss.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))

            train_loss_list.append(loss.item())

        if (epoch + 1) < self.args.sched_args.warm_up and (self.scheduler and self.scheduler_warmup) is not None:
            self.scheduler_warmup.step()
            # logging.info("Current lr: {}".format(self.scheduler_warmup.get_last_lr()))
        else:
            self.scheduler.step(epoch + 1)
            # logging.info("Current lr: {}".format(self.scheduler.get_last_lr()))

        # TODO: DDP sync loss etc.
        if self.args.ddp:
            logging.info("GPU {} done.".format(self.gpu_id))

        # Train Results
        train_acc = round(train_num_top1 / train_num_sample, 3)
        epoch_time = time() - start_train_time
        train_loss = np.mean(train_loss_list)

        # AUC score for each class
        # train_auc = roc_auc_score(train_y_true, train_probs, average='weighted')

        if self.writer:
            self.writer.add_scalar('train_acc/student_{}'.format(self.student_model.student_id),
                                   train_acc, epoch + 1)
            self.writer.add_scalar('train_loss/student_{}'.format(self.student_model.student_id),
                                   train_loss, epoch + 1)
            self.writer.add_scalar('train_time/student_{}'.format(self.student_model.student_id),
                                   epoch_time, epoch + 1)
            # self.writer.add_scalar('train_auc/student_{}'.format(self.student_model.student_id),
            #                       train_auc, epoch + 1)

        logging.info(
            'Epoch: {}/{}, Acc: {:d}/{:d}({:.2%}), Time: {:.2f}s, Mean loss:{:.4f}, lr:{:.4f}'.format(
                epoch + 1, max_epoch, train_num_top1, train_num_sample, train_acc, epoch_time,
                train_loss, lr))
        logging.info('')

        gc.collect()
        torch.cuda.empty_cache()
        return epoch_time, train_acc, train_loss

    def trainer_eval(self, epoch: int, student_id: int, cp_used: bool, ensemble: bool = False) -> dict:
        """
        Eval the student network - check if CP dataset.
        :param epoch:
        :param student_id:
        :param cp_used:
        :param ensemble:
        """
        self.student_model.to(self.gpu_id)
        if cp_used:
            return self.trainer_eval_cp(epoch, student_id, ensemble)

        self.student_model.eval()
        start_eval_time = time()
        with torch.no_grad():
            eval_num_top1, eval_num_top5 = 0, 0
            eval_num_sample, eval_loss = 0, []
            # eval_y_true = [[] for _ in range(self.num_classes)]
            # eval_out_probs = [[] for _ in range(self.num_classes)]
            # eval_all_preds = []
            eval_cm = np.zeros((self.num_classes, self.num_classes))
            eval_iter = self.val_loader if self.no_progress_bar else \
                tqdm(self.val_loader, leave=True, desc="Eval student {}".format(student_id))
            for num, (x, y, _) in enumerate(eval_iter):
                x = x.to(self.gpu_id)
                y = y.to(self.gpu_id)

                # Calculating Output
                out = self.student_model(x)

                loss = self.loss_func(out, y)
                eval_loss.append(loss.item())

                # Accuracies
                eval_num_sample += x.size(0)
                reco_top1 = out.max(1)[1]
                eval_num_top1 += reco_top1.eq(y).sum().item()
                if out.size()[1] > 5:
                    reco_top5 = torch.topk(out, 5)[1]
                    eval_num_top5 += sum([y[n] in reco_top5[n, :] for n in range(x.size(0))])
                else:
                    eval_num_top5 = 0

                # Confusion Matrix
                for i in range(x.size(0)):
                    eval_cm[y[i], reco_top1[i]] += 1

                # Logits to probability with Softmax
                # out_probs = torch.softmax(out, dim=1)

                # for c in range(self.num_classes):
                #     eval_y_true[c].extend((y == c).cpu().numpy())
                #     eval_out_probs[c].extend(out_probs[:, c].cpu().detach().numpy())

                # preds for auc
                # eval_all_preds.append(out.data.cpu().numpy())

                # Showing Progress
                if self.no_progress_bar:
                    logging.info('Batch: {}/{}'.format(num + 1, len(self.val_loader)))

        # AUC score for each class
        # eval_auc = roc_auc_score(eval_y_true, eval_out_probs, average='weighted')
        # Showing Evaluating Results
        eval_top1 = round(eval_num_top1 / eval_num_sample, 3)
        eval_top5 = round(eval_num_top5 / eval_num_sample, 3)
        eval_loss = np.mean(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.val_loader) * self.new_batch_size / eval_time / len(self.args.gpus)

        logging.info(
            'Top-1 accuracy: {:d}/{:d}({:.2%}), Top-5 accuracy: {:d}/{:d}({:.2%}), Mean loss:{:.4f}'.
            format(eval_num_top1, eval_num_sample, eval_top1, eval_num_top5, eval_num_sample, eval_top5, eval_loss))
        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequences/(second*GPU)'.format(eval_time, eval_speed))
        logging.info('')
        if self.writer:
            self.writer.add_scalar('eval_acc/student_{}'.format(student_id), eval_top1, epoch + 1)
            self.writer.add_scalar('eval_loss/student_{}'.format(student_id), eval_loss, epoch + 1)
            # self.writer.add_scalar('eval_auc/student_{}'.format(student_id), eval_auc, epoch + 1)

        torch.cuda.empty_cache()
        metrics = {
            "acc": eval_top1,
            "acc5": eval_top5,
            "auc": 0,
            "cm": eval_cm,
        }
        return metrics

    def trainer_eval_cp(self, epoch: int, student_id: int, ensemble=False) -> Union[tuple[ndarray, ndarray, ndarray],
                                                                                    dict]:
        """
        Eval the student network for the CP dataset.
        :param ensemble:
        :param epoch:
        :param student_id:
        """
        self.student_model.eval()
        all_preds = []
        all_labels = []
        all_video_ids = []
        start_eval_time = time()
        with torch.no_grad():
            eval_iter = self.val_loader if self.no_progress_bar else \
                tqdm(self.val_loader, leave=True, desc="Eval student {}".format(student_id))
            for num, (x, y, video_ids) in enumerate(eval_iter):
                x = x.to(self.gpu_id)
                y = y.to(self.gpu_id)

                # Calculating Output
                out = self.student_model(x)

                # Store predictions, labels and video ids
                all_preds.append(out.data.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_video_ids.append(video_ids)

            # Concatenate
            preds = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)
            video_ids = np.concatenate(all_video_ids)

            # Compute weighted mean loss
            loss_positives = []
            loss_negatives = []

            for sample_pred, sample_label in zip(preds, labels):
                # Convert to tensors and move to the specified device
                sample_pred_tensor = torch.tensor(sample_pred).unsqueeze(0).to(self.gpu_id)
                sample_label_tensor = torch.tensor(sample_label).unsqueeze(0).to(self.gpu_id)

                sample_loss = self.loss_func(sample_pred_tensor, sample_label_tensor)

                sample_loss = sample_loss.item()

                # Append to appropriate list based on the label
                if sample_label == 1:
                    loss_positives.append(sample_loss)
                elif sample_label == 0:
                    loss_negatives.append(sample_loss)

            eval_positive_loss = np.mean(loss_positives)
            eval_negative_loss = np.mean(loss_negatives)
            eval_loss = (eval_positive_loss + eval_negative_loss) / 2

            if ensemble:
                return preds, labels, video_ids

            # Compute some metrics
            metrics_cp = self.evaluate_cp(preds, labels, video_ids, prediction_threshold=0.45, subject=True)
            metrics_cp_win = self.evaluate_cp(preds, labels, video_ids, prediction_threshold=0.45, subject=False)

        eval_time = time() - start_eval_time
        eval_speed = len(self.val_loader) * self.new_batch_size / eval_time / len(self.args.gpus)

        if self.writer:
            self.writer.add_scalar('eval_acc/student_{}'.format(student_id), metrics_cp["acc"], epoch + 1)
            self.writer.add_scalar('eval_loss/student_{}'.format(student_id), eval_loss, epoch + 1)
            self.writer.add_scalar('eval_auc/student_{}'.format(student_id), metrics_cp["auc"], epoch + 1)
            self.writer.add_scalar('eval_pos_loss/student_{}'.format(student_id), eval_positive_loss, epoch + 1)
            self.writer.add_scalar('eval_neg_loss/student_{}'.format(student_id), eval_negative_loss, epoch + 1)
            self.writer.add_scalar('eval_win_auc/student_{}'.format(student_id), metrics_cp_win["auc"], epoch + 1)

        # Logging the metrics
        logging.info("Subject metrics....")

        # confidence interval
        # tn, fp, fn, tp = metrics_cp['cm'].ravel()
        # lower_bound, upper_bound = cooper_pearson_confidence_interval(tn, 118, 0.95)

        # logging.info(f"Specificity Lower bound: {lower_bound}; upper bound: {upper_bound}")

        # lower_bound, upper_bound = cooper_pearson_confidence_interval(tp, 21, 0.95)
        # logging.info(f"Sensitivity Lower bound: {lower_bound}; upper bound: {upper_bound}")

        # lower_bound, upper_bound = cooper_pearson_confidence_interval(tp + tn, 118 + 21, 0.95)
        # logging.info(f"Accuracy Lower bound: {lower_bound}; upper bound: {upper_bound}")

        logging.info(
            f'Accuracy: ({metrics_cp["acc"]:.2%}), loss: {eval_loss:.4f}, AUC: {metrics_cp["auc"]:.4f}, '
            f'Precision: {metrics_cp["precision"]:.3f}, Recall: {metrics_cp["recall"]:.3f}, '
            f'F1: {metrics_cp["F1"]:.3f}, Sensitivity: {metrics_cp["sens"]:.3f}, Specificity: {metrics_cp["spec"]:.3f},'
            f' PPV: {metrics_cp["ppv"]:.3f}, NPV: {metrics_cp["npv"]:.3f}, Bal. Accuracy: {metrics_cp["b_acc"]:.3f} '
            f'CM: {metrics_cp["cm"]}'
        )
        logging.info("Window metrics...")
        logging.info(
            f'Accuracy: ({metrics_cp_win["acc"]:.2%}), loss: {eval_loss:.4f}, AUC: {metrics_cp_win["auc"]:.4f}, '
            f'Precision: {metrics_cp_win["precision"]:.3f}, Recall: {metrics_cp_win["recall"]:.3f}, '
            f'F1: {metrics_cp_win["F1"]:.3f}, Sensitivity: {metrics_cp_win["sens"]:.3f}, '
            f'Specificity: {metrics_cp_win["spec"]:.3f}, PPV: {metrics_cp_win["ppv"]:.3f}, '
            f'NPV: {metrics_cp_win["npv"]:.3f}, Bal. Accuracy: {metrics_cp_win["b_acc"]:.3f} CM: {metrics_cp_win["cm"]}'
        )

        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequences/(second*GPU)'.format(eval_time, eval_speed))
        logging.info('')

        torch.cuda.empty_cache()
        return metrics_cp

    def trainer_eval_single(self, new_student, x, y, video_ids) -> dict:

        self.student_model = new_student
        self.student_model.eval()
        all_preds = []
        all_labels = []
        all_video_ids = []
        start_eval_time = time()
        with torch.no_grad():
            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)
            out = self.student_model(x)

        # Store predictions, labels and video ids
        all_preds.append(out.data.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        all_video_ids.append(video_ids)

        # Concatenate
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        video_ids = np.concatenate(all_video_ids)

    @staticmethod
    def evaluate_cp(
            preds: np.ndarray,
            labels: np.ndarray,
            video_ids: np.ndarray,
            aggregate_binary: bool = False,
            aggregate_binary_threshold: float = 0.5,
            median_aggregation: bool = True,
            prediction_threshold: float = 0.5,
            auto_threshold: bool = False,
            subject: bool = True,
            normalized: bool = False
    ) -> dict:
        # Compute softmax values
        if normalized:
            softmax_vals = preds
        else:
            softmax_vals = softmax(preds, axis=1)

        # Extract scores of sequence parts
        parts_scores = softmax_vals[:, 1]

        # Aggregate scores by video IDs
        video_ids_scores = defaultdict(list)
        video_ids_label = {}
        for score, video_id, label in zip(parts_scores, video_ids, labels):
            video_ids_scores[video_id].append(score)
            video_ids_label[video_id] = label

        # Calculate accumulative scores
        accumulative_scores = []
        outcome_labels = []
        sample_weights = []
        for video_id, scores in video_ids_scores.items():
            label = video_ids_label[video_id]
            scores = np.array(scores)
            if aggregate_binary:
                scores = (scores >= aggregate_binary_threshold).astype(int)
            if subject:
                accumulative_score = np.median(scores) if median_aggregation else np.mean(scores)
                accumulative_score = np.clip(accumulative_score, 0, 1)  # Ensure score is between 0 and 1
                accumulative_scores.append(accumulative_score)
                outcome_labels.append(label)
            else:
                scores = np.clip(scores, 0, 1)
                accumulative_scores.extend(scores)
                outcome_labels.extend([label] * len(scores))
                sample_weights.extend([(1 / len(scores)) / len(video_ids_label)] * len(scores))

        accumulative_scores = np.array(accumulative_scores)
        outcome_labels = np.array(outcome_labels)

        # Compute Area Under ROC Curve
        area_under_curve = round(roc_auc_score(outcome_labels, accumulative_scores, average='weighted',
                                               sample_weight=sample_weights if not subject else None), 3) \
            if not np.any(np.isnan(accumulative_scores)) else 0.5

        if auto_threshold:
            fpr, tpr, thresholds = roc_curve(outcome_labels, accumulative_scores,
                                             sample_weight=sample_weights if not subject else None)
            best_sens_spec = tpr + (1 - fpr)
            prediction_threshold = thresholds[np.argmax(best_sens_spec)]

        # compute some metrics for sake of code stability
        outcome_predictions = (accumulative_scores >= prediction_threshold).astype(int)

        cm = confusion_matrix(outcome_labels, outcome_predictions)
        tn, fp, fn, tp = cm.ravel()
        accuracy = round((tp + tn) / (tp + tn + fp + fn), 3)
        precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0
        recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0
        f1_score = round((2 * precision * recall) / (precision + recall), 3) if (precision + recall) > 0 else 0
        sensitivity = recall
        specificity = round(tn / (tn + fp), 3) if (tn + fp) > 0 else 0
        positive_predictive_value = precision
        negative_predictive_value = round(tn / (tn + fn), 3) if (tn + fn) > 0 else 0
        balanced_accuracy = round((sensitivity + specificity) / 2, 3)

        # Displaying the rounded metrics
        metrics_eval = {
            "acc": accuracy,
            "auc": area_under_curve,
            "cm": cm,
            "precision": precision,
            "recall": recall,
            "F1": f1_score,
            "sens": sensitivity,
            "spec": specificity,
            "ppv": positive_predictive_value,
            "npv": negative_predictive_value,
            "b_acc": balanced_accuracy
        }

        return metrics_eval
