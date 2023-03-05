from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from glob import glob
from sklearn.metrics import (precision_score,
                             recall_score,
                             accuracy_score,
                             average_precision_score,
                             precision_recall_curve,
                             roc_auc_score,
                             roc_curve
                             )

sns.set_theme(style='whitegrid', font_scale=2.0)
THRESHOLD_DEPENDENT = ['accuracy_score', 'recall_score', 'precision_score']

import pdb

class BinaryEvaluator:

    def __init__(self, outdir, task_name=None):
        self.outdir = outdir
        self.task_name = task_name
        self.metrics = {
            'Prevalence': self.compute_prevalence,
            'Accuracy': accuracy_score,
            'Sensitivity': recall_score,
            'Specificity': recall_score,
            'Precision': precision_score,
            'AUROC': roc_auc_score,
            'Average precision': average_precision_score
        }
        os.makedirs(outdir, exist_ok=True)

    def __call__(self, labels, predictions):
        """
        Override in child classes to include functionality to pull labels and
        predictions from some outside source (ex: cosmos db)
        """
        self.get_performance_artifacts(labels, predictions)

    def get_performance_artifacts(self, labels, predictions, **kwargs):
        """
        Computes a suite of performance measures and saves artifacts
        """
        results = self.bootstrap_metric(labels, predictions, self.metrics)
        # results['Task'] = self.task_name
        results['N'] = str(len(labels))          
        with open(os.path.join(self.outdir, "metrics.json"), "w") as fp:
            json.dump(results, fp)

        fig, axs = plt.subplots(1, 2, figsize=(30, 10))
        self.plot_roc_curve(labels=labels,
                            predictions=predictions,
                            title='ROC Curve',
                            ax=axs[0])
        # self.plot_precision_recall(labels=labels,
        #                            predictions=predictions,
        #                            title='PR Curve',
        #                            ax=axs[1])
        self.plot_calibration_curve(labels=labels,
                                    predictions=predictions,
                                    title='Calibration Curve',
                                    ax=axs[1])
        plt.savefig(os.path.join(self.outdir, 'performance_curves.png'),
                    bbox_inches='tight',
                    dpi=300)

    def plot_roc_curve(self, labels, predictions, title, ax, color='black', label=None):
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        auc = roc_auc_score(labels, predictions)
        if label == None:
            label = f"AUC=%0.2f" % auc
        else:
            label = f"{label} AUC=%0.2f" % auc

        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2.0,
            label=label
        )
        ax.plot([0, 1], [0, 1], color="black", lw=1.0, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("1-Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(title)
        ax.legend(loc="lower right")

    def plot_precision_recall(self, labels, predictions, title, ax, label=None, color='black'):
        precision, recall, thresholds = precision_recall_curve(
            labels, predictions)
        auc = average_precision_score(labels, predictions)
        if label == None:
            pr_label = f"AUC=%0.2f" % auc
        else:
            pr_label = f"{label} AUC=%0.2f" % auc

        ax.plot(
            recall,
            precision,
            color=color,
            lw=2.0,
            label=pr_label
        )
        if label == None:
            baseline_label = f"Baseline AUC={round(np.mean(labels), 2)}"
        else:
            baseline_label = f"{label} Baseline AUC={round(np.mean(labels), 2)}"
        ax.plot([0, 1], [np.mean(labels), np.mean(labels)],
                color="black", lw=1.0, linestyle="--",
                label=baseline_label)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower right")

    def calibration_curve_ci(self, y_true, y_prob, sample_weight=None,
                             n_bins=5):
        """
        Adapted from sklearn but allows but bootstraps CI and sample weights
        """
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.searchsorted(bins[1:-1], y_prob)
        prob_trues, prob_preds = [], []
        inds = [i for i in range(len(y_prob))]
        df = pd.DataFrame(data={
            'bin_id': binids,
            'y_true': y_true,
            'sample_weight': sample_weight
        })
        for i in range(100):
            # Get bootstrapped sample (stratified by binid)
            df_boot = df.groupby('bin_id').sample(
                frac=1.0, replace=True).reset_index()
            if sample_weight is None:
                bin_sums = np.bincount(binids, weights=y_prob,
                                       minlength=len(bins))
                bin_true = np.bincount(df_boot['bin_id'].values,
                                       weights=df_boot['y_true'].values,
                                       minlength=len(bins))
                bin_total = np.bincount(binids, minlength=len(bins))

                nonzero = bin_total != 0
                prob_true = bin_true[nonzero] / bin_total[nonzero]
                prob_pred = bin_sums[nonzero] / bin_total[nonzero]

            else:
                bin_sums = np.bincount(binids, weights=y_prob,
                                       minlength=len(bins))
                bin_true = np.bincount(df_boot['bin_id'].values,
                                       weights=df_boot['y_true'].values *
                                       df_boot['sample_weight'].values,
                                       minlength=len(bins))
                bin_total = np.bincount(binids, minlength=len(bins))
                bin_total_true = np.bincount(
                    df_boot['bin_id'].values,
                    weights=df_boot['sample_weight'].values, minlength=len(
                        bins)
                )

                nonzero = bin_total != 0
                prob_true = bin_true[nonzero] / bin_total_true[nonzero]
                prob_pred = bin_sums[nonzero] / bin_total[nonzero]

            prob_trues.append(prob_true)
            prob_preds.append(prob_pred)

        return prob_trues, prob_preds

    def plot_calibration_curve(self, labels, predictions, title, ax, n_bins=5,
                               color='black', draw_baseline=True, label=None,
                               sample_weight=None):
        prob_trues, prob_preds = self.calibration_curve_ci(labels, predictions,
                                                           sample_weight=sample_weight,
                                                           n_bins=n_bins)
        prob_pred = np.mean(prob_preds, axis=0)
        prob_true = np.mean(prob_trues, axis=0)
        prob_true_lower = prob_true - np.percentile(prob_trues, 2.5, axis=0)
        prob_true_upper = np.percentile(prob_trues, 97.5, axis=0) - prob_true
        ax.plot(
            prob_pred,
            prob_true,
            color=color,
            label=label,
        )
        ax.scatter(
            prob_pred,
            prob_true,
            color=color,
        )

        ax.errorbar(prob_pred,
                    prob_true,
                    np.vstack((prob_true_lower, prob_true_upper)),
                    color=color,
                    linestyle='')

        if draw_baseline:
            ax.plot([0, 1], [0, 1],
                    color="black", lw=1.0,
                    label=f"Perfectly Calibrated")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(title)
        ax.legend(loc="lower right")

    def bootstrap_metric(self, labels, predictions, metrics, iters=1000,
                         threshold=0.5):
        """
        Compute metric and 95% confidence interal
        """
        predictions = np.asarray(predictions)
        predicted_labels = np.asarray(
            [1 if p >= threshold else 0 for p in predictions])
        labels = np.asarray(labels)
        inds = [i for i in range(len(predictions))]
        values = {}
        actual_values = {}
        for i in range(iters):
            inds_b = np.random.choice(inds, size=len(inds), replace=True)
            if len(np.unique(labels[inds_b])) < 2: # must have both labels
                continue
            l_b, p_b = labels[inds_b], predictions[inds_b]
            p_b_l = predicted_labels[inds_b]
            for m in metrics:
                if metrics[m].__name__ in THRESHOLD_DEPENDENT:
                    if m == 'Specificity':
                        values.setdefault(m, []).append(
                            metrics[m](l_b, p_b_l, pos_label=0))
                    else:
                        values.setdefault(m, []).append(metrics[m](l_b, p_b_l))
                else:
                    values.setdefault(m, []).append(metrics[m](l_b, p_b))

        for m in metrics:
            if metrics[m].__name__ in THRESHOLD_DEPENDENT:
                if m == 'Specificity':
                    actual_values[m] = metrics[m](
                        labels, predicted_labels, pos_label=0)
                else:
                    actual_values[m] = metrics[m](labels, predicted_labels)
            else:
                actual_values[m] = metrics[m](labels, predictions)
        results = {}
        for v in values:
            mean = '{:.2f}'.format(round(actual_values[v], 2))
            upper = '{:.2f}'.format(round(np.percentile(values[v], 97.5), 2))
            lower = '{:.2f}'.format(round(np.percentile(values[v], 2.5), 2))
            results[v] = f"{mean} [{lower}, {upper}]"

        return results

    def compute_prevalence(self, labels, predictions=None):
        """
        Allows predictions so that I don't have to call funtion differently
        than other metrics but of course doesn't use 
        """
        return np.mean(labels)


class BinaryGroupEvaluator(BinaryEvaluator):
    
    def __init__(self, outdir, task_name=None):
        super().__init__(outdir, task_name)
    
    def __call__(self, df):
        """
        Args:
            df : pandas DataFrame containing the following columns
                label : label of example
                prediction : predictin for example
                group : group memborship
        Notes:
            Examples can belong to multiple groups, df is long format
        """
        fig, axs = plt.subplots(1, 2, figsize=(30, 10))
        colors = sns.color_palette(n_colors=df.group.nunique())
        draw_baseline=True
        elligible_groups = set(['race_White', 'race_Asian', 'race_Black', 
            'race_Pacific Islander', 'race_Native American'])
        for i, group in enumerate(df.groupby('group')):
            # Only compute performance artifacts if we have > 20 labels and
            # have both positive and negative examples
            if len(group[1]) < 20 or group[1].label.nunique() < 2 or group[0] not in elligible_groups:
                print(f"Not stratifying by {group[0]}")
            else: 
                self.get_performance_artifacts(
                    group[1].label, 
                    group[1].prediction,
                    group=group[0],
                    color=colors[i],
                    axs=axs,
                    draw_baseline=draw_baseline
                )
                draw_baseline=False
        plt.savefig(os.path.join(self.outdir, 'group_performance_curves.png'),
            bbox_inches='tight',
            dpi=300)

    def get_performance_artifacts(self, labels, predictions, group, color, axs, draw_baseline):
        """
        Computes a suite of performance measures and saves artifacts
        """
        results = self.bootstrap_metric(labels, predictions, self.metrics)
        results['Group'] = group
        results['N'] = str(len(labels))          
        with open(os.path.join(self.outdir, f"{group}_metrics.json"), "w") as fp:
            json.dump(results, fp)
        self.plot_roc_curve(labels=labels,
                            predictions=predictions,
                            title='ROC Curve',
                            ax=axs[0],
                            label=group,
                            color=color)
        # self.plot_precision_recall(labels=labels,
        #                            predictions=predictions,
        #                            title='PR Curve',
        #                            ax=axs[1],
        #                            label=group,
        #                            color=color)
        self.plot_calibration_curve(labels=labels,
                                    predictions=predictions,
                                    title='Calibration Curve',
                                    ax=axs[1],
                                    label=group,
                                    color=color,
                                    draw_baseline=draw_baseline)