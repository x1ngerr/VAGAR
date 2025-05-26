from __future__ import division
import os
import warnings
import torch
import torch.optim as optim
import numpy as np
import time
from tqdm import tqdm

from config import parameter_parser
from data_utils import prepare_data, Dataset, seed_torch
from models import VAGAR, MLPClassifier
from train_utils import train_epoch_mlp, train_epoch_external
from metrics import evaluate_classifier, TestOutput

def main(opt):
    dataset = prepare_data(opt)
    train_data = Dataset(opt, dataset)
    classifier_list = ['lgbm']

    results_all = {}
    print(f"\nstart {opt.validation} fold CV")
    for classifier_type in classifier_list:
        print(f"\n====== Evaluating classifier: {classifier_type} ======")
        fold_auc_list = []
        fold_aupr_list = []
        fold_f1_list = []
        fold_accuracy_list = []
        fold_recall_list = []
        fold_specificity_list = []
        fold_precision_list = []
        fold_mcc_list = []
        for i in range(opt.validation):
            print(f"\n===== Fold {i + 1} start: {classifier_type} =====")
            train_pos_count = train_data[i][2][0].size(0)
            train_neg_count = train_data[i][2][1].size(0)
            test_pos_count = train_data[i][3][0].size(0)
            test_neg_count = train_data[i][3][1].size(0)


            hidden_list = [128, 128]
            model = VAGAR(opt.mi_num, opt.ci_num, hidden_list, opt)
            model.to(device)

            classifier_head = model.classifier_head
            optimizer = optim.Adam(list(model.parameters()) + list(classifier_head.parameters()), lr=0.0001)
            start_time = time.time()
            test_labels_np, test_pred_np, model, clf, test_features = train_epoch_external(model, train_data[i], optimizer, opt, classifier_type, classifier_head, device)
            end_time = time.time()
            fold_time = end_time - start_time
            print(f"Fold time: {fold_time:.2f} seconds")

            auc_value, aupr_value, f1_value, accuracy_value, recall_value, specificity_value, precision_value, mcc_value = evaluate_classifier(test_labels_np, test_pred_np)
            fold_auc_list.append(auc_value)
            fold_aupr_list.append(aupr_value)
            fold_f1_list.append(f1_value)
            fold_accuracy_list.append(accuracy_value)
            fold_recall_list.append(recall_value)
            fold_specificity_list.append(specificity_value)
            fold_precision_list.append(precision_value)
            fold_mcc_list.append(mcc_value)

            print(f"Fold {i + 1} Results: AUC = {auc_value:.4f}, AUPR = {aupr_value:.4f}, F1 = {f1_value:.4f}, "
                  f"Accuracy = {accuracy_value:.4f}, Recall (Sen) = {recall_value:.4f}, Specificity = {specificity_value:.4f}, "
                  f"Precision = {precision_value:.4f}, MCC = {mcc_value:.4f}")

            TestOutput(clf, classifier_type, test_features, test_labels_np, i + 1, device)

        metrics_cross_avg = np.array([np.mean(fold_auc_list), np.mean(fold_aupr_list), np.mean(fold_f1_list),
                                      np.mean(fold_accuracy_list), np.mean(fold_recall_list),
                                      np.mean(fold_specificity_list), np.mean(fold_precision_list), np.mean(fold_mcc_list)])
        print("\n====== {} all fold result ======".format(classifier_type))
        for idx in range(opt.validation):
            print(f"Fold {idx + 1}: AUC = {fold_auc_list[idx]:.4f}, AUPR = {fold_aupr_list[idx]:.4f}, "
                  f"F1 = {fold_f1_list[idx]:.4f}, Accuracy = {fold_accuracy_list[idx]:.4f}, "
                  f"Recall (Sen) = {fold_recall_list[idx]:.4f}, Specificity = {fold_specificity_list[idx]:.4f}, "
                  f"Precision = {fold_precision_list[idx]:.4f}, MCC = {fold_mcc_list[idx]:.4f}")
        print('\nAverage indicator results:')
        print(f"AUC: {metrics_cross_avg[0]:.4f}, AUPR: {metrics_cross_avg[1]:.4f}, "
              f"F1: {metrics_cross_avg[2]:.4f}, Accuracy: {metrics_cross_avg[3]:.4f}, "
              f"Recall (Sen): {metrics_cross_avg[4]:.4f}, Specificity: {metrics_cross_avg[5]:.4f}, "
              f"Precision: {metrics_cross_avg[6]:.4f}, MCC: {metrics_cross_avg[7]:.4f}")
        results_all[classifier_type] = metrics_cross_avg

    print("\n====== all results ======")
    for clf in classifier_list:
        m = results_all[clf]
        print(f"{clf:10s}: AUC={m[0]:.4f}, AUPR={m[1]:.4f}, "
              f"Accuracy={m[3]:.4f}, Sen={m[4]:.4f}, Specificity={m[5]:.4f}, Precision={m[6]:.4f}, MCC={m[7]:.4f}, F1={m[2]:.4f}, ")


class Metric_fun(object):
    def __init__(self):
        super(Metric_fun, self).__init__()

    def cv_mat_model_evaluate(self, association_mat, predict_mat):
        real_score = np.mat(association_mat.detach().cpu().numpy().flatten())
        predict_score = np.mat(predict_mat.detach().cpu().numpy().flatten())
        return self.get_metrics(real_score, predict_score)

    def get_metrics(self, real_score, predict_score):
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[
            (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]
        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1
        TP = predict_score_matrix * real_score.T
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN
        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T
        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])
        recall_list = tpr
        precision_list = TP / (TP + FP)
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
        PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])
        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)
        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index, 0]
        accuracy = accuracy_list[max_index, 0]
        specificity = specificity_list[max_index, 0]
        recall = recall_list[max_index, 0]
        precision = precision_list[max_index, 0]
        return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision]


if __name__ == '__main__':
    args = parameter_parser()
    warnings.filterwarnings('ignore')
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
