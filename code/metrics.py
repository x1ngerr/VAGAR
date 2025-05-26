import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, recall_score, \
    precision_score, confusion_matrix, roc_curve, precision_recall_curve


def evaluate_classifier(true_labels, pred_scores):
    auc = roc_auc_score(true_labels, pred_scores)
    aupr = average_precision_score(true_labels, pred_scores)
    pred_labels = (pred_scores >= 0.5).astype(int)
    f1 = f1_score(true_labels, pred_labels)
    accuracy = accuracy_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_den if mcc_den != 0 else 0
    return auc, aupr, f1, accuracy, recall, specificity, precision, mcc


def validate_and_predict(classifier, X_test, device):
    if hasattr(classifier, "eval"):
        classifier.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            outputs = classifier(X_test_tensor)
            if isinstance(outputs, torch.Tensor):
                outputs = torch.sigmoid(outputs).cpu().numpy()
            if outputs.ndim == 1 or outputs.shape[1] == 1:
                outputs = np.hstack([1 - outputs, outputs])
        return outputs
    else:
        return classifier.predict_proba(X_test)


def TestOutput(classifier, name, X_test, y_test, fold_index, device):
    from tqdm import tqdm
    from data_utils import StorFile
    import os

    if isinstance(X_test, torch.Tensor):
        X_test = X_test.detach().cpu().numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.detach().cpu().numpy()
    X_test = np.array(X_test)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    if np.isnan(X_test).any():
        raise ValueError("X_test contains NaN values. Please check the data preprocessing.")

    ModelTestOutput = validate_and_predict(classifier, X_test, device)
    LabelPredictionProb = []
    LabelPrediction = []

    for counter in tqdm(range(len(y_test)), desc=f"Testing {name} output"):
        prob_positive = ModelTestOutput[counter][1]
        real_label = y_test[counter]
        LabelPredictionProb.append([real_label, prob_positive])
        pred_label = 1 if prob_positive > 0.5 else 0
        LabelPrediction.append([real_label, pred_label])

    if not os.path.exists('results/CMI-9905'):
        os.makedirs('results/CMI-9905')
    StorFile(LabelPredictionProb, f"results/CMI-9905/{name}RealAndPredictionProbA+B_fold_{fold_index}.csv")
    StorFile(LabelPrediction, f"results/CMI-9905/{name}RealAndPredictionA+B_fold_{fold_index}.csv")

    aupr = average_precision_score(y_test, ModelTestOutput[:, 1])

    return aupr

