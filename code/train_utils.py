import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
from models import vgae_loss_function, MLPClassifier
from data_utils import prepare_inputs
from metrics import evaluate_classifier, TestOutput

def train_epoch_mlp(model, classifier, fold_data, optimizer, opt, device):
    concat_mi_tensor, concat_dis_tensor, G_mi, G_dis = prepare_inputs(fold_data, device)
    pos_pairs = fold_data[2][0].to(device)
    neg_pairs = fold_data[2][1].to(device)
    desired_neg_num = int(pos_pairs.size(0) * opt.pos_neg_ratio)
    if neg_pairs.size(0) > desired_neg_num:
        perm = torch.randperm(neg_pairs.size(0))[:desired_neg_num]
        neg_pairs = neg_pairs[perm]

    bce_loss = nn.BCEWithLogitsLoss()
    bce_loss_gan = nn.BCELoss()
    lambda_gan = 0.1
    lambda_vgae = 0.1

    epoch_iterator = tqdm(range(1, opt.epoch + 1), desc="Training (MLP)", ncols=100)
    start_time = time.time()
    for epoch in epoch_iterator:
        model.train()
        classifier.train()
        optimizer.zero_grad()
        x, y, fake_graph, real_graph, real_disc, fake_disc, recon_mi, mu_mi, logvar_mi, recon_ci, mu_ci, logvar_ci = model(concat_mi_tensor, concat_ci_tensor, G_mi, G_ci)
        pos_feat = torch.cat([x[pos_pairs[:, 0]], y[pos_pairs[:, 1]]], dim=1)
        neg_feat = torch.cat([x[neg_pairs[:, 0]], y[neg_pairs[:, 1]]], dim=1)
        train_features = torch.cat([pos_feat, neg_feat], dim=0)
        train_labels = torch.cat([torch.ones(pos_feat.size(0), 1),
                                  torch.zeros(neg_feat.size(0), 1)], dim=0).to(device)
        pred = classifier(train_features)
        loss_cls = bce_loss(pred, train_labels)
        real_target = torch.ones_like(real_disc)
        fake_target = torch.zeros_like(fake_disc)
        disc_loss = bce_loss_gan(real_disc, real_target) + bce_loss_gan(fake_disc, fake_target)
        gen_loss = bce_loss_gan(fake_disc, real_target)
        loss_gan = disc_loss + gen_loss
        loss_vgae = vgae_loss_function(recon_mi, G_mi, mu_mi, logvar_mi) + vgae_loss_function(recon_ci, G_ci, mu_ci, logvar_ci)
        loss = loss_cls + lambda_gan * loss_gan + lambda_vgae * loss_vgae
        loss.backward()
        optimizer.step()
        epoch_iterator.set_postfix({'Loss': loss.item()})


    test_pos_pairs = fold_data[3][0].to(device)
    test_neg_pairs = fold_data[3][1].to(device)
    desired_test_neg_num = int(test_pos_pairs.size(0) * opt.pos_neg_ratio)
    if test_neg_pairs.size(0) > desired_test_neg_num:
        perm = torch.randperm(test_neg_pairs.size(0))[:desired_test_neg_num]
        test_neg_pairs = test_neg_pairs[perm]

    model.eval()
    classifier.eval()
    with torch.no_grad():
        x, y, _, _, _, _, _, _, _, _, _, _ = model(concat_mi_tensor, concat_ci_tensor, G_mi, G_ci)
        pos_feat_test = torch.cat([x[test_pos_pairs[:, 0]], y[test_pos_pairs[:, 1]]], dim=1)
        neg_feat_test = torch.cat([x[test_neg_pairs[:, 0]], y[test_neg_pairs[:, 1]]], dim=1)
        test_features = torch.cat([pos_feat_test, neg_feat_test], dim=0)
        test_labels = torch.cat([torch.ones(pos_feat_test.size(0), 1),
                                 torch.zeros(neg_feat_test.size(0), 1)], dim=0).to(device)
        test_pred = classifier(test_features)
    end_time = time.time()
    fold_time = end_time - start_time
    print(f"Fold time: {fold_time:.2f} seconds")

    return test_labels, test_pred, model, classifier, test_features


def train_epoch_external(model, fold_data, optimizer, opt, classifier_type, classifier_head, device):
    concat_mi_tensor, concat_ci_tensor, G_mi, G_ci = prepare_inputs(fold_data, device)

    pos_pairs = fold_data[2][0].to(device)
    neg_pairs = fold_data[2][1].to(device)

    desired_neg_num = int(pos_pairs.size(0) * opt.pos_neg_ratio)
    if neg_pairs.size(0) > desired_neg_num:
        perm = torch.randperm(neg_pairs.size(0))[:desired_neg_num]
        neg_pairs = neg_pairs[perm]

    bce_loss = nn.BCEWithLogitsLoss()
    bce_loss_gan = nn.BCELoss()
    lambda_gan = 0.1
    lambda_vgae = 0.1

    epoch_iterator = tqdm(range(1, opt.epoch + 1), desc="Training Model", ncols=100)
    start_time = time.time()
    for epoch in epoch_iterator:
        model.train()
        classifier_head.train()
        optimizer.zero_grad()
        x, y, fake_graph, real_graph, real_disc, fake_disc, recon_mi, mu_mi, logvar_mi, recon_ci, mu_ci, logvar_ci = model(concat_mi_tensor, concat_ci_tensor, G_mi, G_ci)
        pos_feat = torch.cat([x[pos_pairs[:, 0]], y[pos_pairs[:, 1]]], dim=1)
        neg_feat = torch.cat([x[neg_pairs[:, 0]], y[neg_pairs[:, 1]]], dim=1)
        train_features = torch.cat([pos_feat, neg_feat], dim=0)
        train_labels = torch.cat([torch.ones(pos_feat.size(0), 1),
                                  torch.zeros(neg_feat.size(0), 1)], dim=0).to(device)
        pred = classifier_head(train_features)
        loss_cls = bce_loss(pred, train_labels)

        real_target = torch.ones_like(real_disc)
        fake_target = torch.zeros_like(fake_disc)
        disc_loss = bce_loss_gan(real_disc, real_target) + bce_loss_gan(fake_disc, fake_target)
        gen_loss = bce_loss_gan(fake_disc, real_target)
        loss_gan = disc_loss + gen_loss

        loss_vgae = vgae_loss_function(recon_mi, G_mi, mu_mi, logvar_mi) + vgae_loss_function(recon_ci, G_ci, mu_ci, logvar_ci)

        loss = loss_cls + lambda_gan * loss_gan + lambda_vgae * loss_vgae
        loss.backward()
        optimizer.step()
        epoch_iterator.set_postfix({'Loss': loss.item()})

    model.eval()
    classifier_head.eval()
    with torch.no_grad():
        x, y, _, _, _, _, _, _, _, _, _, _ = model(concat_mi_tensor, concat_ci_tensor, G_mi, G_ci)
        pos_feat = torch.cat([x[pos_pairs[:, 0]], y[pos_pairs[:, 1]]], dim=1)
        neg_feat = torch.cat([x[neg_pairs[:, 0]], y[neg_pairs[:, 1]]], dim=1)
        train_features = torch.cat([pos_feat, neg_feat], dim=0).cpu().numpy()
        train_labels = np.concatenate([np.ones(pos_feat.size(0)), np.zeros(neg_feat.size(0))])
        test_pos_pairs = fold_data[3][0].to(device)
        test_neg_pairs = fold_data[3][1].to(device)
        desired_test_neg_num = int(test_pos_pairs.size(0) * opt.pos_neg_ratio)
        if test_neg_pairs.size(0) > desired_test_neg_num:
            perm = torch.randperm(test_neg_pairs.size(0))[:desired_test_neg_num]
            test_neg_pairs = test_neg_pairs[perm]
        pos_feat_test = torch.cat([x[test_pos_pairs[:, 0]], y[test_pos_pairs[:, 1]]], dim=1)
        neg_feat_test = torch.cat([x[test_neg_pairs[:, 0]], y[test_neg_pairs[:, 1]]], dim=1)
        test_features = torch.cat([pos_feat_test, neg_feat_test], dim=0).cpu().numpy()
        test_labels = np.concatenate([np.ones(pos_feat_test.size(0)), np.zeros(neg_feat_test.size(0))])
    end_time = time.time()
    fold_time = end_time - start_time
    print(f"Fold time: {fold_time:.2f} seconds")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)


    if classifier_type == 'lgbm':
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            num_leaves=15,
            learning_rate=0.01,
            n_estimators=500,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=42
        )


    elif classifier_type == 'stacking':
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        estimators = [('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                      ('rf', RandomForestClassifier(class_weight='balanced'))]
        clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced'))
    clf.fit(train_features, train_labels)
    test_pred = clf.predict_proba(test_features)[:, 1]
    return test_labels, test_pred, model, clf, test_features
