import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run Model with Classifier for Association Prediction.")
    parser.add_argument('--data_path', type=str, default='../data/CMI-9905')
    parser.add_argument('--validation', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--mi_num', type=int, default=962)
    parser.add_argument('--ci_num', type=int, default=2346)
    parser.add_argument('--alpha', type=float, default=0.11)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=5)
    parser.add_argument('--nmodal', type=int, default=2)

    parser.add_argument('--classifier', type=str, default='lgbm', choices=['lgbm'])
    parser.add_argument('--pos_neg_ratio', type=float, default=1)
    return parser.parse_args()
