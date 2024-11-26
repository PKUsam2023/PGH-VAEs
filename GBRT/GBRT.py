from model import *
import argparse
import sys
import logging
from getdata import returndata

def GBRT_main(args):
    learning_rate=args.learning_rate
    n_estimators=args.n_estimators
    subsample=args.subsample
    min_samples_split=args.min_samples_split
    max_depth=args.max_depth

    x_train, y_train, x_valid, y_valid = returndata('all')

    MAEx, MSEx, RMSEx, MAE, MSE, RMSE = GBRT_model(x_train, y_train, x_valid, y_valid, learning_rate, n_estimators, subsample, min_samples_split, max_depth)
    print(str(MAE) + ' ' + str(MSE) + ' ' + str(RMSE) + ' ' + str(MAEx) + ' ' + str(MSEx) + ' ' + str(RMSEx))
    logging.basicConfig(filename='logger.log', level=logging.INFO)

    logging.info(f'''learning_rate: {learning_rate}, n_estimators: {n_estimators}, subsample: {subsample}, min_samples_split: {min_samples_split}, max_depth: {max_depth}, MAE: {MAE}, MSE: {MSE}, RMSE: {RMSE}, MAE: {MAEx}, MSE: {MSEx}, RMSE: {RMSEx}''')

def parse_args(args):
    parser = argparse.ArgumentParser(description="parameter")
    parser.add_argument('--learning_rate', default=0.07, type=float)
    parser.add_argument('--n_estimators', default=1300, type=int)
    parser.add_argument('--subsample', default=0.9, type=float)
    parser.add_argument('--min_samples_split', default=15, type=int)
    parser.add_argument('--max_depth', default=1, type=int)
    args = parser.parse_args()
    return args

def do_main():
    args = parse_args(sys.argv[1:])
    GBRT_main(args)

if __name__ == "__main__":
    do_main()
