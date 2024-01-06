import os
import argparse
import pandas as pd

from args import parse_args
from src.utils import Setting, models_load
from src.data_preprocess.xgb_data import xgb_dataloader, xgb_preprocess_data,xgb_datasplit
from src.data_preprocess.lightgbm_data import lightgbm_dataloader, lightgbm_preprocess_data, lightgbm_datasplit
from src.data_preprocess.catboost_data import catboost_dataloader, catboost_preprocess_data,catboost_datasplit
from src.train import train, test

def main(args):
    ####################### Setting for Log
    setting = Setting()
    
    ######################## DATA LOAD
    print(f'--------------- {args.model} Load Data ---------------')
    if args.model in ('XGB'):
        data = gbm_dataloader(args)
    elif args.model in ('LIGHTGBM'):
        data = lightgbm_dataloader(args)
    elif args.model in ('CATBOOST'):
        data = catboost_dataloader(args)
    else:
        pass
    
    
    ######################## DATA PREPROCESS
    print(f'--------------- {args.model} Data PREPROCESSING---------------')
    if args.model in ('XGB'):
        gbm_data = gbm_dataloader(args)
    elif args.model in ('LIGHTGBM'):
        data = lightgbm_preprocess_data(data)
    elif args.model in ('CATBOOST'):
        catboost_data = catboost_dataloader(args)
    else:
        pass
    
    
    ######################## Train/Valid Split
    print(f'--------------- {args.model} Train/Valid Split ---------------')
    if args.model in ('XGB'):
        x_train, y_train, x_valid, y_valid = xgb_datasplit(data)
    elif args.model in ('LIGHTGBM'):
        x_train, y_train, x_valid, y_valid = lightgbm_datasplit(args, data)
    elif args.model in ('CATBOOST'):
        x_train, y_train, x_valid, y_valid = catboost_datasplit(data)
    else:
        pass
    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}, x_valid: {x_valid.shape}, y_valid: {y_valid.shape}')
    
    
    ######################## MODEL LOAD
    print(f'--------------- {args.model} MODEL LOAD---------------')
    model = models_load(args)
    
    
    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    model, valid_auc = train(args, model, x_train, y_train, x_valid, y_valid, setting)


    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    predicts = test(args, model, data)


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    filename = setting.get_submit_filename(args, valid_auc)
    
    write_path = os.path.join(filename)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(predicts):
            w.write("{},{}\n".format(id, p))
        
    submission = pd.read_csv(args.data_dir + 'sample_submission.csv')
    submission['prediction'] = predicts

    submission.to_csv(filename, index=False)
    print('make csv file !!! ', filename)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)