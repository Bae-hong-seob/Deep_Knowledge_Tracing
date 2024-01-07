import os
import argparse
import pandas as pd

from args import parse_args
from src.utils import Setting, models_load
from src.data_preprocess.xgb_data import xgb_dataloader, xgb_preprocess_data,xgb_datasplit
from src.data_preprocess.lightgbm_data import lightgbm_dataloader, lightgbm_preprocess_data, lightgbm_datasplit
from src.data_preprocess.catboost_data import catboost_dataloader, catboost_preprocess_data,catboost_datasplit
from src.train import train, test

from autogluon.tabular import TabularDataset, TabularPredictor


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
        xgb_data = gbm_dataloader(args)
    elif args.model in ('LIGHTGBM'):
        data = lightgbm_preprocess_data(data)
    elif args.model in ('CATBOOST'):
        catboost_data = catboost_dataloader(args)
    else:
        pass
    
    ######################## Autogluon
    print(f'--------------- {args.autogluon} ---------------')
    if args.autogluon == True:
        train_data, label = data[data['answerCode'] != -1], "answerCode"
        
        print(f'--------------- TRAINING ---------------')
        predictor = TabularPredictor(label=label, eval_metric="roc_auc", problem_type="binary").fit(train_data, presets=["best_quality"])
        
        print(f'--------------- PREDICT ---------------')
        test_data = data[data.dataset == 2]
        test_data = test_data[test_data["userID"] != test_data["userID"].shift(-1)]
        test_data = test_data.drop(["answerCode"], axis=1)
        predicts = predictor.predict_proba(test_data)
        
        try:
            output = predictor.evaluate(test_data, silent=True)
            valid_auc = output["roc_auc"]
        except:
            valid_auc = 0.000
            
        os.makedirs(args.saved_model_path, exist_ok=True)
        with open(f'{args.saved_model_path}/{setting.save_time}_Autogluon_{valid_auc:.3f}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        

    ######################## Train/Valid Split
    else:
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
    print(f'--------------- SAVE PREDICT ---------------')
    filename = setting.get_submit_filename(args, valid_auc)
    submission = pd.read_csv(args.data_dir + 'sample_submission.csv')
    submission['prediction'] = predicts

    submission.to_csv(filename, index=False)
    print('make csv file !!! ', filename)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)