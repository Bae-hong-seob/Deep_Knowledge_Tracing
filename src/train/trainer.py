# Feature Selection
FEATS = [
    "KnowledgeTag",
    'length_per_test', 'number_of_users_per_test',
    "answerRate_per_tag", "answerCount_per_tag", "answerVar_per_tag", "answerStd_per_tag",
    "tag_count",
    "mean_elp_tag_all",
    "mean_elp_tag_o",
    "mean_elp_tag_x",
    "answerRate_per_test", "answerCount_per_test", "answerVar_per_test", "answerStd_per_test",
    "cum_answerRate_per_user",
    "problem_correct_per_user",
    "problem_solved_per_user",
    "mean_elp_ass_all",
    "mean_elp_ass_o",
    "mean_elp_ass_x",
    "answerRate_per_ass", "answerCount_per_ass", "answerVar_per_ass", "answerStd_per_ass",
    "elapsed",
    'elapsed_shift',
    "category",
    "acc_answerRate_per_cat",
    "acc_count_per_cat",
    "acc_elapsed_per_cat",
    "correct_answer_per_cat",
    "test_number",
    "mean_elp_pnum_all",
    "mean_elp_pnum_o",
    "mean_elp_pnum_x",
    "acc_tag_count_per_user",
    "problem_count",
    "problem_number",
    "answerRate_per_pnum", "answerCount_per_pnum", "answerVar_per_pnum", "answerStd_per_pnum",
    "problem_position",
    'timeDelta_userAverage',
    'timestep_1', 'timestep_2', 'timestep_3', 'timestep_4', 'timestep_5',
    "median_elapsed_wrong_users", "median_elapsed_correct_users"
]

import os
import pickle
import numpy as np

from sklearn.metrics import roc_auc_score, accuracy_score

def train(args, model, x_train, y_train, x_valid, y_valid, setting):    
    if args.model in ('XGB', 'CATBOOST'):
        model.fit(x_train,y_train, eval_set=[(x_valid, y_valid)])
        valid_auc, valid_acc = valid(args, model, x_train, y_train, x_valid, y_valid)
        
        os.makedirs(args.saved_model_path, exist_ok=True)
        model.save_model(f'{args.saved_model_path}/{setting.save_time}_{args.model}_{valid_auc:.3f}_model.json')
                
        
    elif args.model in ('LIGHTGBM'):
        if args.optuna == True:
            study = optuna.create_study(direction='minimize', study_name='LGBM Regressor')
            func = lambda trial : objective(trial, args, dataloader, model, setting)
            study.optimize(func, n_trials=20)
            
            print(f"\tBest value (rmse): {study.best_value:.5f}")
            print(f"\tBest params:")

            for key, value in study.best_params.items():
                print(f"\t\t{key}: {value}")
                
        else:
            model.fit(X=x_train[FEATS], y=y_train, eval_set=[(x_valid[FEATS], y_valid)], eval_metric="auc"
                      #, verbose=100
                      )
            valid_auc, valid_acc = valid(args, model, x_valid, y_valid)
            
            os.makedirs(args.saved_model_path, exist_ok=True)
            with open(f'{args.saved_model_path}/{setting.save_time}_{args.model}_{valid_auc:.4f}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
                    
            print(f"VALID AUC : {valid_auc} VALID ACC : {valid_acc}\n")
        
    return model, valid_auc

def valid(args, model, x_valid, y_valid):
    #  LGBoost 모델 추론
    preds = model.predict_proba(x_valid[FEATS])[:, 1]
    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)
 
    return auc, acc

def test(args, model, df):
    
    test_df = df[df.dataset == 2]
    # LEAVE LAST INTERACTION ONLY
    test_df = test_df[test_df["userID"] != test_df["userID"].shift(-1)]

    # DROP ANSWERCODE
    test_df = test_df.drop(["answerCode"], axis=1)

    # MAKE PREDICTION
    probs = model.predict_proba(test_df[FEATS])[:,-1]
        
    return probs