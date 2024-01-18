# # Feature Selection
# FEATS = [
#     #'userID', # 추가 시 valid 증가, public 감소. 과적합 의심
#     "KnowledgeTag",
#     'solve_count_per_test', 'number_of_users_per_test', 
#     #'not_solving_count_per_test', 'problem_solving_rate_per_test', # 시험지 별 안 푼 문제 개수, 문제를 푼 비율 추가 시 -> valid score감소, public score 감소
#     "answerRate_per_tag", "answerCount_per_tag", "answerVar_per_tag", "answerStd_per_tag",
#     "tag_count",
#     "mean_elp_tag_all_mean", 
#     #"mean_elp_tag_all_sum", "mean_elp_tag_all_var", "mean_elp_tag_all_std",
#     "mean_elp_tag_o_mean", 
#     #"mean_elp_tag_o_sum", "mean_elp_tag_o_var", "mean_elp_tag_o_std",
#     "mean_elp_tag_x_mean", 
#     #"mean_elp_tag_x_sum", "mean_elp_tag_x_var", "mean_elp_tag_x_std",
#     "answerRate_per_test", "answerCount_per_test", "answerVar_per_test", "answerStd_per_test",
#     "cum_answerRate_per_user",
#     "problem_correct_per_user",
#     "problem_solved_per_user",
#     "mean_elp_ass_all_mean", 
#     #"mean_elp_ass_all_sum", "mean_elp_ass_all_var", "mean_elp_ass_all_std",
#     "mean_elp_ass_o_mean", 
#     #"mean_elp_ass_o_sum", "mean_elp_ass_o_var", "mean_elp_ass_o_std",
#     "mean_elp_ass_x_mean", 
#     #"mean_elp_ass_x_sum", "mean_elp_ass_x_var", "mean_elp_ass_x_std",
#     "answerRate_per_ass", "answerCount_per_ass", "answerVar_per_ass", "answerStd_per_ass",
#     "elapsed",
#     'elapsed_shift',
#     "category",
#     "acc_answerRate_per_cat",
#     "acc_count_per_cat",
#     "acc_elapsed_per_cat",
#     "correct_answer_per_cat",
#     "test_number",
#     "mean_elp_pnum_all_mean", 
#     #"mean_elp_pnum_all_sum", "mean_elp_pnum_all_var", "mean_elp_pnum_all_std",
#     "mean_elp_pnum_o_mean", 
#     #"mean_elp_pnum_o_sum", "mean_elp_pnum_o_var", "mean_elp_pnum_o_std",
#     "mean_elp_pnum_x_mean", 
#     #"mean_elp_pnum_x_sum", "mean_elp_pnum_x_var", "mean_elp_pnum_x_std",
#     "acc_tag_count_per_user",
#     "problem_count",
#     "problem_number",
#     "answerRate_per_pnum", "answerCount_per_pnum", "answerVar_per_pnum", "answerStd_per_pnum",
#     "problem_position",
#     'timeDelta_userAverage',
#     'timestep_1', 'timestep_2', 'timestep_3', 'timestep_4', 'timestep_5', #제거 시 valid, public score 모두 감소
#     "median_elapsed_wrong_users", "median_elapsed_correct_users",
#     #"mean_elapsed_wrong_users", "mean_elapsed_correct_users",
#     #"cum_answerRate_per_user", "problem_correct_per_user", "problem_solved_per_user",
#     #'correct_mean_per_user', 'correct_var_per_user', 'correct_std_per_user',
#     #'number_of_sloved_per_tag', 'correct_mean_per_tag', 'correct_var_per_tag', 'correct_std_per_tag',
# ]

######################## SELECT FEATURE

FEATURE = [
    "userID",
    "assessmentItemID",
    "KnowledgeTag",
    "elapsed",
    "category_high",
    "problem_num",
    "cum_answerRate_per_user",
    "acc_elapsed_per_user",
    "problem_correct_per_user",
    "problem_solved_per_user",
    "correct_answer_per_cat",
    "acc_count_per_cat",
    "acc_answerRate_per_cat",
    "timeDelta_userAverage",
    "timestep_1",
    "timestep_2",
    "timestep_3",
    "timestep_4",
    "timestep_5",
    "hour",
    "weekofyear",
    "problem_correct_per_woy",
    "problem_solved_per_woy",
    "cum_answerRate_per_woy",
]
FEATURE_USER = [
    "answerRate_per_user",
    "answer_cnt_per_user",
    "elapsed_time_median_per_user",
    "assessment_solved_per_user",
]
FEATURE_ITEM = [
    "answerRate_per_item",
    "answer_cnt_per_item",
    "elapsed_time_median_per_item",
    "wrong_users_median_elapsed",
    "correct_users_median_elapsed",
]
FEATURE_TAG = ["tag_exposed", "answerRate_per_tag",
               #"acc_tag_count_per_user" #valid 증가(과적합 예상)
               ]
FEATURE_TEST = ["elapsed_median_per_test", "answerRate_per_test",
    'solve_count_per_test', 'number_of_users_per_test', #valid 그대로, acc만 감소
    'problem_count', 'tag_count', 'problem_position', #public 상승.
]
FEATURE_CAT = ["elapsed_median_per_cat", "answerRate_per_cat",
               #"acc_elapsed_per_cat", #valid 증가(과적합 에상)
               ]
FEATURE_PROBLEM_NUM = [
    "elapsed_median_per_problem_num",
    "answerRate_per_problem_num",
]

FEATURE_ELO = ["elo_assessment", "elo_test", "elo_tag"]

FEATURE += FEATURE_USER
FEATURE += FEATURE_ITEM
FEATURE += FEATURE_TAG
FEATURE += FEATURE_TEST
FEATURE += FEATURE_CAT
FEATURE += FEATURE_PROBLEM_NUM
FEATURE += FEATURE_ELO
FEATS = FEATURE

import os
import pickle
import numpy as np
import pandas as pd

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
            if args.feature_selection:
                model.fit(X=x_train, y=y_train, eval_set=[(x_valid, y_valid)], eval_metric="auc")
                valid_auc, valid_acc = valid(args, model, x_valid, y_valid)
                
                feature_importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': model.feature_importances_})
                
            else:
                print(f'use FEATS feature -> x_train: {x_train[FEATS].shape}, y_train: {y_train.shape}, x_valid: {x_valid[FEATS].shape}, y_valid: {y_valid.shape}')
                model.fit(X=x_train[FEATS], y=y_train, eval_set=[(x_valid[FEATS], y_valid)], eval_metric="auc")
                valid_auc, valid_acc = valid(args, model, x_valid[FEATS], y_valid)
                
                feature_importance_df = pd.DataFrame({'Feature': x_train[FEATS].columns, 'Importance': model.feature_importances_})
            
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            feature_importance_df.to_csv(f'./record/{setting.save_time}_{args.model}_feature_importance.csv', index=False)
            
            os.makedirs(args.saved_model_path, exist_ok=True)
            with open(f'{args.saved_model_path}/{setting.save_time}_{args.model}_{valid_auc:.4f}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
                    
            print(f"VALID AUC : {valid_auc} VALID ACC : {valid_acc}\n")
        
    return model, valid_auc, valid_acc

def valid(args, model, x_valid, y_valid):
    #  LGBoost 모델 추론
    preds = model.predict_proba(x_valid)[:, 1]
    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)
 
    return auc, acc

def test(args, model, test_df):
    if args.feature_selection:
        probs = model.predict_proba(test_df)[:,-1]
    else:
        probs = model.predict_proba(test_df[FEATS])[:,-1]
        
    return probs