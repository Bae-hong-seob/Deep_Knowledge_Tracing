import os
import argparse
import pickle
import numpy as np
import pandas as pd

from args import parse_args
from src.utils import Setting, models_load
from src.data_preprocess.lightgbm_data import lightgbm_dataloader, lightgbm_preprocess_data, lightgbm_datasplit
from src.train import train, test

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectFromModel


from autogluon.tabular import TabularDataset, TabularPredictor

# Feature Selection
FEATS = [ # best feature 52개 + private에서 좋았던 2개 = 54개
    "KnowledgeTag",
    'solve_count_per_test', 'number_of_users_per_test', 
    'not_solving_count_per_test', 'problem_solving_rate_per_test', # 시험지 별 안 푼 문제 개수, 문제를 푼 비율 추가 시 -> valid score감소, public score 감소
    "answerRate_per_tag", "answerCount_per_tag", "answerVar_per_tag", "answerStd_per_tag",
    "tag_count",
    "mean_elp_tag_all_mean", 
    "mean_elp_tag_o_mean", 
    "mean_elp_tag_x_mean", 
    "answerRate_per_test", "answerCount_per_test", "answerVar_per_test", "answerStd_per_test",
    "cum_answerRate_per_user",
    "problem_correct_per_user",
    "problem_solved_per_user",
    "mean_elp_ass_all_mean", 
    "mean_elp_ass_o_mean", 
    "mean_elp_ass_x_mean", 
    "answerRate_per_ass", "answerCount_per_ass", "answerVar_per_ass", "answerStd_per_ass",
    "elapsed",
    'elapsed_shift',
    "category",
    "acc_answerRate_per_cat",
    "acc_count_per_cat",
    "acc_elapsed_per_cat",
    "correct_answer_per_cat",
    "test_number",
    "mean_elp_pnum_all_mean", 
    "mean_elp_pnum_o_mean", 
    "mean_elp_pnum_x_mean", 
    "acc_tag_count_per_user",
    "problem_count",
    "problem_num",
    "answerRate_per_pnum", "answerCount_per_pnum", "answerVar_per_pnum", "answerStd_per_pnum",
    "problem_position",
    'timeDelta_userAverage',
    'timestep_1', 'timestep_2', 'timestep_3', 'timestep_4', 'timestep_5', #제거 시 valid, public score 모두 감소
    "median_elapsed_wrong_users", "median_elapsed_correct_users"
]

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
#     'correct_mean_per_user', 'correct_var_per_user', 'correct_std_per_user',
# ]

# ######################## SELECT FEATURE

# FEATURE = [
#     "userID",
#     "assessmentItemID",
#     "KnowledgeTag",
#     "elapsed",
#     "category_high",
#     "problem_num",
#     "cum_answerRate_per_user",
#     "acc_elapsed_per_user",
#     "problem_correct_per_user",
#     "problem_solved_per_user",
#     "correct_answer_per_cat",
#     "acc_count_per_cat",
#     "acc_answerRate_per_cat",
#     "timeDelta_userAverage",
#     "timestep_1",
#     "timestep_2",
#     "timestep_3",
#     "timestep_4",
#     "timestep_5",
#     "hour",
#     "weekofyear",
#     "problem_correct_per_woy",
#     "problem_solved_per_woy",
#     "cum_answerRate_per_woy",
# ]
# FEATURE_USER = [
#     "answerRate_per_user",
#     "answer_cnt_per_user",
#     "elapsed_time_median_per_user",
#     "assessment_solved_per_user",
# ]
# FEATURE_ITEM = [
#     "answerRate_per_item",
#     "answer_cnt_per_item",
#     "elapsed_time_median_per_item",
#     "wrong_users_median_elapsed",
#     "correct_users_median_elapsed",
# ]
# FEATURE_TEST = ["elapsed_median_per_test", "answerRate_per_test", "answerCount_per_test", "answerVar_per_test", "answerStd_per_test"]
# FEATURE_TAG = ["tag_exposed",  "answerRate_per_tag", "answerCount_per_tag", "answerVar_per_tag", "answerStd_per_tag"]
# FEATURE_CAT = ["elapsed_median_per_cat", "answerRate_per_cat"]
# FEATURE_PROBLEM_NUM = [#"answerRate_per_ass", "answerCount_per_ass", "answerVar_per_ass", "answerStd_per_ass",
#     "elapsed_median_per_problem_num","answerRate_per_problem_num",#"answerCount_per_problem_num","answerVar_per_problem_num","answerStd_per_problem_num"
#     ]
# FEATURE_ELO = ["elo_assessment", "elo_test", "elo_tag"]

# FEATURE += FEATURE_USER
# FEATURE += FEATURE_ITEM
# FEATURE += FEATURE_TEST
# FEATURE += FEATURE_TAG
# FEATURE += FEATURE_CAT
# FEATURE += FEATURE_PROBLEM_NUM
# FEATURE += FEATURE_ELO
# FEATS = FEATURE

def main(args):
    ####################### Setting for Log
    setting = Setting()
    data = lightgbm_dataloader(args)
    data = lightgbm_preprocess_data(data)
    
    if args.autogluon == True:
        args.model = 'Ensemble'
        
    train_data, test_data = data[data['answerCode'] != -1], data[data['answerCode']==-1]
    test_data = test_data.drop(["answerCode"], axis=1)

    if args.autogluon == True:
        predictor = TabularPredictor.load("./AutogluonModels/ag-20240107_135243")
        predicts_df = predictor.predict_proba(test_data)
        predicts_df['predicts'] = predicts_df[[0, 1]].max(axis=1)
        predicts = predicts_df['predicts'].values
        
        try:
            output = predictor.evaluate(train_data, silent=True)
            print('here')
            print(output)
            valid_auc = output["roc_auc"]
        except:
            valid_auc = 0.000
            
    else:
        # User별 split
        # train_data['length_per_test'] = train_data['solve_count_per_test']
        train, valid = train_test_split(train_data, test_size = args.test_size, random_state = args.seed, shuffle = args.data_shuffle)

        y_train = train["answerCode"]
        x_train = train.drop(["answerCode"], axis=1)

        y_valid = valid["answerCode"]
        x_valid = valid.drop(["answerCode"], axis=1)
        
        #with open(f'{args.saved_model_path}/20240108_141553_LIGHTGBM_0.8642_model.pkl', 'rb') as f: #0.8181 best
        with open(f'{args.saved_model_path}/20240109_051923_LIGHTGBM_0.8642_model.pkl', 'rb') as f: #0.8137 github
            predictor = pickle.load(f)
            
        # Get feature importances
        feature_importance = predictor.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': x_train[FEATS].columns, 'Importance': feature_importance / max(feature_importance)})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print(feature_importance_df)
        
        if args.feature_selection:
            # Select features based on importance scores
            sfm = SelectFromModel(predictor, threshold='1.1*median')  # Adjust the threshold as needed
            #sfm = SelectFromModel(predictor, threshold=400)
            sfm.fit(x_train[FEATS], y_train)
            
            # Transform the data to include only important features
            X_train_selected = sfm.transform(x_train[FEATS])
            X_valid_selected = sfm.transform(x_valid[FEATS])
            X_test_selected = sfm.transform(test_data[FEATS])
            
            model = models_load(args)
            model.fit(X_train_selected, y_train)
            
            preds = model.predict_proba(X_valid_selected)[:, 1]
            valid_acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
            valid_auc = roc_auc_score(y_valid, preds)
            
            print(f"VALID AUC : {valid_auc} VALID ACC : {valid_acc}\n")
            
            predicts = model.predict_proba(X_test_selected)[:, 1]
            
        else:
            preds = predictor.predict_proba(x_valid[FEATS])[:, 1]
            valid_acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
            valid_auc = roc_auc_score(y_valid, preds)
            
            print(f"VALID AUC : {valid_auc} VALID ACC : {valid_acc}\n")
            
            predicts = preds
            
            
        with open('record.txt', 'a') as f:
            f.write(f"Tiemstamp:{setting.save_time}, valid auc:{valid_auc}, valid_acc:{valid_acc}\n")
        f.close()
    
    
    output = input('출력 파일 생성하시겠습니까? (y/n): ')
    if output == 'y':
        ######################## SAVE PREDICT
        print(f'--------------- SAVE PREDICT ---------------')
        filename = setting.get_submit_filename(args, valid_auc)
        submission = pd.read_csv(args.data_dir + 'sample_submission.csv')
        
        submission['prediction'] = predicts

        submission.to_csv(filename, index=False)
        print('make csv file !!! ', filename)
    else:
        pass
    
    
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    main(args=args)