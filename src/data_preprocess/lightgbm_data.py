import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split

def lightgbm_preprocess_data(df):

    ########## 기본 feature ##########

    # 유저별 시퀀스를 고려하기 위해 정렬
    df.sort_values(by=["userID", "Timestamp"], inplace=True)
    df.reset_index(inplace = True, drop=True)

    # 카테고리형 feature
    categories = ['userID',"assessmentItemID", "testId"]

    for category in categories:
        df[category] = df[category].astype("category")

    ########## 문제 관련 ##########

    # 문제 대분류 : 시험지 카테고리, 시험지 번호, 문제 번호
    df["category"] = df["testId"].apply(lambda x: int(x[2]))
    df["test_number"] = df["testId"].apply(lambda x: int(x[-3:]))
    df["problem_number"] = df["assessmentItemID"].apply(lambda x: int(x[-3:]))

    # 인접한 testId grouping
    index = df[df['testId'] != df['testId'].shift(-1)].index
    grouping = [0] * (index[0] + 1)
    for i in range(1, len(index)):
        grouping += [i] * (index[i] - index[i-1])
    df['grouping'] = grouping

    # 시험지별 푼 문제 개수, 푼 사용자 수
    length_test = df.groupby('testId')['assessmentItemID'].nunique().reset_index()
    length_test.columns = ['testId', 'solve_count_per_test']

    frequency_of_test = df['testId'].value_counts().sort_index().values

    if not np.array_equal(np.array(frequency_of_test/length_test['solve_count_per_test'], dtype=int), np.array(frequency_of_test/length_test['solve_count_per_test'])):
        raise print('사용자 수가 올바르지 않습니다')

    length_test['number_of_users_per_test'] = np.array(frequency_of_test/length_test['solve_count_per_test'], dtype=int)
    df = pd.merge(df, length_test, on=['testId'], how='left')

    # 시험지 별 문제 수와 태그 수
    f = lambda x: len(set(x))
    test = df.groupby(["testId"]).agg({"problem_number": "max", "KnowledgeTag": f})
    test.reset_index(inplace=True)
    test.columns = ["testId", "problem_count", "tag_count"]
    df = pd.merge(df, test, on="testId", how="left")
    df["problem_position"] = df["problem_number"] / df["problem_count"]

    # 시험지 별 안 푼 문제 개수, 문제를 푼 비율 -> valid score감소, public score 감소
    df['not_solving_count_per_test'] = df['problem_count'] - df['solve_count_per_test']
    df['problem_solving_rate_per_test'] = df['solve_count_per_test'] / df['problem_count']

    # 시험지별 정답 평균, 개수, 분산, 표준편차 -> 평균만 생성시 public score 감소
    per_test = df.groupby(["testId"])["answerCode"].agg(["mean", "sum", 'var', 'std'])
    per_test.columns = ["answerRate_per_test", "answerCount_per_test", "answerVar_per_test", "answerStd_per_test"]
    df = pd.merge(df, per_test, on=["testId"], how="left")

    # 태그별 정답 평균, 개수, 분산, 표준편차
    per_tag = df.groupby(["KnowledgeTag"])["answerCode"].agg(["mean", "sum", 'var', 'std'])
    per_tag.columns = ["answerRate_per_tag", "answerCount_per_tag", "answerVar_per_tag", "answerStd_per_tag"]
    df = pd.merge(df, per_tag, on=["KnowledgeTag"], how="left")

    # 문항별 정답 평균, 개수, 분산, 표준편차 -> 평균만 생성시 public score 감소
    per_ass = df.groupby(["assessmentItemID"])["answerCode"].agg(["mean", "sum", 'var', 'std'])
    per_ass.columns = ["answerRate_per_ass", "answerCount_per_ass", "answerVar_per_ass", "answerStd_per_ass"]
    df = pd.merge(df, per_ass, on=["assessmentItemID"], how="left")

    # 문제 번호별 정답 평균, 개수, 분산, 표준편차
    per_pnum = df.groupby(["problem_number"])["answerCode"].agg(["mean", "sum", 'var', 'std'])
    per_pnum.columns = ["answerRate_per_pnum", "answerCount_per_pnum", "answerVar_per_pnum", "answerStd_per_pnum"]
    df = pd.merge(df, per_pnum, on=["problem_number"], how="left")

    ########## Time 관련 ##########

    # 문제 풀이 시간 ver1 : elapsed shift(1)
    diff = df[["userID", "grouping", "Timestamp"]].groupby(["userID", "grouping"]).diff().fillna(pd.Timedelta(seconds=0))
    df["elapsed_shift"] = diff["Timestamp"].apply(lambda x: x.total_seconds())

    # 문제 풀이 시간 ver2 : 문제 푸는 시간
    df['elapsed'] = df['elapsed_shift'].shift(-1).fillna(value = 0)
    temp = df[df['testId'] == df['testId'].shift(-1)].groupby('grouping')['elapsed'].mean().to_dict()
    df.loc[df['testId'] != df['testId'].shift(-1), 'elapsed'] = df.loc[df['testId'] != df['testId'].shift(-1), 'grouping'].map(temp).fillna(value = 0).astype(int)

    # 맞춘 문제와 틀린 문제
    correct_df = df[df["answerCode"] == 1]
    wrong_df = df[df["answerCode"] == 0]

    # 태그별 모든 문제, 맞춘 문제, 틀린 문제별 풀이 시간의 평균
    mean_elapsed_tag = df.groupby(["KnowledgeTag"])["elapsed"].agg(["mean", "sum", 'var', 'std']).reset_index()
    mean_elapsed_tag.columns = ["KnowledgeTag", "mean_elp_tag_all_mean", "mean_elp_tag_all_sum", "mean_elp_tag_all_var", "mean_elp_tag_all_std"]
    df = pd.merge(df, mean_elapsed_tag, on=["KnowledgeTag"], how="left")

    mean_elapsed_tag_o = correct_df.groupby(["KnowledgeTag"])["elapsed"].agg(["mean", "sum", 'var', 'std']).reset_index()
    mean_elapsed_tag_o.columns = ["KnowledgeTag", "mean_elp_tag_o_mean", "mean_elp_tag_o_sum", "mean_elp_tag_o_var", "mean_elp_tag_o_std"]
    df = pd.merge(df, mean_elapsed_tag_o, on=["KnowledgeTag"], how="left")

    mean_elapsed_tag_x = wrong_df.groupby(["KnowledgeTag"])["elapsed"].agg(["mean", "sum", 'var', 'std']).reset_index()
    mean_elapsed_tag_x.columns = ["KnowledgeTag", "mean_elp_tag_x_mean", "mean_elp_tag_x_sum", "mean_elp_tag_x_var", "mean_elp_tag_x_std"]
    df = pd.merge(df, mean_elapsed_tag_x, on=["KnowledgeTag"], how="left")

    # 문항별 모든 문제, 맞춘 문제, 틀린 문제별 풀이 시간의 평균
    mean_elapsed_ass = df.groupby(["assessmentItemID"])["elapsed"].agg(["mean", "sum", 'var', 'std']).reset_index()
    mean_elapsed_ass.columns = ["assessmentItemID", "mean_elp_ass_all_mean", "mean_elp_ass_all_sum", "mean_elp_ass_all_var", "mean_elp_ass_all_std"]
    df = pd.merge(df, mean_elapsed_ass, on=["assessmentItemID"], how="left")

    mean_elapsed_ass_o = correct_df.groupby(["assessmentItemID"])["elapsed"].agg(["mean", "sum", 'var', 'std']).reset_index()
    mean_elapsed_ass_o.columns = ["assessmentItemID", "mean_elp_ass_o_mean", "mean_elp_ass_o_sum", "mean_elp_ass_o_var", "mean_elp_ass_o_std"]
    df = pd.merge(df, mean_elapsed_ass_o, on=["assessmentItemID"], how="left")

    mean_elapsed_ass_x = wrong_df.groupby(["assessmentItemID"])["elapsed"].agg(["mean", "sum", 'var', 'std']).reset_index()
    mean_elapsed_ass_x.columns = ["assessmentItemID", "mean_elp_ass_x_mean", "mean_elp_ass_x_sum", "mean_elp_ass_x_var", "mean_elp_ass_x_std"]
    df = pd.merge(df, mean_elapsed_ass_x, on=["assessmentItemID"], how="left")

    # 문제 번호별 모든 문제, 맞춘 문제, 틀린 문제별 풀이 시간의 평균
    mean_elapsed_pnum = df.groupby(["problem_number"])["elapsed"].agg(["mean", "sum", 'var', 'std']).reset_index()
    mean_elapsed_pnum.columns = ["problem_number", "mean_elp_pnum_all_mean", "mean_elp_pnum_all_sum", "mean_elp_pnum_all_var", "mean_elp_pnum_all_std"]
    df = pd.merge(df, mean_elapsed_pnum, on=["problem_number"], how="left")

    mean_elapsed_pnum_o = correct_df.groupby(["problem_number"])["elapsed"].agg(["mean", "sum", 'var', 'std']).reset_index()
    mean_elapsed_pnum_o.columns = ["problem_number", "mean_elp_pnum_o_mean", "mean_elp_pnum_o_sum", "mean_elp_pnum_o_var", "mean_elp_pnum_o_std"]
    df = pd.merge(df, mean_elapsed_pnum_o, on=["problem_number"], how="left")

    mean_elapsed_pnum_x = wrong_df.groupby(["problem_number"])["elapsed"].agg(["mean", "sum", 'var', 'std']).reset_index()
    mean_elapsed_pnum_x.columns = ["problem_number", "mean_elp_pnum_x_mean", "mean_elp_pnum_x_sum", "mean_elp_pnum_x_var", "mean_elp_pnum_x_std"]
    df = pd.merge(df, mean_elapsed_pnum_x, on=["problem_number"], how="left")

    # 유저 평균과의 시간 차이
    for i in range(1,6):
        df[f"timestep_{i}"] = df.groupby("userID")["answerCode"].shift(i).fillna(1).astype(int)

    # 유저별 문제풀이 중앙값, 평균과의 차이
    df_time = df.groupby(["userID"])['elapsed'].agg(["median", 'mean'])
    df_time.columns = ['user_median_elapsed', 'user_mean_elapsed']
    df = pd.merge(df, df_time, on=['userID'], how='left')
    df["timeDelta_userAverage"] = df["elapsed"] - df["user_median_elapsed"]
        
    # 문제 정답 / 오답자들의 문제 풀이 시간 중앙값
    col_name = ["median_elapsed_wrong_users", "median_elapsed_correct_users"]
    for i in range(2):
        df_median_elapsed = (df[["assessmentItemID", "answerCode", "elapsed"]].groupby(["assessmentItemID", "answerCode"]).agg("median").reset_index())
        df_median_elapsed = df_median_elapsed[df_median_elapsed["answerCode"] == i].drop("answerCode", axis=1)
        df_median_elapsed.rename(columns={"elapsed": col_name[i]}, inplace=True)
        df = df.merge(df_median_elapsed, on=["assessmentItemID"], how="left")

    # 문제 정답 / 오답자들의 문제 풀이 시간 평균
    col_name = ["mean_elapsed_wrong_users", "mean_elapsed_correct_users"]
    for i in range(2):
        df_median_elapsed = (df[["assessmentItemID", "answerCode", "elapsed"]].groupby(["assessmentItemID", "answerCode"]).agg("mean").reset_index())
        df_median_elapsed = df_median_elapsed[df_median_elapsed["answerCode"] == i].drop("answerCode", axis=1)
        df_median_elapsed.rename(columns={"elapsed": col_name[i]}, inplace=True)
        df = df.merge(df_median_elapsed, on=["assessmentItemID"], how="left")


    ########## User 관련 ##########

    # User별 정답률, 문제푼 횟수, 맞춘 문제수
    df["problem_correct_per_user"] = (df.groupby("userID")["answerCode"].transform(lambda x: x.cumsum().shift(1)).fillna(0))
    df["problem_solved_per_user"] = df.groupby("userID")["answerCode"].cumcount()
    df["cum_answerRate_per_user"] = (df["problem_correct_per_user"] / df["problem_solved_per_user"]).fillna(0)

    # 유저별 Tag 문제 누적 값
    df["acc_tag_count_per_user"] = df.groupby(["userID", "KnowledgeTag"]).cumcount()

    # User별로 대분류별 맞춘 문제 개수, 대분류별 맞춘 문제 개수, 대분류별 정답률
    df["correct_answer_per_cat"] = (df.groupby(["userID", "category"])["answerCode"].transform(lambda x: x.cumsum().shift(1)).fillna(0))
    df["acc_count_per_cat"] = df.groupby(["userID", "category"]).cumcount()
    df["acc_answerRate_per_cat"] = (df["correct_answer_per_cat"] / df["acc_count_per_cat"]).fillna(0)
    df["acc_elapsed_per_cat"] = (df.groupby(["userID", "category"])["elapsed"].transform(lambda x: x.cumsum()).fillna(0))

    return df

def lightgbm_dataloader(args):
    """
    Parameters
    ----------
    Args:
        data_dir : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    train = pd.read_csv(args.data_dir + 'train_data.csv', parse_dates=["Timestamp"])
    test = pd.read_csv(args.data_dir + 'test_data.csv', parse_dates=["Timestamp"])
    
    train['dataset'] = 1
    test['dataset'] = 2
    
    data = pd.concat([train, test])
    
    return data

def lightgbm_datasplit(args, df):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """
    # User별 split
    train, valid = train_test_split(df[df['answerCode'] != -1], test_size = args.test_size, random_state = args.seed, shuffle = args.data_shuffle)

    y_train = train["answerCode"]
    x_train = train.drop(["answerCode"], axis=1)

    y_valid = valid["answerCode"]
    x_valid = valid.drop(["answerCode"], axis=1)

    return x_train, y_train, x_valid, y_valid