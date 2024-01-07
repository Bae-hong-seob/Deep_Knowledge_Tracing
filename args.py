import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument
    
    ############### BASIC OPTION
    arg('--data_dir', type=str, default='data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    
    arg('--model', type=str, default='LIGHTGBM', choices=['XGB', 'LIGHTGBM','CATBOOST'], help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')

    arg("--use_cuda_if_available", default=True, type=bool, help="Use GPU")

    arg("--lr", default=0.001, type=float, help="")
    arg("--model_dir", default="./models/", type=str, help="")
    arg("--model_name", default="best_model.pt", type=str, help="")
    arg('--autogluon', type=bool, default=False, help='하이퍼 파라미터 자동 최적화 설정입니다.')
    
    ############### LightGBM OPTION
    arg("--n_estimators", default=1000, type=int, help="n_estimators")
    
    
    args = parser.parse_args()

    return args
