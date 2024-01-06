from xgboost import XGBRegressor
        
def eXtraGredientBoost(args):
    return XGBRegressor(n_estimators=args.n_estimators,
                        learning_rate=args.lr,
                        early_stopping_rounds=args.early_stopping_rounds,
                        max_depth=args.max_depth,
                        gamma=args.gamma,
                        colsample_bytree=args.colsample_bytree,
                        random_state=args.random_state,
                        eval_metric='rmse',
                        verbose=100
                        )