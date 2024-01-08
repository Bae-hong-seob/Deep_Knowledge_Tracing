from lightgbm import LGBMClassifier

def LIGTHGBM(args):
    return LGBMClassifier(n_estimators=args.n_estimators,
                          random_state=args.seed,
                          lambda_l1=args.lambda_l1,
                          lambda_l2=args.lambda_l2,
                          #verbosity=100
                          )