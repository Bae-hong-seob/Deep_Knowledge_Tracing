from lightgbm import LGBMClassifier

def LIGTHGBM(args):
    return LGBMClassifier(n_estimators=args.n_estimators, 
                          #verbosity=100
                          )