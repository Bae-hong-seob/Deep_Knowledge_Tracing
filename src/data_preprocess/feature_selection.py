from sklearn.feature_selection import SelectFromModel

def feature_selection(args, model, feature_selected_model, x_train, y_train, x_valid, y_valid, x_test, setting):
    
    sfm = SelectFromModel(model, threshold='1.5*median')  # Adjust the threshold as needed
    sfm.fit(x_train, y_train)
    
    # Transform the data to include only important features
    X_train_selected = sfm.transform(x_train)
    X_valid_selected = sfm.transform(x_valid)
    X_test_selected = sfm.transform(x_test)
    
    feature_selected_model.fit(X_train_selected, y_train)
    
    return feature_selected_model, X_valid_selected, X_test_selected