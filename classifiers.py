import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LassoLarsCV
from collections import namedtuple

# is_regression flags whether the 'classifier' is actually a regression, e.g. lasso
Classifier = namedtuple("Classifier", "classifier parameters is_regression") 

classifiers = {
    "lasso_lars_no_intercept":Classifier(classifier=LassoLarsCV,
                                         is_regression=True,
                                         parameters={"normalize":[False],
                                                     "fit_intercept":[False],
                                                     "max_iter":[1000000]}),    

    "svc_linear_no_intercept":Classifier(classifier=LinearSVC,
                                         is_regression=False,
                                         parameters={"fit_intercept":[False],
                                                     "penalty":["l2"],
                                                     "C":[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
                                                     "max_iter":[1000000]}),

    "svc_linear_l1_no_intercept":Classifier(classifier=LinearSVC,
                                            is_regression=False,
                                            parameters={"fit_intercept":[False],
                                                        "penalty":["l1"],
                                                        "dual":[False],
                                                        "C":[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
                                                        "max_iter":[1000000]}),    
    "svc_poly":Classifier(classifier=SVC,
                          is_regression=False,
                          parameters={"kernel":["poly"],
                                      "C":[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
                                      "degree":[3],
                                      "max_iter":[1000000]}),

    "svc_rbf":Classifier(classifier=SVC,
                         is_regression=False,
                         parameters={"kernel":["rbf"],
                                     "C":[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
                                     "max_iter":[1000000]}),    
    }


classification_score_function = lambda search: search.score
regression_score_function     = lambda search: lambda X, y: np.mean(np.sign(search.predict(X) + np.random.randn(*y.shape)*1e-8).astype(int) == np.sign(y).astype(int))

score_function_selector       = lambda is_regression: regression_score_function if is_regression else classification_score_function
score_function                = {name:score_function_selector(clf.is_regression) for name, clf in classifiers.items()}

if __name__ == "__main__":
    with open("classifiers.list", "w") as f:
        for clf in classifiers:
            f.write(clf + "\n")
    print("Wrote classifiers.list")
    
    
                  
