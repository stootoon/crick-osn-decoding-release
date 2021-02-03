import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LassoLarsCV
from collections import namedtuple

Classifier = namedtuple("Classifier", "classifier parameters") 

classifiers = {
    "lasso_lars_no_intercept":Classifier(classifier=LassoLarsCV,
                                         parameters={"normalize":[False],
                                                     "fit_intercept":[False],
                                                     "max_iter":[1000000]}),    

    "svc_linear_no_intercept":Classifier(classifier=LinearSVC,
                                         parameters={"fit_intercept":[False],
                                                     "penalty":["l2"],
                                                     "C":[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
                                                     "max_iter":[1000000]}),

    "svc_linear_l1_no_intercept":Classifier(classifier=LinearSVC,
                                            parameters={"fit_intercept":[False],
                                                        "penalty":["l1"],
                                                        "dual":[False],
                                                        "C":[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
                                                        "max_iter":[1000000]}),    
    "svc_poly":Classifier(classifier=SVC,
                          parameters={"kernel":["poly"],
                                      "C":[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
                                      "degree":[3],
                                      "max_iter":[1000000]}),

    "svc_rbf":Classifier(classifier=SVC,
                         parameters={"kernel":["rbf"],
                                     "C":[0.0001,0.001,0.01,0.1,1,10,100,1000,10000],
                                     "max_iter":[1000000]}),    
    }


if __name__ == "__main__":
    with open("classifiers.list", "w") as f:
        for clf in classifiers:
            f.write(clf + "\n")
    print("Wrote classifiers.list")
    
    
                  
