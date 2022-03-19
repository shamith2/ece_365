import numpy as np
import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# You may use this function as you like.
error = lambda y, yhat: np.mean(y!=yhat)

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Question1(object):
    # The sequence in this problem is different from the one you saw in the jupyter notebook. This makes it easier to grade. Apologies for any inconvenience.
    def BernoulliNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a BernoulliNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        # classifier
        classifier = BernoulliNB()
            
        # computing classifier training time
        start_time = time.time()
        classifier.fit(traindata, trainlabels)
        end_time = time.time()
        fittingTime = end_time - start_time
        
        # predictions on training data
        predtrainlabels = classifier.predict(traindata)
        
        # training error
        trainingError = error(trainlabels, predtrainlabels)
        
        # computing classifier validation time
        start_time = time.time()
        
        # predictions on validation data
        predvallabels = classifier.predict(valdata)
        
        end_time = time.time()
        valPredictingTime = end_time - start_time
        
        # validation error
        validationError = error(vallabels, predvallabels)    

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def MultinomialNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a MultinomialNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        # classifier
        classifier = MultinomialNB()
            
        # computing classifier training time
        start_time = time.time()
        classifier.fit(traindata, trainlabels)
        end_time = time.time()
        fittingTime = end_time - start_time
        
        # predictions on training data
        predtrainlabels = classifier.predict(traindata)
        
        # training error
        trainingError = error(trainlabels, predtrainlabels)
        
        # computing classifier validation time
        start_time = time.time()
        
        # predictions on validation data
        predvallabels = classifier.predict(valdata)
        
        end_time = time.time()
        valPredictingTime = end_time - start_time
        
        # validation error
        validationError = error(vallabels, predvallabels)  

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LinearSVC_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LinearSVC classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        # classifier
        classifier = LinearSVC()
            
        # computing classifier training time
        start_time = time.time()
        classifier.fit(traindata, trainlabels)
        end_time = time.time()
        fittingTime = end_time - start_time
        
        # predictions on training data
        predtrainlabels = classifier.predict(traindata)
        
        # training error
        trainingError = error(trainlabels, predtrainlabels)
        
        # computing classifier validation time
        start_time = time.time()
        
        # predictions on validation data
        predvallabels = classifier.predict(valdata)
        
        end_time = time.time()
        valPredictingTime = end_time - start_time
        
        # validation error
        validationError = error(vallabels, predvallabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LogisticRegression_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LogisticRegression classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        # classifier
        classifier = LogisticRegression()
            
        # computing classifier training time
        start_time = time.time()
        classifier.fit(traindata, trainlabels)
        end_time = time.time()
        fittingTime = end_time - start_time
        
        # predictions on training data
        predtrainlabels = classifier.predict(traindata)
        
        # training error
        trainingError = error(trainlabels, predtrainlabels)
        
        # computing classifier validation time
        start_time = time.time()
        
        # predictions on validation data
        predvallabels = classifier.predict(valdata)
        
        end_time = time.time()
        valPredictingTime = end_time - start_time
        
        # validation error
        validationError = error(vallabels, predvallabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def NN_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a Nearest Neighbor classifier using the given data.

        Make sure to modify the default parameter.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata              (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels            (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below
        # classifier
        classifier = KNeighborsClassifier(n_neighbors = 1)
            
        # computing classifier training time
        start_time = time.time()
        classifier.fit(traindata, trainlabels)
        end_time = time.time()
        fittingTime = end_time - start_time
        
        # predictions on training data
        predtrainlabels = classifier.predict(traindata)
        
        # training error
        trainingError = error(trainlabels, predtrainlabels)
        
        # computing classifier validation time
        start_time = time.time()
        
        # predictions on validation data
        predvallabels = classifier.predict(valdata)
        
        end_time = time.time()
        valPredictingTime = end_time - start_time
        
        # validation error
        validationError = error(vallabels, predvallabels)

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def confMatrix(self,truelabels,estimatedlabels):
        """ Write a function that calculates the confusion matrix (cf. Fig. 2.1 in the notes).

        You may wish to read Section 2.1.1 in the notes -- it may be helpful, but is not necessary to complete this problem.

        Parameters:
        1. truelabels           (Nv, ) numpy ndarray. The ground truth labels.
        2. estimatedlabels      (Nv, ) numpy ndarray. The estimated labels from the output of some classifier.

        Outputs:
        1. cm                   (2,2) numpy ndarray. The calculated confusion matrix.
        """
        cm = np.zeros((2,2))
        # Put your code below
        cm[0,0]=np.sum(np.logical_and(truelabels==1, estimatedlabels==1))   # True Positives
        cm[0,1]=np.sum(np.logical_and(truelabels==-1, estimatedlabels==1))  # False Positive
        cm[1,0]=np.sum(np.logical_and(truelabels==1, estimatedlabels==-1))  # False Negative
        cm[1,1]=np.sum(np.logical_and(truelabels==-1, estimatedlabels==-1)) # True Negatives

        return cm

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Run the classifier you selected in the previous part of the problem on the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. testError            Float. The reported test error. It should be less than 1.
        3. confusionMatrix      (2,2) numpy ndarray. The resulting confusion matrix. This will not be graded.
        """
        # Put your code below
        # classifier selected is LogisticRegression because of all classifiers above,
        # LogisticRegression has the least training and validation error
        classifier = LogisticRegression()
            
        # training classifier
        classifier.fit(traindata, trainlabels)
        
        # predicting on test data
        predtestlabels = classifier.predict(testdata)
        
        # test error
        testError = error(testlabels, predtestlabels)
        
        # confusion matrix
        # You can freely use the following line
        confusionMatrix = self.confMatrix(testlabels, predtestlabels)

        # Do not change this sequence!
        return (classifier, testError, confusionMatrix)

class Question2(object):
    def crossValidationkNN(self, traindata, trainlabels, k):
        """ Write a function which implements 5-fold cross-validation to estimate the error of a classifier with cross-validation with the 0,1-loss for k-Nearest Neighbors (kNN).

        For this problem, take your folds to be 0:N/5, N/5:2N/5, ..., 4N/5:N for cross-validation.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. k                    Integer. The cross-validated error estimates will be outputted for 1,...,k.

        Outputs:
        1. err                  (k+1,) numpy ndarray. err[i] is the cross-validated estimate of using i neighbors (the zero-th component of the vector will be meaningless).
        """
        # Put your code below
        # cross validated errors
        err = np.zeros(k + 1)
        N, d = traindata.shape
        num_groups = 5
        group_elem = N // num_groups
        
        for i in range(1, k + 1):
            prederror = np.zeros(num_groups)
            classifier = KNeighborsClassifier(n_neighbors=i)
            
            for group in range(num_groups):
                # validation indices
                val_idx = slice(group * group_elem, (group + 1) * group_elem)
                mask_idx = np.zeros(N)
                mask_idx[val_idx] = 1
    
                # train knn classifier
                classifier.fit(traindata[mask_idx != 1], trainlabels[mask_idx != 1])
                # predict training labels
                predtrainlabels = classifier.predict(traindata[val_idx])
                # prediction error for each group
                prederror[group] = error(predtrainlabels, trainlabels[val_idx])
                
            # average of prediction errors of all groups
            err[i] = np.mean(prederror)

        return err

    def minimizer_K(self, traindata, trainlabels, k):
        """ Write a function that calls the above function and returns 1) the output from the previous function, 2) the number of neighbors within  1,...,k  that minimizes the cross-validation error, and 3) the correponding minimum error.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. k                    Integer. The cross-validated error estimates will be outputted for 1,...,k.

        Outputs:
        1. err                  (k+1,) numpy ndarray. The output from crossValidationkNN().
        2. k_min                Integer (np.int64 or int). The number of neighbors within  1,...,k  that minimizes the cross-validation error.
        3. err_min              Float. The correponding minimum error.
        """
        err = self.crossValidationkNN(traindata, trainlabels, k)
        
        # Put your code below
        # the number of neighbors within  1,...,k  that minimizes the cross-validation error
        k_min = np.argmin(err[1: k + 1]) + 1
        
        # minimum error
        err_min = err[k_min]

        # Do not change this sequence!
        return (err, k_min, err_min)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train a kNN model on the whole training data using the number of neighbors you found in the previous part of the question, and apply it to the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best k value that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        # finding the number of neighbors within  1,...,k  that minimizes the cross-validation error
        (err, k_min, err_min) = self.minimizer_K(traindata, trainlabels, 30)
        # classifier with n_neighbors = k_min
        classifier = KNeighborsClassifier(n_neighbors=k_min)
        # training the classifier
        classifier.fit(traindata, trainlabels)
        # predicting on test data
        predtestlabels = classifier.predict(testdata)
        # test error
        testError = error(testlabels, predtestlabels)

        # Do not change this sequence!
        return (classifier, testError)

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class Question3(object):
    def LinearSVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15}.

        You should seaerch by hand.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        # list of cross-validated errors for each C
        crossvalerr = []
        
        for c in range(-5, 16):
            # classifier
            classifier = LinearSVC(C = 2**c)
            
            # cross-validated score
            crossvalscore = cross_val_score(classifier, traindata, trainlabels, cv=10)
            
            # crossvalscore gives accuarcy and not error
            crossvalerr.append(1 - np.mean(crossvalscore))
        
        # c with minimum error 
        c_min = np.argmin(crossvalerr)
        C_min = 1.0 / (2 ** abs(c_min - 5))
        
        # minimum error
        min_err = crossvalerr[c_min]

        # Do not change this sequence!
        return (C_min, min_err)

    def SVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15} and \gamma from 2^{-15},...,2^{3}.

        Use GridSearchCV to perform a grid search.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. gamma_min            Float. The hyper-parameter \gamma that minimizes the validation error.
        3. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        # parameters to search
        params = {'C': [2 ** c for c in range(-5, 16)], 'gamma': [2 ** c for c in range(-15, 4)]}
        
        # classifier
        classifier = GridSearchCV(SVC(kernel='rbf'), params, cv=10)
        classifier.fit(traindata, trainlabels)
        
        # best parameters
        bestparams = classifier.best_params_
        C_min = bestparams['C']
        gamma_min = bestparams['gamma']
        
        # cross validation accuracy
        crossvalacc = classifier.cv_results_['mean_test_score'][np.argwhere(np.array(classifier.cv_results_['params']) == bestparams)][0][0]
        
        # minimum error
        min_err = 1 - crossvalacc

        # Do not change this sequence!
        return (C_min, gamma_min, min_err)

    def LogisticRegression_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-14},...,2^{14}.

        You may either use GridSearchCV or search by hand.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        # parameters to search
        params = {'C': [2 ** c for c in range(-14, 15)]}
        
        # classifier
        classifier = GridSearchCV(LogisticRegression(), params, cv=10)
        
        # training classifier
        classifier.fit(traindata, trainlabels)
        
        # best parameters
        bestparams = classifier.best_params_
        C_min = bestparams['C']
        
        # cross validated accuracy
        crossvalacc = classifier.cv_results_['mean_test_score'][np.argwhere(np.array(classifier.cv_results_['params']) == bestparams)][0][0]
        
        # minimum error
        min_err = 1 - crossvalacc

        # Do not change this sequence!
        return (C_min, min_err)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train the best classifier selected above on the whole training set.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best classifier that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        # best parameters
        (C_min, gamma_min, min_err) = self.SVC_crossValidation(traindata, trainlabels)
        
        # classifier
        classifier = SVC(kernel='rbf', C=C_min, gamma=gamma_min)
        
        # training classifier
        classifier.fit(traindata, trainlabels)
        
        # prediction on test data
        predtestlabels = classifier.predict(testdata)
        
        # test error
        testError = error(testlabels, predtestlabels)

        # Do not change this sequence!
        return (classifier, testError)
