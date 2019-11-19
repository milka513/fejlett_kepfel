import cv2
import pickle
from sklearn.linear_model import SGDClassifier as SGD
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import ProcessImage as image
from sklearn.neighbors import KNeighborsClassifier
class train(object):

    def __init__(self):
        pass

    def train(self, X, y):
        #mean = X.mean(axis=0)
        #std = X.std(axis=0)
        #X = (X - mean) / std
        #print(X)
        #self.clf=SGD(loss='log', n_jobs=-1, n_iter_no_change=10, max_iter=10000000,penalty='elasticnet', tol=1e-10)
        #self.clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
        #self.clf.fit(X, y)

        #steps = [('pca', PCA()), ('clf', SVC(kernel='rbf'))]
        steps = [('pca', PCA()), ('clf', SVC(kernel='poly'))]
        #steps = [('pca', PCA()), ('clf', KNeighborsClassifier(n_neighbors=15))]
        pipe = Pipeline(steps)
        pca__n_components=[7]
        n_splits = 6
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
        clf__gamma = np.logspace(-4, -2, 3)  # [.0001, .001, .01]
        clf__C = np.logspace(0, 2, 3)  # [1, 10, 100]
        grid_params = dict(pca__n_components=pca__n_components,
                           clf__gamma=clf__gamma,
                            clf__C=clf__C)
        #grid_params=dict(n_neighbors=15)
        self.grid = GridSearchCV(pipe, grid_params, cv=cv, refit=True, n_jobs=-1, scoring='f1_micro')
        self.grid.fit(X, y)
        with open('out/clf_poly.pkl', 'wb') as f:
            f.flush()
            pickle.dump(self.grid, f)

    def load(self):
        filename='out/clf_poly.pkl'
        with open(filename, 'rb') as f:
            self.grid=pickle.load(f)


    def predict(self, X):

        list=[X]
        #mean = list.mean(axis=0)
        #std = list.std(axis=0)
        #list = (list - mean) / std
        return self.grid.predict(list)

    def test(self):
        im = image.ProcessImages('test')
        X, Y = im.make_all()
        return self.grid.score(X, Y)