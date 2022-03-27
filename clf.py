import joblib
from keras.datasets import mnist
from sklearn.linear_model import Perceptron


def get_dataset(filename=None):
    if filename:
        return joblib.load(filename)
    from sklearn.datasets import fetch_openml
    return fetch_openml('mnist_784',  as_frame=False)


def save_dataset(dataset, filename):
    joblib.dump(dataset, filename)


print('Loading mnist_784 dataset... ')
mnist = get_dataset()
# mnist = get_dataset('mnist.sav')
print('Training Perceptron... ')
clf = Perceptron(max_iter=1000, n_jobs=-1)
clf.fit(mnist.data, mnist.target)
print('Saving clf... ')
joblib.dump(clf, 'clf.sav')
# save_dataset(mnist, 'mnist.sav')
