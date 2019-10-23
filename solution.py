import numpy as np
from numpy import linalg


class SVM:
    def __init__(self,eta, C, niter, batch_size, verbose):
        self.eta = eta; self.C = C; self.niter = niter; self.batch_size = batch_size; self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (in this homework, m will be 10)
        returns : numpy array of shape (n,m)
        """
        result = - np.ones(shape=(len(y), m))
        for i in range(len(y)):
            result[i][y[i]] = 1
        return result

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401)
        y : numpy array of shape (minibatch size, 10)
        returns : float
        """
        regularization = 1/2 * np.sum(np.power(linalg.norm(self.w, axis=0), 2))
        losses = 1 - np.dot(x, self.w) * y
        losses[losses <= 0] = 0
        sums = np.sum(losses ** 2, axis=0)
        return np.sum(sums) * self.C / x.shape[0] + regularization

    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401)
        y : numpy array of shape (minibatch size, 10)
        returns : numpy array of shape (401, 10)
        """
        x_copy = x.copy()
        losses = (1-np.dot(x_copy, self.w)*y)
        max_losses = np.maximum(0, losses)
        const_x = -2*self.C/y.shape[0]*np.transpose(x_copy)
        return np.dot(const_x, max_losses*y) + self.w
        


    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (number of examples to infer, 401)
        returns : numpy array of shape (number of examples to infer, 10)
        """
        res = np.dot(x, self.w)
        y = np.argmax(res, axis=1)
        return self.make_one_versus_all_labels(y, res.shape[1])

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (number of examples, 10)
        y : numpy array of shape (number of examples, 10)
        returns : float
        """
        mult = y * y_inferred
        sums = np.sum(mult, axis=1)
        return len(sums[sums == y.shape[1]]) / y.shape[0]

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, 401)
        y_train : numpy array of shape (number of training examples, 10)
        x_test : numpy array of shape (number of training examples, 401)
        y_test : numpy array of shape (number of training examples, 10)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x,y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train,y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test,y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print("Iteration %d:" % iteration)
                print("Train accuracy: %f" % train_accuracy)
                print("Train loss: %f" % train_loss)
                print("Test accuracy: %f" % test_accuracy)
                print("Test loss: %f" % test_loss)
                print("")

        return train_loss, train_accuracy, test_loss, test_accuracy

if __name__ == "__main__":
    # Load the data files
    print("Loading data...")

    x_train = np.load("train_features.npy")
    x_test = np.load("test_features.npy")
    y_train = np.load("train_labels.npy")
    y_test = np.load("test_labels.npy")

    print("Fitting the model...")
    svm = SVM(eta=0.001, C=30, niter=200, batch_size=5000, verbose=False)


    # test
    # svm.w = np.zeros([401, 10])
    # y = svm.make_one_versus_all_labels(y_train, 10)
    # loss = svm.compute_loss(x_train, y)
    # infer = svm.infer(x_test)
    # accuracy = svm.compute_accuracy(np.array([[ 1 ,-1 ,-1],[-1 , 1 ,-1]]), np.array([[ 1, -1, -1], [-1, -1,  1]]))

    train_loss, train_accuracy, test_loss, test_accuracy = svm.fit(x_train, y_train, x_test, y_test)

    # to infer after training, do the following:
    y_inferred = svm.infer(x_test)

    # to compute the gradient or loss before training, do the following:
    y_train_ova = svm.make_one_versus_all_labels(y_train, 10) # one-versus-all labels
    print(y_train_ova)
    svm.w = np.zeros([401, 10])
    grad = svm.compute_gradient(x_train, y_train_ova)
    loss = svm.compute_loss(x_train, y_train_ova)

