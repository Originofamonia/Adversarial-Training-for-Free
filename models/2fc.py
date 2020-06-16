import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def mse_loss():
    h_loss = nn.MSELoss()
    xent_loss = nn.CrossEntropyLoss()
    x = torch.tensor([0.5, 1], dtype=torch.float, requires_grad=True)  # x
    x_adv = torch.tensor([0.9, 0.6], dtype=torch.float, requires_grad=True)  # x_adv
    w1 = torch.tensor([[-1, 3], [-2, 4]], dtype=torch.float, requires_grad=True)
    b1 = torch.tensor([0, 0], dtype=torch.float, requires_grad=True)
    w2 = torch.tensor([-5, 6], dtype=torch.float, requires_grad=True)
    b2 = torch.tensor(0, dtype=torch.float, requires_grad=True)

    h = torch.matmul(w1, x) + b1
    y = torch.matmul(w2, h) + b2
    print('h: {}'.format(h))
    print('y: {}'.format(y))
    h_adv = torch.matmul(w1, x_adv) + b1
    y_adv = torch.matmul(w2, h_adv) + b2
    print('h_adv: {}'.format(h_adv))
    print('y_adv: {}'.format(y_adv))
    l = h_loss(h_adv, h)
    # l = xent_loss(y_adv, torch.tensor([1.0]))
    print('loss: {}'.format(l))
    l.backward()
    print('w1.grad: {}; w2.grad: {}; b1.grad: {}'.format(w1.grad, w2.grad, b1.grad))


def main():
    N = 100  # number of points per class
    D = 2  # dimensionality
    K = 2  # number of classes
    x = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    # lets visualize the data:
    # plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.show()

    # Train a Linear Classifier
    # initialize parameters randomly
    w = 0.01 * np.random.randn(D, K)
    b = np.zeros((1, K))

    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3  # regularization strength

    # gradient descent loop
    num_examples = x.shape[0]
    for i in range(num_examples):

        # evaluate class scores, [N x K]
        inputs = x
        scores = np.dot(inputs, w) + b

        # compute the class probabilities
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

        # compute the loss: average cross-entropy loss and regularization
        correct_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(w * w)
        loss = data_loss + reg_loss
        if i % 10 == 0:
            print("iteration %d: loss %f" % (i, loss))

        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples), y] -= 1
        dscores /= num_examples

        # backpropagate the gradient to the parameters (W,b)
        dW = np.dot(x.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)

        dW += reg * w  # regularization gradient

        # perform a parameter update
        w += -step_size * dW
        b += -step_size * db

    # evaluate training set accuracy
    scores = np.dot(x, w) + b
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == y)))

    # plot the resulting classifier
    h = 0.02
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()


if __name__ == '__main__':
    # main()
    mse_loss()
