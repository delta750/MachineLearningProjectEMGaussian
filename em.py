# Author: Kapetanas Nick 
import numpy as np
import math
from PIL import Image

np.random.seed(0)

# K = Number of Gaussians in the mixture. In other words, number of "clusters"
Kp = [1, 2, 4, 8, 16, 32, 64]
for K in Kp:

    max_iterations = 100
    tol = 1e-6

    # initialize the weights
    p = np.ones((K, 1)) * 1 / K


    # load the image that we want to segment
    image_data = np.array(Image.open('im.jpg'))
    working_data = image_data.reshape((image_data.shape[0] * image_data.shape[1], image_data.shape[2]))

    # log_likelihoods
    log_likelihoods = []

    X = working_data

    # N is the number of data samples
    N, D = X.shape

    # initialize the means with random values
    M = np.array([np.mean(X, axis=0), ] * K) + np.random.randn(K, D)

    # responsibility matrix is initialized to all zeros
    # we have responsibility for each of n points for eack of k gaussians
    gamma = np.zeros((N, K))

    # the sigma is initialized with the variance of each row of the image
    sigma = np.ones((K, 1)) * np.var(working_data, axis=0)



    # M Step
    # We calculate the new mean and covariance for each gaussian by utilizing the new responsibilities

    def m_step(sigma, N_k):
        for k in range(1, K):
            # means
            M[k] = 1. / N_k[k] * np.sum(gamma[:, k] * X.T, axis=1).T

            # Simplicity calculation
            XMeinusM = np.matrix(X - M[k])

            # covariances
            sigma[k] = np.diag(1 / N_k[k] * np.dot(np.multiply(XMeinusM.T, gamma[:, k]), XMeinusM))
            # the probabilities
            p[k] = 1. / N * N_k[k]

        return sigma, M, p


    while len(log_likelihoods) < max_iterations:

        for k in range(1, K):
            # We compute the Gaussians
            nominator = np.exp(-np.power((X - M[k, :]), 2) / (2 * sigma[k, :]))
            denominator = np.sqrt(2 * np.math.pi * sigma[k, :])
            result = nominator / denominator
            productResult = np.product(result, axis=1)

            gamma[:, k] = productResult * p[k]

        # Likelihood computation
        log_likelihood = np.sum(np.log(np.sum(gamma, axis=1)))

        # We add the new calculated likelihood
        log_likelihoods.append(log_likelihood)

        # Final form of gamma function
        gamma = (gamma.T / np.sum(gamma, axis=1)).T

        # We sum all the rows of gamma as a part of m-step
        N_k = np.sum(gamma, axis=0)

        sigma, M, p = m_step(sigma, N_k)

        # check for onvergence
        if len(log_likelihoods) < 2:
            continue
        if np.abs(log_likelihood - log_likelihoods[-2]) < tol:
            break


    def index_maker():

        index = np.argmax(gamma, axis=1)
        newMeans = M[index]
        return newMeans


    def image_maker(K):
        newMeans = index_maker()
        imgTemp = np.array((newMeans).reshape((690, 550, 3)))
        img = Image.fromarray(imgTemp.astype('uint8'))
        img.save(str(K) + "_result.jpg")


    def error_calculator(gamma, M, X, N):
        mTemp = index_maker()
        error = (np.power(np.linalg.norm(X - mTemp), 2) * 1. / N)
        print(error, "error")
        return error

    image_maker(K)
    error_calculator(gamma, M, X, N)

