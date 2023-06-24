import numpy as np

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    
    ## RETURN GAUSSIAN PROBABILITY
    def gaussian(mu, sigma2, d, x):
        p = (1/((2*np.pi*sigma2)**(d/2)))*np.exp(-(1/(2*sigma2))*np.linalg.norm(mu-x)**2)
        return p

    ## GAUSSIAN MIXTURE PROPERTIES
    # GMM = mixture[0]

    n = len(X) #dataset size
    k = mixture.mu.shape[0] #number of clusters
    d = mixture.mu.shape[1] #data dimension R^d
    post = [] #soft counts
    l_theta = 0 #log_likelihood

    for i in range(0, n): #over all the datapoints

        p_x_theta_ks = [] #list of individual likelihoods
        p_x_theta = 0 #sum of likelihoods

        for j in range(0, k): #over all the clusters

            # Calculating the likelihood for each data point regarding to each cluster
            p_x_theta_k = mixture.p[j]*gaussian(mixture.mu[j], mixture.var[j], d, X[i])

            # Updating 'p_x_theta_k' list
            p_x_theta_ks.append(p_x_theta_k) 

            # Calculating total likelihood for each data point
            p_x_theta += p_x_theta_k

        # Calculating the likelihood for each cluster
        likelihoods = p_x_theta_ks/p_x_theta
        
        # log-likelihood update
        l_theta += np.log(p_x_theta)

        # Assigning X[i] to the cluster where it has the greatest likelihood
        post.append(likelihoods.tolist())
               
    return np.array(post), l_theta 


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        mu[j, :] = post[:, j] @ X / n_hat[j]
        sse = ((mu[j] - X)**2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * n_hat[j])
        
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    
    prev_l_theta = None
    l_theta = None
    while (prev_l_theta is None or abs(prev_l_theta - l_theta) > abs(l_theta)*10**(-6)):
        prev_l_theta = l_theta
        post, l_theta = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, l_theta


def bic(X: np.ndarray, mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a
    mixture of gaussians

    Args:
        X: (n, d) array holding the data
        mixture: a mixture of spherical gaussian
        log_likelihood: the log-likelihood of the data

    Returns:
        float: the BIC for this mixture
    """
    
    n, d = X.shape
    K, _ = mixture.mu.shape
    
    p = K*(d + 2) - 1
    
    return log_likelihood - 0.5*p*np.log(n)
    
    
