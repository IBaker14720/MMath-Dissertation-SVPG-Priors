import abc
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple

# --- Prior class ---

class BasePrior(abc.ABC):
    """
    Base class for q(theta)
    """

    @abc.abstractmethod
    def log_prob(self, theta: tf.Tensor):
        """
        Log-density log q(theta).

        Parameters:
            theta: tf.Tensor of shape [M, D] or [D].

        Returns:
            log_q(theta): tf.Tensor of shape [M] or scalar.
        """
        pass

    @abc.abstractmethod
    def grad_log_prob(self, theta: tf.Tensor):
        """
        Gradient of log q(theta) for SVPG

        Parameters:
            theta: tf.Tensor of shape [M, D].

        Returns:
            grad_log_q(theta): tf.Tensor of shape [M, D].
        """
        pass

    @abc.abstractmethod
    def sample(self, num_samples: int, dim: int):
        """
        Draw samples from the prior

        Parameters:
            num_samples: number of parameter vectors to sample (M).
            dim: dimension of theta (D).

        Returns:
            samples: tf.Tensor of shape [num_samples, dim].
        """
        pass


class UninformativePrior(BasePrior):
    # Flat/Uniform: log q(theta) = constant, grad_log_q = 0, no samples with special structure.
    def __init__(self):
        self.name = "Uninformative Prior"

    def log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        return tf.zeros(tf.shape(theta)[0])

    def grad_log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(theta)

    def sample(self, num_samples: int, dim: int) -> tf.Tensor:
        # Needs some range so because over all real numbers, impossible to sample
        return tf.random.normal(shape=(num_samples, dim), mean=0.0, stddev=100000)


class GaussianPrior(BasePrior):
    # N(mu, sigma^2): diagonal covariance
    def __init__(self, mean=0.0, std=1.0):
        self.name = f"Gaussian Prior, mean = {mean}, std = {std}"
        self.mean = tf.convert_to_tensor(mean, dtype=tf.float32)
        self.std  = tf.convert_to_tensor(std,  dtype=tf.float32)
        self.var  = self.std ** 2

    def log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def grad_log_prob(self, theta):
        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        return -(theta - self.mean) / (self.var + 1e-8)

    def sample(self, num_samples: int, dim: int) -> tf.Tensor:
        return tf.random.normal(shape=(num_samples, dim), mean=self.mean, stddev=self.std)

class LaplacePrior(BasePrior):
    # q(theta) = 1/2b * exp(-|x-mu|/b)
    def __init__(self, location = 0.0, scale = 1.0):
        self.name = f"Laplace Prior, location = {location}, std = {scale}"
        self.location = location
        self.scale = scale

    def log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        # log q(theta) = -log(2b)-|x-mu|/b
        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        log_pdf_per_theta = -tf.math.log(2.0 * self.scale) - tf.abs(theta - self.location) / self.scale
        
        # sum over parameters
        return tf.reduce_sum(log_pdf_per_theta, axis=-1)

    def grad_log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        # grad log p(theta) = -(1/b) * sign(theta - mu), ignoring non-differentiability at 0.

        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        return -(1.0/self.scale) * tf.sign(theta - self.location)

    def sample(self, num_samples: int, dim: int) -> tf.Tensor:
        samples = np.random.laplace(loc=self.location, scale=self.scale, size=(num_samples, dim)).astype(np.float32)
        return tf.convert_to_tensor(samples, dtype=tf.float32)

class CauchyPrior(BasePrior):
    # Cauchy(x0, gamma): q(theta) = 1 / (pi * gamma * (1 + ((x-x0)/gamma)^2))
    def __init__(self, location=0.0, scale=1.0):
        self.name = f"Cauchy Prior, location = {location}, scale = {scale}"
        self.location = tf.convert_to_tensor(location, dtype=tf.float32)
        self.scale = tf.convert_to_tensor(scale, dtype=tf.float32)

    def log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        log_pdf_per_theta = -tf.math.log(np.pi * self.scale) - tf.math.log(1.0 + ((theta - self.location) / self.scale)**2)
        return tf.reduce_sum(log_pdf_per_theta, axis=-1)

    def grad_log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        return (-2.0 * (theta - self.location)) / (self.scale**2 + (theta - self.location)**2)

    def sample(self, num_samples: int, dim: int) -> tf.Tensor:
        samples = np.random.standard_cauchy(size=(num_samples, dim)).astype(np.float32)
        samples = float(self.scale.numpy()) * samples + float(self.location.numpy())
        return tf.convert_to_tensor(samples, dtype=tf.float32)

class StudentTPrior(BasePrior):
    # Student-t prior with ν degrees of freedom.
    def __init__(self, df: float = 3.0, loc: float = 0.0, scale: float = 1.0):
        self.df = df
        self.loc = loc
        self.scale = scale

    def log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        # TODO: implement if needed.
        raise NotImplementedError

    def grad_log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        """
        ∇ log tν(θ; loc, scale) – heavy-tailed.
        """
        # TODO: implement gradient formula.
        raise NotImplementedError

    def sample(self, num_samples: int, dim: int) -> tf.Tensor:
        z = tf.random.normal((num_samples, dim), dtype=tf.float32)
        chi2 = tf.random.gamma((num_samples, dim), alpha=self.df / 2.0, beta=2.0)
        samples = z / tf.sqrt(chi2 / self.df)

        return self.loc + self.scale * samples




class GaussianMixturePrior(BasePrior):
    """
    Mixture of K diagonal Gaussians.
    """
    def __init__(self,
                 means: List[float],
                 stds: List[float],
                 weights: List[float]):
        """
        Args:
            means: list of length K.
            stds: list of length K.
            weights: list of length K, should sum to 1.
        """
        self.means = tf.constant(means, dtype=tf.float32)   # [K]
        self.stds  = tf.constant(stds,  dtype=tf.float32)   # [K]
        self.vars  = self.stds ** 2
        self.weights = tf.constant(weights, dtype=tf.float32)  # [K]

    def log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        # TODO: implement mixture log_prob if needed.
        raise NotImplementedError

    def grad_log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        """
        ∇ log Σ_k w_k N(θ | μ_k, σ_k^2) = Σ_k r_k ∇ log N_k
        where r_k are responsibilities.
        """
        # TODO: implement mixture gradient carefully.
        raise NotImplementedError

    def sample(self, num_samples: int, dim: int) -> tf.Tensor:
        # TODO: implement mixture sampling (choose component per sample, then sample Gaussian).
        raise NotImplementedError
    


class EmpiricalGaussianPrior(BasePrior):
    """
    Empirical Gaussian prior based on previously observed parameter vectors.

    θ ~ N(μ_emp, Σ_emp) with diagonal covariance estimated from data.
    """
    def __init__(self, theta_samples: np.ndarray):
        """
        Args:
            theta_samples: array of shape [N, D] from previous training runs.
        """
        theta_samples = np.asarray(theta_samples)
        self.mean_vec = theta_samples.mean(axis=0)
        self.std_vec  = theta_samples.std(axis=0) + 1e-8  # avoid zeros
        self.var_vec  = self.std_vec ** 2

    def log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        # TODO: implement if needed.
        raise NotImplementedError

    def grad_log_prob(self, theta: tf.Tensor) -> tf.Tensor:
        """
        Diagonal empirical Gaussian: ∇_θ log p(θ) = -(θ - μ_emp) / σ_emp^2
        """
        return -(theta - self.mean_vec) / self.var_vec

    def sample(self, num_samples: int, dim: int) -> tf.Tensor:
        # Note: dim should match the length of mean_vec; you can assert this.
        # TODO: implement sampling from N(mean_vec, diag(var_vec)).
        raise NotImplementedError
