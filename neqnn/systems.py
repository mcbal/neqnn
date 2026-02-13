import torch
from einops import einsum, rearrange


def entropy_production(beta, J, D):
    return beta * (
        einsum(J, D, "... i j, ... i j -> ... ")
        - einsum(J, D, "... j i, ... i j -> ... ")
    )


#
# `plefka_t_1_t_naive_mf` mean-field approximation of kinetic vector-spin system
#


def theta(x, J, m0):
    return x + einsum(J, m0, "... i j, ... j d -> ... i d")


def gamma(theta, beta, scale):
    return torch.sqrt(1 + beta**2 * (theta**2).sum(dim=-1, keepdims=True) / scale**2)


def phi(theta, beta, scale):
    return beta * theta / (1 + gamma(theta, beta, scale))


def m_plefka_t_1_t_naive_mf(theta, *, beta, scale):
    """Magnetizations."""
    return phi(theta, beta, scale)


def td_corr_plefka_t_1_t_naive_mf(theta1, J1, theta0, *, beta, scale):
    """Time-delayed correlations."""
    gamma1, gamma0 = map(lambda t: gamma(t, beta, scale), (theta1, theta0))
    m1, m0 = beta * theta1 / (1 + gamma1), beta * theta0 / (1 + gamma0)
    return (
        beta
        * J1
        * (
            (1 / (1 + gamma1))
            * (
                scale**2
                - rearrange(
                    einsum(m0, m0, "... j d, ... j d -> ... j"), "... j -> ... 1 j"
                )
            )
            - (1 / (scale**2 * gamma1))
            * (
                rearrange(
                    einsum(m1, m1, "... i d, ... i d -> ... i"), "... i -> ... i 1"
                )
                / (1 + rearrange(gamma0, "... j 1 -> ... 1 j"))
                - (
                    einsum(m1, m0, "... i d, ... j d -> ... i j") ** 2
                    / (scale**2 * rearrange(gamma0, "... j 1 -> ... 1 j"))
                )
            )
        )
    )
