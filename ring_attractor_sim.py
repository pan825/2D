import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Ring Attractor Network Simulation
# -----------------------------------------------------------------------------

def simulate_ring_attractor(
    N: int = 200,
    T: float = 1000.0,
    dt: float = 0.1,
    tau: float = 10.0,
    w0: float = 0.0,
    w1: float = 1.0,
    external_input: float = 0.0,
    noise_std: float = 0.01,
    activation_fn: str = "relu",
):
    """Simulate a continuous ring attractor network.

    Parameters
    ----------
    N : int
        Number of neurons uniformly distributed on the ring.
    T : float
        Total simulation time (ms).
    dt : float
        Integration time step (ms).
    tau : float
        Membrane time constant (ms).
    w0 : float
        Uniform (isotropic) component of the connectivity.
    w1 : float
        Amplitude of the cosine connectivity profile.
    external_input : float
        Constant external input added to all neurons.
    noise_std : float
        Standard deviation of Gaussian white noise added at each step.
    activation_fn : str
        Activation function to convert membrane potentials to firing rates.

    Returns
    -------
    t : ndarray, shape (steps,)
        Time vector.
    r : ndarray, shape (steps, N)
        Firing rates of all neurons over time.
    """

    steps = int(T / dt)
    t = np.arange(0, T, dt)

    # Preferred angles for each neuron on a ring
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

    # Connectivity matrix following a circular cosine kernel
    # w_ij = w0 + w1 * cos(theta_i - theta_j)
    theta_diff = theta[:, None] - theta[None, :]
    W = w0 + w1 * np.cos(theta_diff)
    W /= N  # normalize by N to keep input sizes independent of network size

    # Activation function
    if activation_fn == "relu":
        f = lambda x: np.maximum(0.0, x)
    elif activation_fn == "sigmoid":
        f = lambda x: 1.0 / (1.0 + np.exp(-x))
    else:
        raise ValueError(f"Unknown activation_fn '{activation_fn}'")

    # State variables: membrane potentials and firing rates
    v = np.random.randn(N) * 0.1  # small random initialization
    r = np.zeros((steps, N), dtype=float)

    sqrt_dt = np.sqrt(dt)

    for k in range(steps):
        # Compute recurrent input
        input_rec = W @ f(v)

        # Euler integration of membrane potential dynamics
        dv = (
            -v + input_rec + external_input
        ) * (dt / tau) + noise_std * sqrt_dt * np.random.randn(N)
        v += dv

        # Update firing rates array
        r[k] = f(v)

    return t, r


def plot_activity(t: np.ndarray, r: np.ndarray, sample_every: int = 10):
    """Visualize the firing rate activity as a heatmap over time."""
    plt.figure(figsize=(8, 6))
    plt.imshow(
        r[::sample_every].T,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 0, r.shape[1]],
        cmap="viridis",
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index (ordered by preferred angle)")
    plt.title("Ring Attractor Activity Map")
    plt.colorbar(label="Firing rate")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage of the simulator
    t, r = simulate_ring_attractor(
        N=200,
        T=2000.0,
        dt=0.1,
        tau=10.0,
        w0=0.0,
        w1=2.0,
        external_input=0.0,
        noise_std=0.02,
        activation_fn="relu",
    )

    plot_activity(t, r) 