import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy.random as rnd
import rir_generator as rir
import librosa as rosa
import tqdm

# Generate AR(1) signal
output_dir = "figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
def get_signal(N, amp=1, alpha=0.9):
    """
    Generate an AR(1) process.
    """
    b = [1, 0]
    a = [1, -alpha]
    x = amp * rnd.randn(N)
    y = sig.lfilter(b, a, x)
    return y

# Generate noise with desired SNR
def get_noise(SNR, s, seed=None):
    if seed is not None:
        rng = rnd.default_rng(seed)
    else:
        rng = rnd
    P_signal = np.var(s)
    P_noise = P_signal / 10 ** (SNR / 10)
    noise = np.sqrt(P_noise) * rng.normal(size=s.size)
    return noise, P_noise

# Load the room impulse response (RIR)
def load_filter(L=600, rir_options=None):
    """
    Generate the impulse response (RIR) using rir-generator.
    """
    if rir_options is None:
        rir_options = dict(
            c=340,                    # Sound velocity (m/s)
            fs=8e3,                   # Sample frequency (samples/s)
            r=[2, 1.5, 1],            # Receiver position(s) [x y z] (m)
            s=[2, 3.5, 2],            # Source position [x y z] (m)
            L=[5, 4, 6],              # Room dimensions [x y z] (m)
            reverberation_time=0.225,  # Reverberation time (s)
        )
    h = rir.generate(nsample=L, **rir_options).ravel()
    return h

# Generate input signal, convolve with impulse response, add noise
def generate_signals(w, L, N, SNR, alpha=0.9, seed=None, return_noise=False):
    """
    Generate the input signal and the noisy desired signal.
    """
    x = get_signal(N, alpha=alpha)
    s = sig.convolve(w, x, mode='full')[:N]  # Convolution with the impulse response h

    e, v_e = get_noise(SNR, s, seed=seed)  # Generate noise
    d = s + e  # Add noise to the convolved signal

    return x, d, v_e  # Return input signal, noisy output, clean output, and noise
def get_frames(x, L):
    N = x.size
    if N < L:
        xx = np.pad(x, (0, L-N), mode='constant')
    else:
        xx = x.copy()

    x_pad = np.pad(xx, (L-1, 0), mode='constant')
    X = rosa.util.frame(x_pad, frame_length=L, hop_length=1)[::-1, :]

    return X
# Plot AR(1) signal and impulse response
def plot_signals(x, h, N, output_dir):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plot AR(1) signal
    ax[0].plot(x[:N], label='AR(1) Signal')
    ax[0].set_title("AR(1) Signal")
    ax[0].set_xlabel("Samples")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid(True)

    # Plot impulse response
    ax[1].plot(h, label='Impulse Response (h)', color='green')
    ax[1].set_title("Impulse Response (h)")
    ax[1].set_xlabel("Samples")
    ax[1].set_ylabel("Amplitude")
    ax[1].grid(True)

    # # Plot clean output (convolution result)
    # ax[2].plot(s[:N], label='Convolved Signal (s)', color='orange')
    # ax[2].set_title("Convolved Signal (s)")
    # ax[2].set_xlabel("Samples")
    # ax[2].set_ylabel("Amplitude")
    # ax[2].grid(True)
    #
    # # Plot noisy output
    # ax[3].plot(d[:N], label='Noisy Signal (d)', color='red')
    # ax[3].set_title("Noisy Signal (d)")
    # ax[3].set_xlabel("Samples")
    # ax[3].set_ylabel("Amplitude")
    # ax[3].grid(True)

    plt.tight_layout()

    # Save figure to 'figures' directory
    output_file = os.path.join(output_dir, "ar_signal_impulse_response.png")
    plt.savefig(output_file)

    # Optionally show the plot (remove or comment this line if not needed)
    plt.show()

# Plot the signals
# plot_signals(x, h_true, d, output_dir)

def mse(true_output, predicted_output):
    return np.mean((true_output - predicted_output) ** 2)

# First derivative g_t and second derivative h_t from the log-likelihood
def g_t(z_t, y_t, beta):
    return (z_t - y_t) / beta

def h_t(beta):
    return 1 / beta

# Kalman filter for AR(1) process with impulse response and Gaussian noise
def kalman_filter(X, y, beta, eps=0.1, v_0=0.01):
    """
    Kalman filter for AR(1) process with impulse response and Gaussian noise.
    X: Input matrix of AR(1) process frames
    y: Noisy desired output
    beta: Noise variance (from Gaussian noise)
    eps_in: Process noise variance
    v_0: Initial variance
    num_iterations: Number of iterations
    """
    n_samples, n_features = X.shape
    mu = np.zeros(n_features)  # Initialize weights (mean estimate)
    V = np.eye(n_features) * v_0  # Initialize covariance matrix

    if len(np.array(eps).shape) == 0:
        eps = np.ones(n_samples) * eps # Process noise
    loss = np.zeros(n_samples)
    # Kalman Filter Iterations
    for t in tqdm.trange(n_samples):
        # idx = i % n_samples  # Ensure we cycle through the dataset
        x = X[t, :]  # Current input vector (AR(1) frame)
        # y_t = y[t]  # Current observation (noisy desired output)

        # Prediction Step: Update Covariance
        V = V + eps[t] * np.eye(n_features)

        # Compute omega_t (Eq 38)
        omega_t = x.T @ V @ x

        # Compute the first and second derivatives (g_t and h_t)
        z_t = x @ mu # Predicted observation
        g_value = g_t(z_t, y[t], beta)  # First derivative of log-likelihood
        h_value = h_t(beta)  # Second derivative of log-likelihood

        # Update mean
        mu = mu - (V @ x) * g_value / (1 + h_value * omega_t)
        # Update Covariance (V)
        V = V - (V @ x[:, None] @ x[None, :] @ V) * h_value / (1 + h_value * omega_t)
        loss[t] = ((y[t] - z_t) ** 2)/2*beta

    return mu, V, loss

def reconstruct_output(X, weights):
    return np.dot(X, weights)  # Linear model prediction: y_t = w^T x_t + epsilon

# Plot the desired output vs reconstructed output
def plot_desired_vs_reconstructed(d, predicted_output):
    plt.figure(figsize=(10, 5))
    plt.plot(d, label='Desired Output (d[n])', color='blue')
    plt.plot(predicted_output, label='Reconstructed Output (Predicted)', color='red', linestyle='--')
    plt.title("Desired Output vs Reconstructed Output")
    plt.xlabel("Sample Index")
    plt.ylabel("Output Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss(loss):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the loss in dB scale
    ax.plot(10 * np.log10(loss), color='teal', linewidth=2)

    ax.set_title("Loss Function over Time", fontsize=16, fontweight='bold')
    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("Loss (dB)", fontsize=14)

    ax.grid(True, linestyle='--', alpha=0.7)

    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_facecolor('#f5f5f5')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Plot the desired output vs reconstructed output in subplots for different SNRs
def plot_desired_vs_reconstructed_subplots(SNR_values, d_list, predicted_output_list,output_dir):
    fig, axs = plt.subplots(len(SNR_values), 1, figsize=(10, 10))

    for i, SNR in enumerate(SNR_values):
        axs[i].plot(d_list[i], label='Desired Output (d[n])', color='blue')
        axs[i].plot(predicted_output_list[i], label='Reconstructed Output', color='red', linestyle='--')
        axs[i].set_title(f"Desired vs Reconstructed Output (SNR = {SNR} dB)")
        axs[i].set_xlabel("Sample Index")
        axs[i].set_ylabel("Output Value")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    # plt.show()
    file_path = os.path.join(output_dir, 'desired_vs_reconstructed_for_different_SNRs.png')
    plt.savefig(file_path)
    print(f"Figure saved at {file_path}")

# Plot the loss functions for different SNRs in subplots
def plot_loss_subplots(SNR_values, loss_list,output_dir):
    fig, axs = plt.subplots(len(SNR_values), 1, figsize=(10, 10))

    for i, SNR in enumerate(SNR_values):
        axs[i].plot(10 * np.log10(loss_list[i]), color='teal', linewidth=2)
        axs[i].set_title(f"Loss Function over Time (SNR = {SNR} dB)")
        axs[i].set_xlabel("Iterations")
        axs[i].set_ylabel("Loss (dB)")
        axs[i].grid(True)

    plt.tight_layout()
    # Save the figure in the output directory
    file_path = os.path.join(output_dir, 'Loss_for_different_SNRs.png')
    plt.savefig(file_path)
    print(f"Figure saved at {file_path}")


# Running the Kalman filter and plotting results for different SNR
def run_kf_with_different_snr(snr_values, h_true, L, N, alpha,output_dir):
    d_list = []
    predicted_output_list = []
    loss_list = []

    for SNR in snr_values:
        print(f"Running Kalman Filter with SNR = {SNR} dB")

        # Generate signals with current SNR
        x, d, v_e = generate_signals(h_true, L, N, SNR, alpha=alpha)
        d_list.append(d)

        # Convert input to frames
        X = get_frames(x, L).T

        # Apply the Kalman filter
        mu, V, loss = kalman_filter(X, d, v_e, eps=0.001, v_0=0.01)
        loss_list.append(loss)

        # Reconstruct the output using the estimated weights
        reconstructed_output = reconstruct_output(X, mu)
        predicted_output_list.append(reconstructed_output)

    # Plot the desired output vs reconstructed output for different SNRs
    plot_desired_vs_reconstructed_subplots(snr_values, d_list, predicted_output_list,output_dir)

    # Plot the loss functions for different SNRs
    plot_loss_subplots(snr_values, loss_list, output_dir)


# Plot the desired output vs reconstructed output in subplots for different sample sizes N
def plot_desired_vs_reconstructed_subplots_Samples(sample_sizes, d_list, predicted_output_list,output_dir):
    fig, axs = plt.subplots(len(sample_sizes), 1, figsize=(10, 10))

    for i, N in enumerate(sample_sizes):
        axs[i].plot(d_list[i], label='Desired Output (d[n])', color='blue')
        axs[i].plot(predicted_output_list[i], label='Reconstructed Output', color='red', linestyle='--')
        axs[i].set_title(f"Desired vs Reconstructed Output (N = {N})")
        axs[i].set_xlabel("Sample Index")
        axs[i].set_ylabel("Output Value")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    # plt.show()
    # Save the figure in the output directory
    file_path = os.path.join(output_dir, 'desired_vs_reconstructed_for_different_sample_sizes_N.png')
    plt.savefig(file_path)
    print(f"Figure saved at {file_path}")


# Plot the loss functions for different sample sizes N
def plot_loss_subplots_Samples(sample_sizes, loss_list,output_dir):
    fig, axs = plt.subplots(len(sample_sizes), 1, figsize=(10, 10))

    for i, N in enumerate(sample_sizes):
        axs[i].plot(10 * np.log10(loss_list[i]), color='teal', linewidth=2)
        axs[i].set_title(f"Loss Function over Time (N = {N})")
        axs[i].set_xlabel("Iterations")
        axs[i].set_ylabel("Loss (dB)")
        axs[i].grid(True)

    plt.tight_layout()
    # Save the figure in the output directory
    file_path = os.path.join(output_dir, 'Loss_for_different_sample_sizes_N.png')
    plt.savefig(file_path)
    print(f"Figure saved at {file_path}")


# Running the Kalman filter and plotting results for different sample sizes N
def run_kf_with_different_samples(sample_sizes, h_true, L, SNR, alpha,output_dir):
    d_list = []
    predicted_output_list = []
    loss_list = []

    for N in sample_sizes:
        print(f"Running Kalman Filter with N = {N} samples")

        # Generate signals with current sample size N
        x, d, v_e = generate_signals(h_true, L, N, SNR, alpha=alpha)
        d_list.append(d)

        # Convert input to frames
        X = get_frames(x, L).T

        # Apply the Kalman filter
        mu, V, loss = kalman_filter(X, d, v_e, eps=0.001, v_0=0.01)
        loss_list.append(loss)

        # Reconstruct the output using the estimated weights
        reconstructed_output = reconstruct_output(X, mu)
        predicted_output_list.append(reconstructed_output)

    # Plot the desired output vs reconstructed output for different sample sizes
    plot_desired_vs_reconstructed_subplots_Samples(sample_sizes, d_list, predicted_output_list,output_dir)

    # Plot the loss functions for different sample sizes
    plot_loss_subplots_Samples(sample_sizes, loss_list,output_dir)

# Main experiment
L = 600  # Impulse response length
N = 3000  # Number of samples in AR(1) signal
SNR = 20  # Signal-to-noise ratio in dB
alpha = 0.9  # AR(1) process coefficient
sample_sizes = [600, 1000, 2000, 3000]  # Different numbers of samples N
SNR_values = [0, 10, 20]  # Different SNR levels in dB
# Load the impulse response
rir_options = dict(
    c=340,                    # Sound velocity (m/s)
    fs=8e3,                   # Sample frequency (samples/s)
    r=[2, 1.5, 1],            # Receiver position(s) [x y z] (m)
    s=[2, 3.5, 2],            # Source position [x y z] (m)
    L=[5, 4, 6],              # Room dimensions [x y z] (m)
    reverberation_time=0.225,  # Reverberation time (s)
)

h_true = load_filter(L, rir_options)  # True impulse response

# Generate AR(1) signal and noisy output
x, d, v_e = generate_signals(h_true, L, N, SNR, alpha=alpha)

X = get_frames(x, L).T

# # Apply the Kalman filter
# # mu, V, loss = kalman_filter(X, d, v_e, eps=0.001, v_0=0.01)
#
# # Plot or evaluate results
# print("Final estimated weights (mean):", mu)
# print("Final estimated covariance:", V)
#
#
#
#
# reconstructed_output = reconstruct_output(X, mu)
# print(d, "\n", reconstructed_output)
# # Plot the desired output vs reconstructed output
# plot_desired_vs_reconstructed(d, reconstructed_output)
# #plot Loss
# plot_loss(loss)
# print("Final estimated weights (mean):",mu)

# Assuming you have already loaded h_true from RIR generator
run_kf_with_different_snr(SNR_values, h_true, L, N, alpha,output_dir)

run_kf_with_different_samples(sample_sizes, h_true, L, SNR, alpha,output_dir)
