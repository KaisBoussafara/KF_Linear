import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy.random as rnd
import os
import rir_generator as rir
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
def get_noise(SNR, x, seed=None):
    if seed is not None:
        rng = rnd.default_rng(seed)
    else:
        rng = rnd
    P_signal = np.var(x)
    P_noise = P_signal / 10 ** (SNR / 10)
    noise = np.sqrt(P_noise) * rng.normal(size=x.size)
    return noise

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

# Generate AR(1) input signal, convolve with impulse response, add noise
def generate_signals(w, L, N, SNR, alpha=0.9, seed=None, return_noise=False):
    """
    Generate the input AR(1) signal and the noisy desired signal.
    """
    x = get_signal(N, alpha=alpha)
    s = sig.convolve(w, x, mode='full')[:N]  # Convolution with the impulse response h

    e = get_noise(SNR, s, seed=seed)  # Generate noise
    d = s + e  # Add noise to the convolved signal

    return x, d, s, e  # Return input signal, noisy output, clean output, and noise

# Plot AR(1) signal and impulse response
def plot_signals(x, h, d, s, N, output_dir):
    fig, ax = plt.subplots(4, 1, figsize=(10, 10))

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

    # Plot clean output (convolution result)
    ax[2].plot(s[:N], label='Convolved Signal (s)', color='orange')
    ax[2].set_title("Convolved Signal (s)")
    ax[2].set_xlabel("Samples")
    ax[2].set_ylabel("Amplitude")
    ax[2].grid(True)

    # Plot noisy output
    ax[3].plot(d[:N], label='Noisy Signal (d)', color='red')
    ax[3].set_title("Noisy Signal (d)")
    ax[3].set_xlabel("Samples")
    ax[3].set_ylabel("Amplitude")
    ax[3].grid(True)

    plt.tight_layout()

    # Save figure to 'figures' directory
    output_file = os.path.join(output_dir, "ar_signal_impulse_response.png")
    plt.savefig(output_file)

    # Optionally show the plot (remove or comment this line if not needed)
    plt.show()


# Main experiment
L = 600  # Impulse response length
N = 1000  # Number of samples in AR(1) signal
SNR = 20  # Signal-to-noise ratio in dB
alpha = 0.9  # AR(1) process coefficient

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
x, d, s, e = generate_signals(h_true, L, N, SNR, alpha=alpha)

# Plot the signals
plot_signals(x, h_true, d, s, N, output_dir)


