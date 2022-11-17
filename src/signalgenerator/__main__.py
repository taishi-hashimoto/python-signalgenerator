import numpy as np
from . import Channel, Signal, SincPulse, Noise, DC, RectangularPulse


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Data acquisition parameters.
    sampling_frequency = 100e3  # [Hz]
    total_seconds = 10  # [s]

    print("dt = {:.2f} us".format(1/sampling_frequency*1e6))

    # Total length of the signal.
    size = int(total_seconds * sampling_frequency)

    # Time axis in seconds.
    taxis = np.arange(size) / sampling_frequency

    # Parameters for generating an electric field pulse of lightning.

    # Simulated lighting pulse.
    sg_lit = Channel(
        Signal(
            SincPulse(0, 3),
            delay=3.5,  # [s]
            # width=25e-6,  # [s],
            width=0.1,  # [s],
            amplitude=0.5,  # [V]
        ),
        Noise(0.01),
        DC(0.1))

    # GPS PPS.
    sg_pps = Channel(
        Signal(
            RectangularPulse(),
            delay=0.6,  # [s]
            width=0.1,  # [s],
            amplitude=1,  # [V]
            interval=1.
        ),
        Noise(0.005),
        DC(-0.2))

    # Samples it.
    sig = sg_lit.sample(size, sampling_frequency)
    pps = sg_pps.sample(size, sampling_frequency)

    # Plot it.
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

    axes[0].plot(taxis, sig)
    axes[0].set_ylabel("E-field [V]")

    axes[1].plot(taxis, pps)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("PPS [V]")
    fig.tight_layout()
    plt.show()
