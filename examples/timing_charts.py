import matplotlib.pyplot as plt
from signalgenerator import (
    nsamples, timeaxis,
    SignalGenerator, Channel, Signal,
    RectangularPulse, Step
)

# Data acquisition parameters.
sampling_frequency = 100e3  # [Hz]
total_seconds = 10  # [s]

# Total length of the signal.
size = nsamples(sampling_frequency, total_seconds)

# Time axis for plotting.
taxis = timeaxis(sampling_frequency, size)

sg = SignalGenerator(
    Channel(
        Signal(
            RectangularPulse(),
            delay=1.5,  # [s]
            width=1.2,  # [s]
            amplitude=1.,  # [V]
            interval=3,  # [s]
        ),
    ),
    Channel(
        Signal(
            Step(),
            delay=1.5
        ),
    ),
    Channel(
        Signal(
            RectangularPulse(),
            delay=0.3,
            width=0.5,
            amplitude=0.7,
            interval=3.,
        ),
        Signal(
            RectangularPulse(),
            delay=1.5,
            width=1.5,
            interval=3.
        ),
    )
)

samples = sg.sample(size, sampling_frequency)

print(samples)

samples.plot.line(x="time", row="channel", figsize=(16, 6))

plt.show()
