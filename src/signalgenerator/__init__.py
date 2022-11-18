"A sampled waveform generator."
import numpy as np
import xarray as xr
from scipy.optimize import newton, brent
from math import isclose
from typing import List, Union


__all__ = [
    "nsamples", "timeaxis", "Waveform", "RectangularPulse", "Step",
    "SincPulse", "Noise", "DC", "Signal", "Channel", "SignalGenerator"
]


def nsamples(fs: float, duration: float):
    """Utility function to compute the number of samples.

    Parameters
    ----------
    fs: float
        The sampling frequency in Hz.
    duration:float
        The duration in seconds.

    Returns
    -------
    int
        The number of samples.
    """
    return int(duration * fs)


def timeaxis(fs: float, n: int):
    """Utility function to build time axis for plotting.

    Parameters
    ----------
    fs: float
        The sampling frequency in Hz.
    n: int
        The number of samples.

    Returns
    -------
    ndarray
        Time axis for plotting.
    """
    return np.arange(n) / fs


class Waveform:
    """
    Basic waveform.
    """

    def __init__(self):
        pass

    def width(self):
        return 1.

    def hi(self):
        return 1.

    def lo(self):
        return 0.

    def position(self):
        return 0.

    def domain(self):
        return 0., 1.

    def __call__(self, x):
        return np.where(np.logical_or(x < 0, x > 1), self.lo(), self.hi())


class RectangularPulse(Waveform):
    """
    A rectangular waveform.
    """
    def __init__(self):
        super().__init__()


class Step(Waveform):
    "Stepwise function."

    def __init__(self, down=False):
        self._down = down

    def width(self):
        return np.inf

    def domain(self):
        return -np.inf, np.inf

    def __call__(self, x):
        if self._down:
            cond = x > 0
        else:
            cond = x < 0
        return np.where(cond, self.lo(), self.hi())


class SincPulse(Waveform):
    """A sinc pulse.

    The sinc function is filtered by a raised-cosine function with the roll-off
    factor of 1.
    """
    def __init__(self, left: int = 0, right: int = 3):
        """Initialize a waveform.
        The domain of this function is [-m, n] where m := left + 1 > 1 and
        n := right + 1 > 1.
        The filter function is scaled as [-pi, pi] -> [-m, n].
        Small integers are recommended for left and right.
        Incorrect result will be returned for left or right larger than 14.

        Parameters
        ----------
        left:
            The number of peaks in the head.
        right:
            The number of peaks in the tail.
        """
        super().__init__()
        self.m = left + 1
        self.n = right + 1

        x_p = brent(lambda x: -self(x), brack=(-1, 0, 1))
        y_p = self(x_p)

        # Get the pulse width.
        l_w = newton(lambda x: self(x) - y_p / 2, (-1 + x_p) / 2)
        r_w = newton(lambda x: self(x) - y_p / 2, (+1 + x_p) / 2)
        p_w = r_w - l_w

        self._x = x_p
        self._h = y_p
        self._w = p_w

    def domain(self):
        return -self.m, self.n

    def width(self):
        return self._w

    def hi(self):
        return self._h

    def lo(self):
        return 0.

    def position(self):
        return self._x

    def _model(self, x):
        return np.sinc(x) * (
            np.cos((2 * (x + self.m) / (self.n + self.m) - 1) * np.pi) + 1)

    def __call__(self, x):
        return np.where(
            np.logical_and(-self.m <= x, x <= self.n),
            self._model(x),
            self.lo())


class Noise(Waveform):
    """Base class for the noise.
    This class represents a standard normal distribution (mean = 0, std = 1).
    """

    def __init__(self, std: float = 1.):
        super().__init__()
        self._s = std

    def width(self):
        return np.inf

    def hi(self):
        return 1.

    def lo(self):
        return 0.

    def position(self):
        return 0.

    def domain(self):
        return -np.inf, np.inf

    def __call__(self, x):
        return np.random.normal(0, self._s, np.shape(x))


class DC(Waveform):
    """
    Direct current (DC).
    """

    def __init__(self, level=1.):
        super().__init__()
        self._l = level

    def width(self):
        return np.inf

    def hi(self):
        return self._l

    def lo(self):
        return 0.

    def position(self):
        return 0.

    def domain(self):
        return -np.inf, np.inf

    def __call__(self, x):
        return np.full_like(x, self._l)


class Signal:
    def __init__(
        self,
        waveform,
        amplitude=1.,
        delay=0.,
        width=None,
        interval=None
    ):
        """Bind a waveform in a physical dimension.

        The LOW and HIGH levels are normalized to [0., amplitude].

        Parameters
        ----------
        waveform:
            The base waveform.
        amplitude:
            The peak height. Default is 1. If None, no scaling occurs for the
            amplitude.
        delay:
            The time of pulse in an arbitrary dimension, but it must be the
            same as pulse_width.
        width:
            The width of the signal in time. If not specified, the x_scale is
            set to 1.
        interval:
            The repetition interval in second. If specified, the waveform is
            repeated by this.
        """
        self.waveform = waveform
        self.amplitude = amplitude
        self.delay = delay
        self.width = width
        self.ipp = interval
        if self.width is None:
            self.x_scale = 1.
        else:
            self.x_scale = self.waveform.width() / self.width
        if (
            self.waveform.hi() - self.waveform.lo() == 0. or
            self.amplitude is None
        ):
            self.y_scale = 1.
        else:
            self.y_scale = self.amplitude / (
                self.waveform.hi() - self.waveform.lo())
        self.x_offset = self.waveform.position()

    def single(self, t) -> np.ndarray:
        """
        Evaluate the waveform at the specified time.
        Single shot.

        Parameters
        ----------
        t:
            Time in an arbitrary dimension (but the same as self.delay).
            Can be an array.

        Returns
        -------
        ndarray
            An array representing the signal.
        """
        return (
            self.waveform((t - self.delay) * self.x_scale + self.x_offset) -
            self.waveform.lo()
        ) * self.y_scale

    def sample(
        self,
        n: int,
        fs: float,
        oversample: int = 1,
        undersample: Union[bool, int] = True
    ):
        """
        Sample a single or repeated waveform(s) with n sample points at the
        specified sampling frequency. If interval is set, this function can
        generate repeated pulse waveform. In that case, oversample option may
        be required to make the calculation accurate. If specified, the signal
        is internally oversampled, and returned with decimation.

        Parameters
        ----------
        n:
            The number of sample points.
        fs:
            The sampling frequency.
        oversample:
            The oversample factor for the repeated pulse mode.
        undersample:
            The decimation factor for the repeated pulse mode.

        Returns
        -------
        ndarray
            An array of the length n. The sampled waveform.
        """
        if undersample is True:
            undersample = oversample  # Same.
        elif undersample is None or undersample is False:
            undersample = 1
        l_ = n * oversample  # Oversampled length.
        t = np.arange(l_) / (fs * oversample)  # Oversampled time series.
        if self.ipp is None:
            w = self.single(t)
        else:  # Repeated mode.
            s = float(fs * self.ipp * oversample)
            if not isclose(s - int(s), 0):
                raise ValueError((
                        "fs * ipp * oversample must be an integer, "
                        "where remaining is {:e}"
                    ).format(s-int(s)))
            s = int(s)  # Oversampled step
            # Zero-padded waveform.
            w = np.zeros(1 << (2*l_-1).bit_length())
            w[:l_] = self.single(t)
            i = np.zeros_like(w)
            o = np.zeros_like(w)
            i[:l_:s] = 1
            o[:oversample] = 1 / oversample
            # Pulse repetition and oversampling are calculated in frequency
            # domain.
            w = np.real(
                np.fft.ifft(
                    np.fft.fft(w) * np.fft.fft(o) * np.fft.fft(i)))
        # Undersampling.
        return w[:l_:undersample]

    def __add__(self, other):
        return Channel(self, other)


class Channel:
    """
    A single channel of the signal generator.
    """

    def __init__(self, *signals, name: str = None):
        self._name = name
        self._signals = []
        for each in signals:
            if isinstance(each, Channel):
                self._signals.extend(each._signals)
            elif isinstance(each, Signal):
                self._signals.append(each)
            elif isinstance(each, Waveform):
                self._signals.append(Signal(each, amplitude=None))
            else:
                raise ValueError("Waveform or Signal is required.")

    @property
    def name(self) -> str:
        return self._name if self._name else ""

    @property
    def signals(self) -> List[Signal]:
        return self._signals

    def sample(self, n, fs, oversample=1, undersample=True):
        return np.sum([
            signal.sample(n, fs, oversample, undersample)
            for signal in self.signals
        ], axis=0)

    def __add__(self, other):
        return Channel(*self._signals, other)


class SignalGenerator:
    "Signal generator with multiple channels."

    def __init__(self, *channels):
        self._channels = []
        for each in channels:
            if isinstance(each, SignalGenerator):
                self._channels.extend(each._channels)
            elif isinstance(each, Channel):
                self._channels.append(each)
            elif isinstance(each, Signal):
                self._channels.append(Channel(each))
            elif isinstance(each, Waveform):
                self._channels.append(Channel(Signal(each, amplitude=None)))
            else:
                raise ValueError("Channel, Waveform or Signal is required.")

    @property
    def channels(self) -> List[Channel]:
        "Channels of the signal generator."
        return self._channels

    def sample(
        self,
        n: int,
        fs: float,
        oversample: int = 1,
        undersample: Union[bool, int] = True
    ) -> xr.DataArray:
        """Sample all channels and pack the result in a `xarray.DataArray`.

        Parameters
        ----------
        See `Signal.sample`.

        Returns
        -------
        xarray.DataArray
            Sampled waveforms.
        """
        data = np.vstack([
            ch.sample(n, fs, oversample, undersample)
            for ch in self.channels
        ])
        dims = ["channel", "time"]
        names = []
        for ic, ch in enumerate(self.channels, 1):
            if ch.name:
                names.append(ch.name)
            else:
                names.append(f"Ch.{ic}")
        taxis = timeaxis(fs, n)
        coords = {
            "channel": names,
            "time": taxis,
        }
        return xr.DataArray(
            data, coords=coords, dims=dims,
            attrs={
                "fs": fs,
                "oversample": oversample,
                "undersample": undersample,
            })

    def __add__(self, other):
        return SignalGenerator(*self._channels, other)
