"""Fluidity (Laban 'Flow') via SPARC.

Primary backend : pyEyesWeb's Smoothness (InfoMus / Casa Paganini).
Fallback backend: a numpy SPARC (Balasubramanian et al. 2015) so the installation never
                  crashes if the library's API changes -- it is under active refactoring
                  and the docs show *different* signatures between versions.

The adapter probes the installed version at start-up with a synthetic signal and only keeps
pyEyesWeb if it returns a finite number. Check the printed backend name when you start.
"""
import numpy as np


def sparc_numpy(speed, fs, pad=4, fc=10.0, amp_th=0.05):
    """Spectral arc length of a speed profile. More negative = less smooth."""
    speed = np.asarray(speed, float)
    if len(speed) < 8 or speed.max() < 1e-9:
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(speed))) + pad))
    f = np.arange(nfft) * fs / nfft
    mag = np.abs(np.fft.fft(speed, nfft))
    mag /= mag.max()
    sel = f <= fc
    f, mag = f[sel], mag[sel]
    idx = np.where(mag >= amp_th)[0]
    if len(idx) < 2:
        return float("nan")
    f, mag = f[idx[0]:idx[-1] + 1], mag[idx[0]:idx[-1] + 1]
    return float(-np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(mag) ** 2)))


class SparcBackend:
    """Uniform interface: push(speed) -> sparc (nan until the window is full)."""

    def __init__(self, rate_hz, window_len):
        self.rate_hz, self.window_len = rate_hz, window_len
        self.name = "numpy-fallback"
        self._buf = []
        self._pe = None
        self._try_pyeyesweb()

    # -- pyeyesweb discovery ---------------------------------------------------
    def _try_pyeyesweb(self):
        try:
            import pyeyesweb  # noqa: F401
        except Exception:
            return
        Smooth = None
        for path in ("pyeyesweb.low_level", "pyeyesweb.mid_level",
                     "pyeyesweb.mid_level.smoothness", "pyeyesweb"):
            try:
                mod = __import__(path, fromlist=["Smoothness"])
                Smooth = getattr(mod, "Smoothness")
                break
            except Exception:
                continue
        try:
            from pyeyesweb.data_models.sliding_window import SlidingWindow
        except Exception:
            try:
                from pyeyesweb.data_models import SlidingWindow
            except Exception:
                return
        if Smooth is None:
            return

        def build():
            try:
                sm = Smooth(rate_hz=self.rate_hz, signal_type="velocity")
            except TypeError:
                sm = Smooth(rate_hz=self.rate_hz)
            for kw in ({"max_length": self.window_len, "n_signals": 1, "n_dims": 1},
                       {"max_length": self.window_len, "n_columns": 1}):
                try:
                    return sm, SlidingWindow(**kw)
                except TypeError:
                    continue
            raise RuntimeError("unknown SlidingWindow signature")

        def push(sm, win, v):
            for sample in ([[v]], [v]):               # new API wants 2-D, older wants 1-D
                try:
                    win.append(sample)
                    return
                except Exception:
                    continue
            raise RuntimeError("cannot append to SlidingWindow")

        def read(res):
            val = getattr(res, "sparc", None)
            if val is None and isinstance(res, dict):
                val = res.get("sparc")
            return float(val)

        try:                                           # probe with a synthetic bell-shaped speed
            sm, win = build()
            t = np.linspace(0, 1, self.window_len)
            last = float("nan")
            for v in np.sin(np.pi * t) ** 2:
                push(sm, win, float(v))
                if len(win) >= self.window_len:
                    last = read(sm(win))
            if np.isfinite(last):
                self._pe = (sm, win, push, read)
                self.name = "pyeyesweb"
        except Exception as e:                         # any API mismatch -> fallback
            print(f"[smoothness] pyeyesweb probe failed ({e!r}); using numpy fallback")

    # -- per-frame ---------------------------------------------------------------
    def push(self, speed):
        if self._pe is not None:
            sm, win, push, read = self._pe
            push(sm, win, float(speed))
            if len(win) < self.window_len:
                return float("nan")
            try:
                return read(sm(win))
            except Exception:
                return float("nan")
        self._buf.append(float(speed))
        self._buf = self._buf[-self.window_len:]
        if len(self._buf) < self.window_len:
            return float("nan")
        return sparc_numpy(self._buf, self.rate_hz)
