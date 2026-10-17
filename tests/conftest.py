"""Offline plot tests must not require a desktop GUI or a working Tk/Qt install."""

import matplotlib


# This affects pytest only, not the application's interactive plotting backend.
matplotlib.use("Agg")
