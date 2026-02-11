import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__ or __name__)  # load from package metadata
except importlib.metadata.PackageNotFoundError:
    # Local source checkout without editable install.
    __version__ = "0.0.0"
