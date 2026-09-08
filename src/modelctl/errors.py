class ModelctlError(Exception):
    """An expected modelctl failure suitable for a concise CLI error."""


class ManifestError(ModelctlError):
    pass


class ValidationError(ModelctlError):
    pass


class CatalogStaleViewError(ModelctlError):
    """A refresh would replace a non-empty catalog with an empty live view;
    the client's store view may be degraded."""
