class ModelctlError(Exception):
    """An expected modelctl failure suitable for a concise CLI error."""


class ManifestError(ModelctlError):
    pass


class ValidationError(ModelctlError):
    pass
