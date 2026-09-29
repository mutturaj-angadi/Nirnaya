"""Exception types raised by nirnaya-presolve."""


class PresolveError(Exception):
    """Base class for all nirnaya-presolve errors."""


class ModelBuildError(PresolveError):
    """Raised when the reduced model cannot be reconstructed via core's
    public constructors (should only happen on a genuine programming bug,
    since presolve validates as it goes)."""


class RecoveryError(PresolveError):
    """Raised when a reduced-space solution cannot be mapped back to the
    original variable/constraint space (e.g. missing values, wrong status)."""
