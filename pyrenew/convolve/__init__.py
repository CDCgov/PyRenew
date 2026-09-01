"""
Low-level functions and classes for common
convolutions computed in renewal modeling.
"""

# backward compatibility, but moved. Issue deprecation warning
# for importing from `pyrenew.convolve`
from pyrenew.convolve.delays import (
    compute_delay_ascertained_incidence,
    compute_prop_already_reported,
)
from pyrenew.convolve.scanner import (
    BaseBackwardLookingConvolutionScanner,
    ConvolveAndMultiplyScanner,
    Scanner,
)
from pyrenew.convolve.scanner_factories import (
    new_convolve_scanner,
    new_double_convolve_scanner,
)

__all__ = [
    "compute_delay_ascertained_incidence",
    "compute_prop_already_reported",
    "new_convolve_scanner",
    "new_double_convolve_scanner",
    "Scanner",
    "BaseBackwardLookingConvolutionScanner",
    "ConvolveAndMultiplyScanner",
]
