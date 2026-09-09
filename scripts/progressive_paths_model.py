"""Compositional paraxial model for the progressive desktop path animation.

This is a finite, periodic, transverse-grid model, not an evaluation of an
infinite path integral.  Free evolution is the exact matrix exponential of
the spectral transverse Laplacian on that grid.  Inserting complete,
unmasked intermediate grids therefore leaves the field unchanged.

The localized source has Gaussian amplitude, with amplitude width 0.18.
The initial aperture transmits the central 49 samples of the base grid.
Removing that aperture changes the field; adding imaginary slices does not.
The old desktop animation's spherical-distance weights are not used here.

``FresnelModel(n).groups(b, aperture=False, x=4.5)`` returns one complex
contribution per transverse sample, ordered from bottom to top.  At multiple
slices, each such contribution groups all complete paths through that sample.
``initial_contributions(b)`` returns the 49 nonzero initial aperture terms on
the base grid.  ``profile`` returns complex amplitudes; ``intensity_profile``
returns their squared magnitudes.  All models use one common display phase.

Run ``python scripts/progressive_paths_model.py --check`` for numerical QA.
Only NumPy is required.  No files are written by the check.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np


TOTAL_DISTANCE = 9.0
MIDPOINT = TOTAL_DISTANCE / 2.0
WAVELENGTH = 0.5
WAVE_NUMBER = 2.0 * math.pi / WAVELENGTH
SOURCE_SIGMA = 0.18
BASE_GRID_SIZE = 385
BASE_SPACING = 5.6 / 48.0
PERIOD = BASE_GRID_SIZE * BASE_SPACING
GRID_SIZES = (385, 769, 1537)
APERTURE_HALF_HEIGHT = 2.8
DETECTOR_B_VALUES = np.linspace(-3.2, 3.2, 193)


def gaussian_source(y: np.ndarray, sigma: float = SOURCE_SIGMA) -> np.ndarray:
    """Continuous Gaussian amplitude, normalized in physical L2 units."""
    normalization = 1.0 / math.sqrt(sigma * math.sqrt(math.pi))
    return normalization * np.exp(-0.5 * (y / sigma) ** 2)


def _common_phase() -> complex:
    """Rotate the base-grid free amplitude at B=0 onto the positive real axis."""
    y = (np.arange(BASE_GRID_SIZE) - BASE_GRID_SIZE // 2) * BASE_SPACING
    q = 2.0 * math.pi * np.fft.fftfreq(BASE_GRID_SIZE, d=BASE_SPACING)
    coefficients = np.fft.fft(gaussian_source(y))
    evolution = np.exp(-0.5j * TOTAL_DISTANCE * q**2 / WAVE_NUMBER)
    at_center = np.mean(coefficients * evolution * np.exp(-1j * q * y[0]))
    return complex(np.exp(-1j * np.angle(at_center)))


DISPLAY_PHASE = _common_phase()


@dataclass(frozen=True)
class FresnelModel:
    """One transverse discretization of the same periodic physical domain.

    ``n=385, 769, 1537`` refines the mesh while holding ``period`` fixed.
    Field arrays contain physical amplitudes sampled on ``y``.  The FFT
    propagator already includes its integration normalization: do not insert
    extra powers of ``dy`` into propagation or grouped contributions.
    """

    n: int = BASE_GRID_SIZE
    period: float = PERIOD
    total_distance: float = TOTAL_DISTANCE
    wavelength: float = WAVELENGTH
    source_sigma: float = SOURCE_SIGMA

    def __post_init__(self) -> None:
        if self.n < 3 or self.n % 2 == 0:
            raise ValueError("The transverse grid must have an odd size >= 3.")
        if min(self.period, self.total_distance, self.wavelength, self.source_sigma) <= 0:
            raise ValueError("Physical scales must be positive.")

    @property
    def dy(self) -> float:
        return self.period / self.n

    @property
    def k(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @property
    def midpoint(self) -> float:
        return self.total_distance / 2.0

    @cached_property
    def y(self) -> np.ndarray:
        return (np.arange(self.n) - self.n // 2) * self.dy

    @cached_property
    def q(self) -> np.ndarray:
        return 2.0 * math.pi * np.fft.fftfreq(self.n, d=self.dy)

    @cached_property
    def source(self) -> np.ndarray:
        return gaussian_source(self.y, self.source_sigma).astype(complex)

    @cached_property
    def mid_field(self) -> np.ndarray:
        return self.propagate(self.source, self.midpoint)

    @cached_property
    def aperture_mask(self) -> np.ndarray:
        # Tolerance includes both boundary samples at +/-2.8 on the base grid.
        return (np.abs(self.y) <= APERTURE_HALF_HEIGHT + 1e-12).astype(float)

    @property
    def aperture_indices(self) -> np.ndarray:
        return np.flatnonzero(self.aperture_mask)

    def transfer(self, distance: float) -> np.ndarray:
        return np.exp(-0.5j * distance * self.q**2 / self.k)

    def propagate(self, field: np.ndarray, distance: float) -> np.ndarray:
        """Exact finite-grid free evolution; negative distances are allowed."""
        field = np.asarray(field, dtype=complex)
        if field.shape != (self.n,):
            raise ValueError(f"Expected a field with shape ({self.n},).")
        return np.fft.ifft(self.transfer(distance) * np.fft.fft(field))

    def endpoint_row(self, distance: float, b: float) -> np.ndarray:
        """Linear row mapping a sampled field to arbitrary real endpoint b.

        This uses the same trigonometric interpolation as ``profile``.
        It is not nearest-grid rounding or interpolation of intensity.
        """
        spectral_row = self.transfer(distance) * np.exp(1j * self.q * (b - self.y[0]))
        return np.fft.fft(spectral_row) / self.n

    def evaluate(self, field: np.ndarray, distance: float, b_values: np.ndarray) -> np.ndarray:
        """Propagate and continuously evaluate complex field amplitudes."""
        values = np.atleast_1d(np.asarray(b_values, dtype=float))
        if values.ndim != 1:
            raise ValueError("Detector coordinates must be one-dimensional.")
        spectrum = self.transfer(distance) * np.fft.fft(np.asarray(field, dtype=complex))
        evaluation = np.exp(1j * np.outer(values - self.y[0], self.q))
        return (evaluation @ spectrum) / self.n

    def groups(self, b: float, aperture: bool = False, x: float | None = None) -> np.ndarray:
        """Contributions grouped by position on one selected intermediate plane.

        ``aperture=True`` describes the physical aperture at the midpoint,
        which has 49 nonzero terms on the base grid.  Free groups can be
        evaluated at any intermediate x and always sum to the same endpoint
        amplitude.  Returned terms include the common display phase.
        """
        x = self.midpoint if x is None else float(x)
        if not 0.0 <= x <= self.total_distance:
            raise ValueError("Intermediate x must lie between source and detector.")
        if aperture and not math.isclose(x, self.midpoint, abs_tol=1e-12):
            raise ValueError("The physical aperture is fixed at the midpoint.")
        field = self.mid_field if x == self.midpoint else self.propagate(self.source, x)
        if aperture:
            field = field * self.aperture_mask
        return DISPLAY_PHASE * self.endpoint_row(self.total_distance - x, b) * field

    def initial_contributions(self, b: float) -> np.ndarray:
        """Nonzero physical-aperture terms (49 terms for the base model)."""
        return self.groups(b, aperture=True)[self.aperture_indices]

    def profile(self, b_values: np.ndarray | None = None, aperture: bool = False) -> np.ndarray:
        """Complex detector amplitudes, in the supplied endpoint order."""
        values = DETECTOR_B_VALUES if b_values is None else np.asarray(b_values, dtype=float)
        if aperture:
            amplitudes = self.evaluate(self.mid_field * self.aperture_mask, self.midpoint, values)
        else:
            amplitudes = self.evaluate(self.source, self.total_distance, values)
        return DISPLAY_PHASE * amplitudes

    def intensity_profile(self, b_values: np.ndarray | None = None, aperture: bool = False) -> np.ndarray:
        return np.abs(self.profile(b_values, aperture=aperture)) ** 2

    def amplitude(self, b: float, aperture: bool = False) -> complex:
        return complex(self.profile(np.array([b]), aperture=aperture)[0])

    def norm(self, field: np.ndarray) -> float:
        return float(self.dy * np.sum(np.abs(field) ** 2))


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def run_checks() -> dict[str, object]:
    """Check the physical and numerical properties used by the animation."""
    model = FresnelModel()
    rng = np.random.default_rng(4917)
    random_field = rng.normal(size=model.n) + 1j * rng.normal(size=model.n)
    split = model.propagate(model.propagate(random_field, 2.7), 6.3)
    direct = model.propagate(random_field, TOTAL_DISTANCE)
    composition = _relative_error(split, direct)
    norm_error = abs(model.norm(direct) / model.norm(random_field) - 1.0)

    # Includes off-grid endpoints and planes near both ends of propagation.
    group_error = 0.0
    for b in (-2.713, 0.0, 0.94, 3.111):
        for x in (0.13, 2.0, MIDPOINT, 7.3, 8.9):
            error = abs(np.sum(model.groups(b, x=x)) - model.amplitude(b))
            group_error = max(group_error, float(error))
        aperture_error = abs(np.sum(model.initial_contributions(b)) - model.amplitude(b, aperture=True))
        group_error = max(group_error, float(aperture_error))

    # Explicit 5^2 path expansion at two imaginary planes, with an arbitrary
    # input field and an off-grid detector: independent of Gaussian geometry.
    tiny = FresnelModel(n=5, period=7.0)
    tiny_input = rng.normal(size=5) + 1j * rng.normal(size=5)
    first = tiny.propagate(tiny_input, 2.1)
    middle_kernel = np.fft.ifft(tiny.transfer(3.7))
    last_row = tiny.endpoint_row(3.2, 0.731)
    enumerated = sum(
        last_row[j] * middle_kernel[(j - i) % tiny.n] * first[i]
        for i in range(tiny.n) for j in range(tiny.n)
    )
    tiny_direct = tiny.endpoint_row(TOTAL_DISTANCE, 0.731) @ tiny_input
    enumeration_error = float(abs(enumerated - tiny_direct))

    profiles = [FresnelModel(n=n).profile() for n in GRID_SIZES]
    refinement_errors = [_relative_error(profile, profiles[-1]) for profile in profiles[:-1]]
    # Double the padded domain while keeping the original physical spacing.
    wide_n = 2 * BASE_GRID_SIZE - 1
    wide = FresnelModel(n=wide_n, period=wide_n * BASE_SPACING)
    padding_error = _relative_error(model.profile(), wide.profile())

    # Independent unbounded-space Gaussian solution provides an additional
    # check against the periodic window and the sign of Fresnel evolution.
    complex_variance = SOURCE_SIGMA**2 + 1j * TOTAL_DISTANCE / WAVE_NUMBER
    normalization = 1.0 / math.sqrt(SOURCE_SIGMA * math.sqrt(math.pi))
    analytic = (
        DISPLAY_PHASE * normalization * SOURCE_SIGMA / np.sqrt(complex_variance)
        * np.exp(-DETECTOR_B_VALUES**2 / (2.0 * complex_variance))
    )
    analytic_error = _relative_error(profiles[-1], analytic)

    metrics = {
        "composition_relative_error": composition,
        "norm_relative_error": norm_error,
        "grouped_amplitude_max_absolute_error": group_error,
        "explicit_25_path_absolute_error": enumeration_error,
        "refinement_relative_errors_vs_1537": refinement_errors,
        "double_padding_relative_error": padding_error,
        "unbounded_gaussian_relative_error": analytic_error,
        "initial_opening_count": len(model.aperture_indices),
        "base_free_center_imaginary_part": abs(model.amplitude(0.0).imag),
    }
    assert len(model.aperture_indices) == 49, metrics
    assert composition < 1e-11 and norm_error < 1e-12, metrics
    assert group_error < 1e-11 and enumeration_error < 1e-11, metrics
    assert max(refinement_errors) < 1e-4, metrics
    assert padding_error < 1e-4 and analytic_error < 1e-4, metrics
    assert abs(model.amplitude(0.0).imag) < 1e-12, metrics
    return {
        "status": "passed",
        "metrics": metrics,
        "model": {
            "grid_sizes": GRID_SIZES,
            "fixed_period": PERIOD,
            "base_spacing": BASE_SPACING,
            "source_amplitude_sigma": SOURCE_SIGMA,
            "physical_aperture_x": MIDPOINT,
            "detector_distance": TOTAL_DISTANCE,
        },
        "limitations": [
            "Finite periodic-grid paraxial propagation with a localized Gaussian source.",
            "Removing the physical aperture changes the field; imaginary unmasked slices do not.",
            "Refinement checks concern free propagation after mask removal, not changing physical slit widths.",
            "Multiple-slice phasors are grouped path sums; displayed routes are representative.",
            "No finite stage is claimed to evaluate the infinite continuum path integral.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run numerical checks and print a JSON report")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(run_checks(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
