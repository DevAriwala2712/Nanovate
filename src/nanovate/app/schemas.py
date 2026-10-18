"""Pydantic request / response models for /api/v1 (units: dose e-/px, lengths nm = px, times ms)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

API_VERSION = "v1"


class SimulateParams(BaseModel):
    """SimulatedSource parameters. ``shift`` null = drawn from the acquisition stream, as in the experiments."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)   # no "50" -> 50, true -> 1, NaN, inf
    family: str = "manhattan"
    kind: str = "bridge"
    dose: float = Field(50.0, description="e-/px; one of the Exp-2 doses")
    ler_sigma: float = Field(1.0, ge=0.5, le=2.5, description="nm")
    grain: float = Field(0.15, ge=0.0, le=0.30, description="conductor grain contrast, fraction")
    shift: Optional[list[float]] = Field(None, min_length=2, max_length=2, description="(dy, dx) px, each within +-8")
    warp: float = Field(0.0, ge=0.0, le=1.0, description="px")
    seed: int = Field(0, ge=0, le=2**31 - 1)
    split: str = "live"
    layout_id: int = Field(0, ge=0, le=999)
    die: int = Field(0, ge=0, le=899)


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_id: Optional[str] = Field(None, max_length=64)
    simulate: Optional[SimulateParams] = None
    include_arrays: bool = Field(False, description="also return float32 arrays (base64), for tests and reports")

    @model_validator(mode="after")
    def _one_source(self):
        if (self.input_id is None) == (self.simulate is None):
            raise ValueError("give exactly one of input_id or simulate")
        return self


class Envelope(BaseModel):
    api_version: str = API_VERSION
    request_id: str
    commit: str
    config_hash: str


class HealthResponse(Envelope):
    version: str
    cnn_loaded: bool
    cnn_threshold: float
    frozen_cfg: dict[str, Any]
    dose_grid: list[float]
    d2d_rules_cached: list[float]
    calibrations_cached: list[str]
    synthetic_note: str


class SimulateResponse(Envelope):
    input_id: str
    seed: int
    provenance: str
    source_type: str
    params: dict[str, Any]
    images: dict[str, Any]
    truth: Optional[dict[str, Any]]
    warnings: list[str]
    arrays: Optional[dict[str, Any]] = None


class VerifyResponse(Envelope):
    input_id: str
    seed: int
    provenance: str
    source_type: str
    params: dict[str, Any]
    warnings: list[str]
    verifier: dict[str, Any]
    cnn: dict[str, Any]
    d2d: dict[str, Any]
    truth: Optional[dict[str, Any]]
    agreement: dict[str, str]
    calibration: dict[str, Any]
    images: dict[str, Any]
    overlays: dict[str, Any]
    locator: dict[str, Any] = Field(description="display-only site locator; never feeds a verdict")
    explanation: dict[str, list[str]]
    runtime: dict[str, Any] = Field(description="cache state and timings; the only fields that may differ between identical requests")
    arrays: Optional[dict[str, Any]] = None
