# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural quality metrics integration.

Provides integration with neural translation quality metrics:
- COMET: Reference-based metric using XLM-RoBERTa
- CometKiwi: Reference-free quality estimation
- XCOMET: State-of-the-art metric with error span detection (WMT 2024)

Based on WMT 2025 Metrics Shared Task findings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from comet import Comet

logger = logging.getLogger(__name__)


class ErrorSpan(BaseModel):
    """Error span detected by XCOMET.

    Represents a specific error location in the translation with
    its severity and confidence level.
    """

    text: str = Field(description="The problematic text phrase")
    start: int = Field(description="Start character position", ge=0)
    end: int = Field(description="End character position", ge=0)
    severity: str = Field(description="Error severity: minor, major, or critical")
    confidence: float = Field(description="Detection confidence (0-1)", ge=0.0, le=1.0)


class NeuralMetricsResult(BaseModel):
    """Result from neural metrics evaluation.

    Contains scores from COMET, CometKiwi, and optionally XCOMET
    with error span detection.
    """

    comet_score: float | None = Field(
        default=None, description="COMET score (reference-based), 0-1 range", ge=0.0, le=1.0
    )
    kiwi_score: float | None = Field(
        default=None, description="CometKiwi score (reference-free), 0-1 range", ge=0.0, le=1.0
    )
    xcomet_score: float | None = Field(
        default=None,
        description="XCOMET score with error detection (reference-based), 0-1 range",
        ge=0.0,
        le=1.0,
    )
    quality_estimate: str | None = Field(
        default=None,
        description="Quality classification based on scores: high, medium, or low",
        pattern=r"^(high|medium|low)$",
    )
    error_spans: list[ErrorSpan] = Field(
        default_factory=list, description="Error spans detected by XCOMET (if available)"
    )

    def get_composite_score(
        self,
        comet_weight: float = 0.3,
        kiwi_weight: float = 0.2,
        xcomet_weight: float = 0.5,
    ) -> float:
        """Calculate weighted composite score from available metrics.

        Args:
            comet_weight: Weight for COMET score (default: 0.3)
            kiwi_weight: Weight for CometKiwi score (default: 0.2)
            xcomet_weight: Weight for XCOMET score (default: 0.5, highest as it's SOTA)

        Returns:
            Composite score in 0-1 range

        Raises:
            ValueError: If no scores are available
        """
        if self.comet_score is None and self.kiwi_score is None and self.xcomet_score is None:
            raise ValueError("No neural metric scores available for composite calculation")

        score = 0.0
        total_weight = 0.0

        # XCOMET gets highest weight as it's state-of-the-art (WMT 2024)
        if self.xcomet_score is not None:
            score += xcomet_weight * self.xcomet_score
            total_weight += xcomet_weight

        if self.comet_score is not None:
            score += comet_weight * self.comet_score
            total_weight += comet_weight

        if self.kiwi_score is not None:
            score += kiwi_weight * self.kiwi_score
            total_weight += kiwi_weight

        return score / total_weight if total_weight > 0 else 0.0


class NeuralMetrics:
    """Neural quality metrics integration.

    Provides COMET, CometKiwi, and XCOMET metrics for translation quality evaluation.
    Models are downloaded automatically on first use.

    Example:
        >>> metrics = NeuralMetrics(use_xcomet=True)
        >>> await metrics.initialize()
        >>> result = await metrics.evaluate(
        ...     source="Hello, world!",
        ...     translation="Hola, mundo!",
        ...     reference="¡Hola, mundo!"
        ... )
        >>> print(f"XCOMET: {result.xcomet_score:.3f}")
        >>> for span in result.error_spans:
        ...     print(f"Error at [{span.start}:{span.end}]: {span.text} ({span.severity})")
    """

    def __init__(self, use_gpu: bool = False, use_xcomet: bool = True):
        """Initialize neural metrics.

        Args:
            use_gpu: Whether to use GPU for inference (default: False for CPU)
            use_xcomet: Whether to use XCOMET for error span detection (default: True)
        """
        self.use_gpu = use_gpu
        self.use_xcomet = use_xcomet
        self.comet_model: Comet | None = None
        self.kiwi_model: Comet | None = None
        self.xcomet_model: Comet | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize and download neural metric models.

        Downloads pre-trained COMET, CometKiwi, and optionally XCOMET models from Unbabel.
        This may take time on first run but models are cached locally.

        Raises:
            RuntimeError: If model initialization fails
        """
        if self._initialized:
            return

        try:
            from comet import download_model, load_from_checkpoint

            logger.info("Downloading neural metric models (first time may take several minutes)...")

            # Download COMET model (reference-based)
            comet_model_path = download_model("Unbabel/wmt22-comet-da")
            self.comet_model = load_from_checkpoint(comet_model_path)

            # Download CometKiwi model (reference-free)
            kiwi_model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
            self.kiwi_model = load_from_checkpoint(kiwi_model_path)

            # Download XCOMET model (error span detection) if requested
            if self.use_xcomet:
                logger.info("Downloading XCOMET-XL model (3.5B params, may take time)...")
                xcomet_model_path = download_model("Unbabel/XCOMET-XL")
                self.xcomet_model = load_from_checkpoint(xcomet_model_path)
                logger.info("XCOMET-XL model loaded successfully")

            self._initialized = True
            logger.info("Neural metrics models loaded successfully")

        except ImportError as e:
            raise RuntimeError(
                "Failed to import COMET library. Install with: pip install unbabel-comet>=2.2.0"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize neural metrics: {e}") from e

    async def evaluate_with_xcomet(
        self, source: str, translation: str, reference: str
    ) -> NeuralMetricsResult:
        """Evaluate translation with XCOMET (error span detection).

        Args:
            source: Source text
            translation: Translation to evaluate
            reference: Reference (gold standard) translation

        Returns:
            NeuralMetricsResult with XCOMET score and error spans

        Raises:
            RuntimeError: If models not initialized or XCOMET not enabled
        """
        if not self._initialized or self.xcomet_model is None:
            raise RuntimeError(
                "XCOMET not initialized. Set use_xcomet=True and call initialize() first."
            )

        try:
            # Prepare data for XCOMET
            data = [{"src": source, "mt": translation, "ref": reference}]

            # Run inference
            gpus = 1 if self.use_gpu else 0
            results = self.xcomet_model.predict(data, batch_size=1, gpus=gpus)

            xcomet_score = float(results.scores[0])

            # Extract error spans from metadata
            error_spans: list[ErrorSpan] = []
            if hasattr(results, "metadata") and hasattr(results.metadata, "error_spans"):
                raw_spans = results.metadata.error_spans
                if raw_spans and len(raw_spans) > 0:
                    for span_dict in raw_spans[0]:  # First sample
                        error_spans.append(
                            ErrorSpan(
                                text=span_dict.get("text", ""),
                                start=span_dict.get("start", 0),
                                end=span_dict.get("end", 0),
                                severity=span_dict.get("severity", "minor"),
                                confidence=span_dict.get("confidence", 0.0),
                            )
                        )

            # Classify quality based on XCOMET score
            quality_estimate = self._classify_quality(xcomet_score)

            return NeuralMetricsResult(
                comet_score=None,
                kiwi_score=None,
                xcomet_score=xcomet_score,
                quality_estimate=quality_estimate,
                error_spans=error_spans,
            )

        except Exception as e:
            logger.error(f"XCOMET evaluation failed: {e}")
            return NeuralMetricsResult(
                comet_score=None,
                kiwi_score=None,
                xcomet_score=None,
                quality_estimate="low",
                error_spans=[],
            )

    async def evaluate_with_reference(
        self, source: str, translation: str, reference: str
    ) -> NeuralMetricsResult:
        """Evaluate translation quality with reference using COMET.

        Args:
            source: Source text
            translation: Translation to evaluate
            reference: Reference (gold standard) translation

        Returns:
            NeuralMetricsResult with COMET score

        Raises:
            RuntimeError: If models not initialized
        """
        if not self._initialized or self.comet_model is None:
            raise RuntimeError("Neural metrics not initialized. Call initialize() first.")

        try:
            # Prepare data for COMET
            data = [{"src": source, "mt": translation, "ref": reference}]

            # Run inference
            gpus = 1 if self.use_gpu else 0
            results = self.comet_model.predict(data, batch_size=1, gpus=gpus)

            comet_score = float(results.scores[0])

            # Classify quality based on COMET score
            quality_estimate = self._classify_quality(comet_score)

            return NeuralMetricsResult(
                comet_score=comet_score,
                kiwi_score=None,
                xcomet_score=None,
                quality_estimate=quality_estimate,
                error_spans=[],
            )

        except Exception as e:
            logger.error(f"COMET evaluation failed: {e}")
            return NeuralMetricsResult(
                comet_score=None,
                kiwi_score=None,
                xcomet_score=None,
                quality_estimate="low",
                error_spans=[],
            )

    async def evaluate_reference_free(self, source: str, translation: str) -> NeuralMetricsResult:
        """Evaluate translation quality without reference using CometKiwi.

        Args:
            source: Source text
            translation: Translation to evaluate

        Returns:
            NeuralMetricsResult with CometKiwi score

        Raises:
            RuntimeError: If models not initialized
        """
        if not self._initialized or self.kiwi_model is None:
            raise RuntimeError("Neural metrics not initialized. Call initialize() first.")

        try:
            # Prepare data for CometKiwi (no reference needed)
            data = [{"src": source, "mt": translation}]

            # Run inference
            gpus = 1 if self.use_gpu else 0
            results = self.kiwi_model.predict(data, batch_size=1, gpus=gpus)

            kiwi_score = float(results.scores[0])

            # Classify quality based on CometKiwi score
            quality_estimate = self._classify_quality(kiwi_score)

            return NeuralMetricsResult(
                comet_score=None,
                kiwi_score=kiwi_score,
                xcomet_score=None,
                quality_estimate=quality_estimate,
                error_spans=[],
            )

        except Exception as e:
            logger.error(f"CometKiwi evaluation failed: {e}")
            return NeuralMetricsResult(
                comet_score=None,
                kiwi_score=None,
                xcomet_score=None,
                quality_estimate="low",
                error_spans=[],
            )

    async def evaluate(
        self, source: str, translation: str, reference: str | None = None
    ) -> NeuralMetricsResult:
        """Evaluate translation with all available metrics.

        If reference is provided, runs COMET, XCOMET (if enabled), and CometKiwi.
        Otherwise only runs CometKiwi (reference-free).

        Args:
            source: Source text
            translation: Translation to evaluate
            reference: Optional reference translation

        Returns:
            NeuralMetricsResult with available scores and error spans

        Raises:
            RuntimeError: If models not initialized
        """
        if not self._initialized:
            raise RuntimeError("Neural metrics not initialized. Call initialize() first.")

        # Always run reference-free evaluation
        kiwi_result = await self.evaluate_reference_free(source, translation)

        # If reference available, also run reference-based metrics
        if reference is not None:
            comet_result = await self.evaluate_with_reference(source, translation, reference)

            # Run XCOMET if enabled (state-of-the-art with error spans)
            xcomet_result = None
            if self.use_xcomet and self.xcomet_model is not None:
                xcomet_result = await self.evaluate_with_xcomet(source, translation, reference)

            # Combine results - prefer XCOMET quality estimate if available
            quality_estimate = (
                xcomet_result.quality_estimate
                if xcomet_result and xcomet_result.quality_estimate
                else comet_result.quality_estimate
            )

            composite_score = NeuralMetricsResult(
                comet_score=comet_result.comet_score,
                kiwi_score=kiwi_result.kiwi_score,
                xcomet_score=xcomet_result.xcomet_score if xcomet_result else None,
                quality_estimate=quality_estimate,
                error_spans=xcomet_result.error_spans if xcomet_result else [],
            )

            return composite_score

        return kiwi_result

    def _classify_quality(self, score: float) -> str:
        """Classify quality level based on neural metric score.

        Args:
            score: Neural metric score (0-1 range)

        Returns:
            Quality classification: 'high', 'medium', or 'low'
        """
        if score >= 0.80:
            return "high"
        elif score >= 0.60:
            return "medium"
        else:
            return "low"

    async def cleanup(self) -> None:
        """Cleanup models and free memory."""
        self.comet_model = None
        self.kiwi_model = None
        self.xcomet_model = None
        self._initialized = False
        logger.info("Neural metrics models unloaded")
