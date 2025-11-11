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

Based on WMT 2025 Metrics Shared Task findings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from comet import Comet

logger = logging.getLogger(__name__)


class NeuralMetricsResult(BaseModel):
    """Result from neural metrics evaluation.

    Contains scores from both reference-based (COMET) and
    reference-free (CometKiwi) metrics.
    """

    comet_score: float | None = Field(
        default=None, description="COMET score (reference-based), 0-1 range", ge=0.0, le=1.0
    )
    kiwi_score: float | None = Field(
        default=None, description="CometKiwi score (reference-free), 0-1 range", ge=0.0, le=1.0
    )
    quality_estimate: str | None = Field(
        default=None,
        description="Quality classification based on scores: high, medium, or low",
        pattern=r"^(high|medium|low)$",
    )

    def get_composite_score(self, comet_weight: float = 0.6, kiwi_weight: float = 0.4) -> float:
        """Calculate weighted composite score from available metrics.

        Args:
            comet_weight: Weight for COMET score (default: 0.6)
            kiwi_weight: Weight for CometKiwi score (default: 0.4)

        Returns:
            Composite score in 0-1 range

        Raises:
            ValueError: If no scores are available
        """
        if self.comet_score is None and self.kiwi_score is None:
            raise ValueError("No neural metric scores available for composite calculation")

        score = 0.0
        total_weight = 0.0

        if self.comet_score is not None:
            score += comet_weight * self.comet_score
            total_weight += comet_weight

        if self.kiwi_score is not None:
            score += kiwi_weight * self.kiwi_score
            total_weight += kiwi_weight

        return score / total_weight if total_weight > 0 else 0.0


class NeuralMetrics:
    """Neural quality metrics integration.

    Provides COMET and CometKiwi metrics for translation quality evaluation.
    Models are downloaded automatically on first use.

    Example:
        >>> metrics = NeuralMetrics()
        >>> await metrics.initialize()
        >>> result = await metrics.evaluate(
        ...     source="Hello, world!",
        ...     translation="Hola, mundo!",
        ...     reference="¡Hola, mundo!"
        ... )
        >>> print(f"COMET: {result.comet_score:.3f}")
    """

    def __init__(self, use_gpu: bool = False):
        """Initialize neural metrics.

        Args:
            use_gpu: Whether to use GPU for inference (default: False for CPU)
        """
        self.use_gpu = use_gpu
        self.comet_model: Comet | None = None
        self.kiwi_model: Comet | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize and download neural metric models.

        Downloads pre-trained COMET and CometKiwi models from Unbabel.
        This may take time on first run but models are cached locally.

        Raises:
            RuntimeError: If model initialization fails
        """
        if self._initialized:
            return

        try:
            from comet import download_model, load_from_checkpoint

            logger.info("Downloading COMET models (first time may take several minutes)...")

            # Download COMET model (reference-based)
            comet_model_path = download_model("Unbabel/wmt22-comet-da")
            self.comet_model = load_from_checkpoint(comet_model_path)

            # Download CometKiwi model (reference-free)
            kiwi_model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
            self.kiwi_model = load_from_checkpoint(kiwi_model_path)

            self._initialized = True
            logger.info("Neural metrics models loaded successfully")

        except ImportError as e:
            raise RuntimeError(
                "Failed to import COMET library. " "Install with: pip install unbabel-comet"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize neural metrics: {e}") from e

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
                quality_estimate=quality_estimate,
            )

        except Exception as e:
            logger.error(f"COMET evaluation failed: {e}")
            return NeuralMetricsResult(comet_score=None, kiwi_score=None, quality_estimate="low")

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
                quality_estimate=quality_estimate,
            )

        except Exception as e:
            logger.error(f"CometKiwi evaluation failed: {e}")
            return NeuralMetricsResult(comet_score=None, kiwi_score=None, quality_estimate="low")

    async def evaluate(
        self, source: str, translation: str, reference: str | None = None
    ) -> NeuralMetricsResult:
        """Evaluate translation with both COMET and CometKiwi.

        If reference is provided, runs both COMET (with reference) and
        CometKiwi (reference-free). Otherwise only runs CometKiwi.

        Args:
            source: Source text
            translation: Translation to evaluate
            reference: Optional reference translation

        Returns:
            NeuralMetricsResult with available scores

        Raises:
            RuntimeError: If models not initialized
        """
        if not self._initialized:
            raise RuntimeError("Neural metrics not initialized. Call initialize() first.")

        # Always run reference-free evaluation
        kiwi_result = await self.evaluate_reference_free(source, translation)

        # If reference available, also run COMET
        if reference is not None:
            comet_result = await self.evaluate_with_reference(source, translation, reference)

            # Combine results
            composite_score = NeuralMetricsResult(
                comet_score=comet_result.comet_score,
                kiwi_score=kiwi_result.kiwi_score,
                quality_estimate=comet_result.quality_estimate,  # Prefer COMET classification
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
        self._initialized = False
        logger.info("Neural metrics models unloaded")
