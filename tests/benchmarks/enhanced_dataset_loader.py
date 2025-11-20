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

"""Enhanced dataset loaders for FLORES-200, WMT-MQM and other benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EnhancedDatasetLoader:
    """Enhanced loader for translation benchmark datasets."""

    # Language code mapping (FLORES-200 format to ISO)
    LANG_CODE_MAP = {
        "eng_Latn": "en",
        "rus_Cyrl": "ru",
        "zho_Hans": "zh",
        "spa_Latn": "es",
        "fra_Latn": "fr",
        "deu_Latn": "de",
    }

    # Reverse mapping
    ISO_TO_FLORES = {v: k for k, v in LANG_CODE_MAP.items()}

    def __init__(self, data_dir: str | Path = "tests/benchmarks/data"):
        """Initialize the enhanced dataset loader.

        Args:
            data_dir: Directory to store/load benchmark data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    async def load_flores200(
        self, src_lang: str, tgt_lang: str, split: str = "devtest", sample_size: int | None = None
    ) -> list[dict[str, Any]]:
        """Load FLORES-200 dataset from HuggingFace.

        Args:
            src_lang: Source language ISO code (en, ru, zh)
            tgt_lang: Target language ISO code (en, ru, zh)
            split: Dataset split (dev, devtest, test)
            sample_size: Number of samples to load (None for all)

        Returns:
            List of samples with 'source', 'translation', 'id', 'domain'
        """
        try:
            from datasets import load_dataset

            # Convert ISO codes to FLORES-200 format
            src_flores = self.ISO_TO_FLORES.get(src_lang, f"{src_lang}_Latn")
            tgt_flores = self.ISO_TO_FLORES.get(tgt_lang, f"{tgt_lang}_Latn")

            # Load dataset
            dataset = load_dataset("facebook/flores", name="all", split=split)

            samples: list[dict[str, Any]] = []
            for idx, item in enumerate(dataset):
                if sample_size and len(samples) >= sample_size:
                    break

                # FLORES-200 has columns for each language
                if f"sentence_{src_flores}" in item and f"sentence_{tgt_flores}" in item:
                    samples.append(
                        {
                            "id": f"flores200_{split}_{idx}",
                            "source": item[f"sentence_{src_flores}"],
                            "translation": item[f"sentence_{tgt_flores}"],
                            "source_lang": src_lang,
                            "target_lang": tgt_lang,
                            "domain": "general",
                            "dataset": "flores200",
                        }
                    )

            return samples

        except Exception as e:
            print(f"Warning: Failed to load FLORES-200 from HuggingFace: {e}")
            print("Falling back to local cached data if available...")
            return await self._load_cached_flores200(src_lang, tgt_lang, split, sample_size)

    async def load_wmt_mqm(
        self, src_lang: str, tgt_lang: str, sample_size: int | None = None
    ) -> list[dict[str, Any]]:
        """Load WMT-MQM dataset with error annotations.

        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            sample_size: Number of samples to load

        Returns:
            List of samples with error annotations
        """
        try:
            # Try to load from google/wmt-mqm-human-evaluation if available
            import datasets

            # WMT-MQM currently available for en-de and zh-en
            lang_pair = f"{src_lang}-{tgt_lang}"

            if lang_pair not in ["en-de", "zh-en"]:
                print(f"WMT-MQM not available for {lang_pair}, using simulated data")
                return await self._generate_simulated_mqm(src_lang, tgt_lang, sample_size or 100)

            # Load from HuggingFace (if the dataset is available)
            dataset = datasets.load_dataset("google/wmt-mqm-human-evaluation", split="train")

            samples: list[dict[str, Any]] = []
            for idx, item in enumerate(dataset):
                if sample_size and len(samples) >= sample_size:
                    break

                # Parse MQM annotations
                errors = self._parse_mqm_errors(item.get("error_annotations", []))

                samples.append(
                    {
                        "id": f"wmt_mqm_{lang_pair}_{idx}",
                        "source": item.get("source", ""),
                        "translation": item.get("translation", ""),
                        "source_lang": src_lang,
                        "target_lang": tgt_lang,
                        "errors": errors,
                        "mqm_score": item.get("mqm_score"),
                        "dataset": "wmt_mqm",
                    }
                )

            return samples

        except Exception as e:
            print(f"Warning: Failed to load WMT-MQM: {e}")
            return await self._generate_simulated_mqm(src_lang, tgt_lang, sample_size or 100)

    async def _load_cached_flores200(
        self, src_lang: str, tgt_lang: str, split: str, sample_size: int | None
    ) -> list[dict[str, Any]]:
        """Load FLORES-200 from local cache.

        Priority order:
        1. quality_{src_lang}_{tgt_lang}.json (100+ LLM-generated samples)
        2. flores200_{src_lang}_{tgt_lang}_{split}.json (cached FLORES-200)
        3. Fallback samples (minimal data)

        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            split: Dataset split
            sample_size: Number of samples

        Returns:
            List of cached samples
        """
        # Priority 1: Check for quality LLM-generated data (100+ samples)
        quality_file = self.data_dir / f"quality_{src_lang}_{tgt_lang}.json"
        if quality_file.exists():
            print(f"✅ Using quality LLM data: {quality_file.name}")
            data: list[dict[str, Any]] = json.loads(quality_file.read_text(encoding="utf-8"))
            return data[:sample_size] if sample_size else data

        # Priority 2: Check for FLORES-200 cached data
        cache_file = self.data_dir / f"flores200_{src_lang}_{tgt_lang}_{split}.json"
        if cache_file.exists():
            print(f"Using FLORES-200 cached data: {cache_file.name}")
            data_cached: list[dict[str, Any]] = json.loads(cache_file.read_text(encoding="utf-8"))
            return data_cached[:sample_size] if sample_size else data_cached

        # Priority 3: Return fallback sample data
        print("⚠️  No cached data found, using minimal fallback samples")
        return self._get_fallback_samples(src_lang, tgt_lang, sample_size or 10)

    def _parse_mqm_errors(self, error_annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Parse MQM error annotations.

        Args:
            error_annotations: Raw error annotations

        Returns:
            Parsed errors in standardized format
        """
        errors = []
        for ann in error_annotations:
            errors.append(
                {
                    "category": ann.get("category", "unknown"),
                    "subcategory": ann.get("subcategory", "unknown"),
                    "severity": ann.get("severity", "minor"),
                    "location": [ann.get("start", 0), ann.get("end", 0)],
                    "description": ann.get("description", ""),
                }
            )
        return errors

    async def _generate_simulated_mqm(
        self, src_lang: str, tgt_lang: str, sample_size: int
    ) -> list[dict[str, Any]]:
        """Generate simulated MQM data with errors for testing.

        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            sample_size: Number of samples to generate

        Returns:
            List of simulated samples with errors
        """
        # This will be implemented by the bad translation generator
        return []

    def _get_fallback_samples(
        self, src_lang: str, tgt_lang: str, sample_size: int
    ) -> list[dict[str, Any]]:
        """Get fallback sample data when datasets are not available.

        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            sample_size: Number of samples

        Returns:
            List of fallback samples
        """
        # Comprehensive fallback data for all language pairs
        fallback_data = {
            "en-ru": [
                {
                    "source": "Artificial intelligence is transforming the world.",
                    "translation": "Искусственный интеллект преобразует мир.",
                },
                {
                    "source": "Machine translation quality has improved significantly.",
                    "translation": "Качество машинного перевода значительно улучшилось.",
                },
                {
                    "source": "Natural language processing enables human-computer interaction.",
                    "translation": "Обработка естественного языка обеспечивает взаимодействие человека с компьютером.",
                },
            ],
            "en-zh": [
                {
                    "source": "Artificial intelligence is transforming the world.",
                    "translation": "人工智能正在改变世界。",
                },
                {
                    "source": "Machine translation quality has improved significantly.",
                    "translation": "机器翻译质量已经显著提高。",
                },
                {
                    "source": "Natural language processing enables human-computer interaction.",
                    "translation": "自然语言处理实现人机交互。",
                },
            ],
            "ru-en": [
                {
                    "source": "Искусственный интеллект преобразует мир.",
                    "translation": "Artificial intelligence is transforming the world.",
                },
                {
                    "source": "Качество машинного перевода значительно улучшилось.",
                    "translation": "Machine translation quality has improved significantly.",
                },
                {
                    "source": "Обработка естественного языка обеспечивает взаимодействие человека с компьютером.",
                    "translation": "Natural language processing enables human-computer interaction.",
                },
            ],
            "zh-en": [
                {
                    "source": "人工智能正在改变世界。",
                    "translation": "Artificial intelligence is transforming the world.",
                },
                {
                    "source": "机器翻译质量已经显著提高。",
                    "translation": "Machine translation quality has improved significantly.",
                },
                {
                    "source": "自然语言处理实现人机交互。",
                    "translation": "Natural language processing enables human-computer interaction.",
                },
            ],
            "ru-zh": [
                {
                    "source": "Искусственный интеллект преобразует мир.",
                    "translation": "人工智能正在改变世界。",
                },
                {
                    "source": "Качество машинного перевода значительно улучшилось.",
                    "translation": "机器翻译质量已经显著提高。",
                },
            ],
            "zh-ru": [
                {
                    "source": "人工智能正在改变世界。",
                    "translation": "Искусственный интеллект преобразует мир.",
                },
                {
                    "source": "机器翻译质量已经显著提高。",
                    "translation": "Качество машинного перевода значительно улучшилось.",
                },
            ],
        }

        lang_pair = f"{src_lang}-{tgt_lang}"
        base_samples = fallback_data.get(lang_pair, fallback_data["en-ru"])

        # Create samples
        samples = []
        for i in range(sample_size):
            base = base_samples[i % len(base_samples)]
            samples.append(
                {
                    "id": f"fallback_{lang_pair}_{i}",
                    "source": base["source"],
                    "translation": base["translation"],
                    "source_lang": src_lang,
                    "target_lang": tgt_lang,
                    "domain": "general",
                    "dataset": "fallback",
                }
            )

        return samples

    async def save_to_cache(self, samples: list[dict[str, Any]], filename: str) -> None:
        """Save samples to local cache.

        Args:
            samples: List of samples to save
            filename: Cache filename
        """
        cache_file = self.data_dir / filename
        cache_file.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(samples)} samples to {cache_file}")
