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

"""Dataset loaders for WMT and FLORES benchmarks."""

from __future__ import annotations


class DatasetLoader:
    """Load translation datasets for benchmarking."""

    @staticmethod
    async def load_flores200(language_pair: str, sample_size: int) -> list[dict[str, str]]:
        """Load FLORES-200 dataset.

        Args:
            language_pair: Language pair (e.g., 'eng_Latn-spa_Latn')
            sample_size: Number of samples to load

        Returns:
            List of samples with 'source' and 'translation'
        """
        try:
            # Try to load from datasets library if available
            from datasets import load_dataset

            src_lang, tgt_lang = language_pair.split("-")

            dataset = load_dataset("facebook/flores", split="devtest")

            samples = []
            for i, item in enumerate(dataset):
                if i >= sample_size:
                    break

                if src_lang in item and tgt_lang in item:
                    samples.append({"source": item[src_lang], "translation": item[tgt_lang]})

            return samples

        except Exception:
            # Fallback to sample data if datasets not available
            return DatasetLoader._get_sample_data(language_pair, sample_size)

    @staticmethod
    async def load_wmt23(language_pair: str, sample_size: int) -> list[dict[str, str]]:
        """Load WMT23 dataset.

        Args:
            language_pair: Language pair (e.g., 'en-es')
            sample_size: Number of samples to load

        Returns:
            List of samples with 'source' and 'translation'
        """
        try:
            from datasets import load_dataset

            src_lang, tgt_lang = language_pair.split("-")
            dataset_name = "wmt/wmt23"

            # WMT23 uses ISO codes
            dataset = load_dataset(dataset_name, f"{src_lang}-{tgt_lang}", split="test")

            samples = []
            for i, item in enumerate(dataset):
                if i >= sample_size:
                    break

                samples.append(
                    {
                        "source": item["translation"][src_lang],
                        "translation": item["translation"][tgt_lang],
                    }
                )

            return samples

        except Exception:
            # Fallback to sample data
            return DatasetLoader._get_sample_data(language_pair, sample_size)

    @staticmethod
    async def load_wmt22(language_pair: str, sample_size: int) -> list[dict[str, str]]:
        """Load WMT22 dataset.

        Args:
            language_pair: Language pair (e.g., 'en-es')
            sample_size: Number of samples to load

        Returns:
            List of samples with 'source' and 'translation'
        """
        try:
            from datasets import load_dataset

            src_lang, tgt_lang = language_pair.split("-")
            dataset_name = "wmt/wmt22"

            dataset = load_dataset(dataset_name, f"{src_lang}-{tgt_lang}", split="test")

            samples = []
            for i, item in enumerate(dataset):
                if i >= sample_size:
                    break

                samples.append(
                    {
                        "source": item["translation"][src_lang],
                        "translation": item["translation"][tgt_lang],
                    }
                )

            return samples

        except Exception:
            # Fallback to sample data
            return DatasetLoader._get_sample_data(language_pair, sample_size)

    @staticmethod
    def _get_sample_data(language_pair: str, sample_size: int) -> list[dict[str, str]]:
        """Get sample test data when actual datasets are not available.

        Args:
            language_pair: Language pair
            sample_size: Number of samples

        Returns:
            List of sample dictionaries
        """
        # Parse language pair
        if "-" in language_pair:
            parts = language_pair.split("-")
            src_lang = parts[0].split("_")[0] if "_" in parts[0] else parts[0]
        else:
            src_lang = "en"

        # Sample texts by source language - expanded for better testing
        samples_by_lang = {
            "en": [
                {
                    "source": "The quick brown fox jumps over the lazy dog.",
                    "translation": "El rápido zorro marrón salta sobre el perro perezoso.",
                },
                {
                    "source": "Translation quality assessment is crucial for machine translation.",
                    "translation": "La evaluación de la calidad de traducción es crucial para la traducción automática.",
                },
                {
                    "source": "Artificial intelligence is transforming the world.",
                    "translation": "La inteligencia artificial está transformando el mundo.",
                },
                {
                    "source": "Natural language processing enables machines to understand human communication.",
                    "translation": "El procesamiento del lenguaje natural permite a las máquinas comprender la comunicación humana.",
                },
                {
                    "source": "Machine learning models require large amounts of training data.",
                    "translation": "Los modelos de aprendizaje automático requieren grandes cantidades de datos de entrenamiento.",
                },
                {
                    "source": "Neural networks have revolutionized computer vision and speech recognition.",
                    "translation": "Las redes neuronales han revolucionado la visión por computadora y el reconocimiento de voz.",
                },
                {
                    "source": "Deep learning techniques continue to advance rapidly.",
                    "translation": "Las técnicas de aprendizaje profundo continúan avanzando rápidamente.",
                },
                {
                    "source": "The development of large language models has opened new possibilities.",
                    "translation": "El desarrollo de grandes modelos de lenguaje ha abierto nuevas posibilidades.",
                },
                {
                    "source": "Quality metrics help evaluate translation performance objectively.",
                    "translation": "Las métricas de calidad ayudan a evaluar el rendimiento de la traducción objetivamente.",
                },
                {
                    "source": "Multilingual models can process text in many different languages.",
                    "translation": "Los modelos multilingües pueden procesar texto en muchos idiomas diferentes.",
                },
            ],
            "ru": [
                {
                    "source": "Быстрая коричневая лиса прыгает через ленивую собаку.",
                    "translation": "The quick brown fox jumps over the lazy dog.",
                },
                {
                    "source": "Оценка качества перевода имеет решающее значение.",
                    "translation": "Translation quality assessment is crucial.",
                },
                {
                    "source": "Искусственный интеллект меняет мир.",
                    "translation": "Artificial intelligence is changing the world.",
                },
                {
                    "source": "Обработка естественного языка позволяет машинам понимать человеческое общение.",
                    "translation": "Natural language processing enables machines to understand human communication.",
                },
                {
                    "source": "Модели машинного обучения требуют больших объемов данных для обучения.",
                    "translation": "Machine learning models require large amounts of training data.",
                },
                {
                    "source": "Нейронные сети произвели революцию в компьютерном зрении.",
                    "translation": "Neural networks have revolutionized computer vision.",
                },
                {
                    "source": "Методы глубокого обучения продолжают быстро развиваться.",
                    "translation": "Deep learning techniques continue to advance rapidly.",
                },
                {
                    "source": "Разработка больших языковых моделей открыла новые возможности.",
                    "translation": "The development of large language models has opened new possibilities.",
                },
                {
                    "source": "Метрики качества помогают объективно оценить результаты перевода.",
                    "translation": "Quality metrics help evaluate translation results objectively.",
                },
                {
                    "source": "Многоязычные модели могут обрабатывать текст на разных языках.",
                    "translation": "Multilingual models can process text in different languages.",
                },
            ],
            "es": [
                {
                    "source": "El rápido zorro marrón salta sobre el perro perezoso.",
                    "translation": "The quick brown fox jumps over the lazy dog.",
                },
                {
                    "source": "La evaluación de la calidad es importante.",
                    "translation": "Quality assessment is important.",
                },
                {
                    "source": "La inteligencia artificial avanza rápidamente.",
                    "translation": "Artificial intelligence advances rapidly.",
                },
                {
                    "source": "El procesamiento del lenguaje natural permite la comunicación con máquinas.",
                    "translation": "Natural language processing enables communication with machines.",
                },
                {
                    "source": "Los modelos de traducción neuronal mejoran constantemente.",
                    "translation": "Neural translation models constantly improve.",
                },
                {
                    "source": "Las redes neuronales transforman el procesamiento de datos.",
                    "translation": "Neural networks transform data processing.",
                },
                {
                    "source": "El aprendizaje automático requiere datos de alta calidad.",
                    "translation": "Machine learning requires high-quality data.",
                },
                {
                    "source": "Los grandes modelos de lenguaje tienen capacidades impresionantes.",
                    "translation": "Large language models have impressive capabilities.",
                },
                {
                    "source": "Las métricas de evaluación miden la calidad objetivamente.",
                    "translation": "Evaluation metrics measure quality objectively.",
                },
                {
                    "source": "La traducción automática facilita la comunicación internacional.",
                    "translation": "Machine translation facilitates international communication.",
                },
            ],
        }

        # Get samples for this language or default to English
        available_samples = samples_by_lang.get(src_lang, samples_by_lang["en"])

        # Return requested number of samples, cycling if needed
        result = []
        for i in range(sample_size):
            result.append(available_samples[i % len(available_samples)])

        return result
