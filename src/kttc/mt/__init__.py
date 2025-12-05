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

"""Machine Translation integration layer for KTTC.

Provides interfaces and implementations for MT engines like DeepL.
Used for generating reference translations during QA evaluation.
"""

from .base import BaseMTProvider, MTError, MTQuotaExceededError, TranslationResult
from .deepl_provider import DeepLProvider

__all__ = [
    "BaseMTProvider",
    "DeepLProvider",
    "MTError",
    "MTQuotaExceededError",
    "TranslationResult",
]
