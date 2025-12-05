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

"""Lightweight RAG (Retrieval-Augmented Generation) for KTTC.

Uses BM25 for efficient, CPU-based document retrieval without heavy embedding models.
Designed to work on laptops without GPU.

Key features:
- BM25 ranking (Okapi BM25) for document retrieval
- In-memory index for fast querying
- No external dependencies required
- Works with glossaries, translation memories, and examples
"""

from .bm25_retriever import BM25Retriever, Document, RetrievalResult
from .context_builder import ContextBuilder

__all__ = [
    "BM25Retriever",
    "ContextBuilder",
    "Document",
    "RetrievalResult",
]
