"""findex — a search engine built one lab at a time."""

__version__ = "1.0.0"

from findex.corpus import Document, iter_documents  # noqa: E402
from findex.tokenize import tokenize  # noqa: E402

__all__ = ["Document", "__version__", "iter_documents", "tokenize"]
