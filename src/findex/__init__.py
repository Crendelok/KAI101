"""findex — a search engine built one lab at a time."""

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize

__all__ = ["Document", "iter_documents", "tokenize"]
