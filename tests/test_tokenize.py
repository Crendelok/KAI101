import inspect

from findex.tokenize import tokenize


def toks(text: str) -> list[str]:
    return list(tokenize(text))


def test_is_a_generator():
    assert inspect.isgeneratorfunction(tokenize)
    assert next(tokenize("Hello world")) == "hello"


def test_empty_and_whitespace():
    assert toks("") == []
    assert toks("   \n\t ...!?") == []


def test_mixed_case_and_casefold():
    assert toks("Hello WORLD hElLo") == ["hello", "world", "hello"]
    assert toks("Straße") == toks("STRASSE") == ["strasse"]  # lower() would miss this


def test_punctuation_splits():
    assert toks("Hi, there! (really?) end.") == ["hi", "there", "really", "end"]
    assert toks("3.14 and snake_case") == ["3", "14", "and", "snake", "case"]


def test_cyrillic():
    assert toks("Привіт, Світе! Київ") == ["привіт", "світе", "київ"]


def test_combining_mark_matches_precomposed():
    precomposed = "café"
    decomposed = "café"
    assert precomposed != decomposed  # look identical, aren't
    assert toks(precomposed) == toks(decomposed) == ["café"]


def test_combining_mark_without_precomposed_form_stays_in_word():
    # 'q' + combining acute has no NFC precomposed form; must not split the word.
    assert toks("aq́b") == ["aq́b"]


def test_apostrophes_kept_inside_words_and_unified():
    assert toks("don't Don’t") == ["don't", "don't"]
    assert toks("п'ять пʼять п’ять") == ["п'ять"] * 3
    assert toks("'quoted'") == ["quoted"]


def test_hyphens_split():
    assert toks("state-of-the-art") == ["state", "of", "the", "art"]


def test_digits_kept():
    assert toks("Python3 came out in 2008") == ["python3", "came", "out", "in", "2008"]
