from typing import List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfClauseRepresentation:
    """
    TF-IDF representation for legal clauses.

    This is our lexical retrieval baseline.

    The same vectorizer must be used for both:
        1. indexing clauses
        2. transforming query clauses
    """

    def __init__(
        self,
        max_features: int = 20000,
        ngram_range=(1, 2),
        min_df: int = 1,
        sublinear_tf: bool = True,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            lowercase=True,
            strip_accents="unicode",
        )

        self.matrix = None
        self.fitted = False

    def fit(
        self,
        texts: Sequence[str],
    ):
        """
        Learn the TF-IDF vocabulary from clause texts.
        """

        if not texts:
            raise ValueError(
                "Cannot fit TF-IDF on an empty list."
            )

        cleaned_texts = [
            str(text) if text else ""
            for text in texts
        ]

        self.matrix = self.vectorizer.fit_transform(
            cleaned_texts
        )

        self.fitted = True

        return self

    def transform(
        self,
        texts: Sequence[str],
    ):
        """
        Transform new texts using the fitted TF-IDF vocabulary.
        """

        if not self.fitted:
            raise RuntimeError(
                "TF-IDF representation has not been fitted yet."
            )

        cleaned_texts = [
            str(text) if text else ""
            for text in texts
        ]

        return self.vectorizer.transform(
            cleaned_texts
        )

    def fit_transform(
        self,
        texts: Sequence[str],
    ):
        """
        Fit the vectorizer and transform the texts.
        """

        if not texts:
            raise ValueError(
                "Cannot fit TF-IDF on an empty list."
            )

        cleaned_texts = [
            str(text) if text else ""
            for text in texts
        ]

        self.matrix = self.vectorizer.fit_transform(
            cleaned_texts
        )

        self.fitted = True

        return self.matrix

    def get_feature_count(self) -> int:
        """
        Return number of learned TF-IDF features.
        """

        if not self.fitted:
            return 0

        return len(
            self.vectorizer.get_feature_names_out()
        )