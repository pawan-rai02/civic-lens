"""
NLP Utilities for Civic Lens Project

Shared text processing functions for NYC and Bangalore complaint analysis.
Ensures consistent feature extraction across both cities.
"""

import re
from typing import List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation


def clean_text(text: str) -> str:
    """
    Clean and normalize text for NLP processing.
    
    Converts to lowercase, removes punctuation and special characters,
    collapses whitespace. Used for NYC descriptor/resolution_description
    and Bangalore Staff Remarks.
    
    Parameters
    ----------
    text : str
        Raw text to clean
        
    Returns
    -------
    str
        Cleaned text with only lowercase letters, numbers, and single spaces
        
    Examples
    --------
    >>> clean_text("Water MAIN Break! [Emergency]")
    'water main break emergency'
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and punctuation, keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Collapse multiple spaces into single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def urgency_score(text: str, keywords: List[str]) -> int:
    """
    Calculate urgency score based on keyword matches.
    
    Counts how many urgency-related keywords appear in the text.
    Case-insensitive matching. Pass different keyword lists per city
    as needed (e.g., NYC may include "flooding", Bangalore different terms).
    
    Parameters
    ----------
    text : str
        Text to score (should be pre-cleaned with clean_text)
    keywords : List[str]
        List of urgency keywords to match (e.g., ['emergency', 'urgent', 'dangerous'])
        
    Returns
    -------
    int
        Count of unique urgency keywords found in text
        
    Examples
    --------
    >>> urgency_score("emergency water leak urgent", ["emergency", "urgent", "leak"])
    3
    >>> urgency_score("routine maintenance", ["emergency", "urgent"])
    0
    """
    if pd.isna(text) or not isinstance(text, str):
        return 0
    
    text_lower = text.lower()
    
    # Count unique keyword matches (word boundary matching to avoid partial matches)
    matches = sum(1 for keyword in keywords if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
    
    return matches


def fit_tfidf_svd(
    corpus: List[str],
    max_features: int = 5000,
    n_components: int = 50,
    min_df: int = 2,
    max_df: float = 0.95,
    random_state: int = 42
) -> Tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]:
    """
    Fit TF-IDF vectorizer and reduce dimensionality with TruncatedSVD.
    
    Creates dense feature representations of text data. Returns fitted
    transformers so they can be saved and used to score new complaints later.
    
    Parameters
    ----------
    corpus : List[str]
        List of cleaned text documents
    max_features : int, default=5000
        Maximum number of TF-IDF features
    n_components : int, default=50
        Number of SVD components (dense feature dimensions)
    min_df : int, default=2
        Ignore terms appearing in fewer than min_df documents
    max_df : float, default=0.95
        Ignore terms appearing in more than max_df fraction of documents
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    features : np.ndarray
        Dense feature matrix of shape (n_documents, n_components)
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer (can be pickled for later use)
    svd : TruncatedSVD
        Fitted SVD transformer (can be pickled for later use)
        
    Examples
    --------
    >>> corpus = ["water leak emergency", "noise complaint routine", "broken streetlight"]
    >>> features, vectorizer, svd = fit_tfidf_svd(corpus, n_components=2)
    >>> features.shape
    (3, 2)
    """
    # Filter out empty strings
    corpus_clean = [doc if doc else " " for doc in corpus]
    
    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),  # Use unigrams and bigrams
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus_clean)
    
    # Reduce dimensionality with TruncatedSVD
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state
    )
    
    features = svd.fit_transform(tfidf_matrix)
    
    return features, vectorizer, svd


def is_boilerplate_remark(text: str, boilerplate_set: Set[str]) -> bool:
    """
    Check if text is an exact-match boilerplate remark.
    
    Bangalore-specific function to flag generic remarks like "attended", "closed"
    that don't provide meaningful information. Uses exact matching after cleaning.
    
    Parameters
    ----------
    text : str
        Staff remark text (should be pre-cleaned with clean_text)
    boilerplate_set : Set[str]
        Set of exact boilerplate phrases to match
        e.g., {"attended", "closed", "completed", "no action required"}
        
    Returns
    -------
    bool
        True if text exactly matches a boilerplate phrase, False otherwise
        
    Examples
    --------
    >>> boilerplate = {"attended", "closed", "completed"}
    >>> is_boilerplate_remark("attended", boilerplate)
    True
    >>> is_boilerplate_remark("attended to water leak on main street", boilerplate)
    False
    """
    if pd.isna(text) or not isinstance(text, str):
        return True  # Treat null/empty as boilerplate
    
    text_clean = text.strip().lower()
    
    return text_clean in boilerplate_set


def fit_lda(
    corpus: List[str],
    n_topics: int = 12,
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.95,
    max_iter: int = 20,
    random_state: int = 42
) -> Tuple[np.ndarray, LatentDirichletAllocation, TfidfVectorizer]:
    """
    Fit Latent Dirichlet Allocation (LDA) topic model.
    
    Generic topic modeling function. Primarily used for NYC (which has rich
    descriptor/resolution text), but written generically for reuse if Bangalore
    remark diversity is higher than expected.
    
    Parameters
    ----------
    corpus : List[str]
        List of cleaned text documents
    n_topics : int, default=12
        Number of latent topics to extract
    max_features : int, default=5000
        Maximum number of features for vectorization
    min_df : int, default=2
        Ignore terms appearing in fewer than min_df documents
    max_df : float, default=0.95
        Ignore terms appearing in more than max_df fraction of documents
    max_iter : int, default=20
        Maximum iterations for LDA fitting
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    topic_distributions : np.ndarray
        Document-topic distribution matrix of shape (n_documents, n_topics)
    lda_model : LatentDirichletAllocation
        Fitted LDA model (can be pickled for later use)
    vectorizer : TfidfVectorizer
        Fitted vectorizer used for LDA input (can be pickled for later use)
        
    Examples
    --------
    >>> corpus = ["water leak emergency", "noise complaint routine", "broken streetlight"]
    >>> topics, model, vectorizer = fit_lda(corpus, n_topics=2)
    >>> topics.shape
    (3, 2)
    >>> topics.sum(axis=1)  # Each row sums to ~1.0
    array([1., 1., 1.])
    """
    # Filter out empty strings
    corpus_clean = [doc if doc else " " for doc in corpus]
    
    # Vectorize with TF-IDF (or could use CountVectorizer)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    doc_term_matrix = vectorizer.fit_transform(corpus_clean)
    
    # Fit LDA
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method='online',
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    topic_distributions = lda_model.fit_transform(doc_term_matrix)
    
    return topic_distributions, lda_model, vectorizer


def get_top_lda_terms(
    lda_model: LatentDirichletAllocation,
    vectorizer: TfidfVectorizer,
    n_terms: int = 10
) -> List[List[str]]:
    """
    Extract top terms for each LDA topic.
    
    Helper function to interpret LDA results by showing the most
    representative terms for each topic.
    
    Parameters
    ----------
    lda_model : LatentDirichletAllocation
        Fitted LDA model
    vectorizer : TfidfVectorizer
        Fitted vectorizer used with the LDA model
    n_terms : int, default=10
        Number of top terms to return per topic
        
    Returns
    -------
    List[List[str]]
        List of topics, each containing the top n_terms for that topic
        
    Examples
    --------
    >>> topics, model, vectorizer = fit_lda(corpus, n_topics=2)
    >>> top_terms = get_top_lda_terms(model, vectorizer, n_terms=5)
    >>> len(top_terms)
    2
    >>> len(top_terms[0])
    5
    """
    feature_names = vectorizer.get_feature_names_out()
    top_terms_list = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        # Get indices of top terms
        top_indices = topic.argsort()[-n_terms:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        top_terms_list.append(top_terms)
    
    return top_terms_list
