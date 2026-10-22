"""Retrieve catalogue entries that match the customer's stated interests."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from products import GOAL_LABELS, PRODUCTS


def retrieve_offers(selected_interests: list[str]) -> list[dict]:
    """Rank static catalogue descriptions against customer-selected interests."""
    if not selected_interests:
        return []

    product_items = list(PRODUCTS.items())
    product_texts = [
        f"{product['name']} {product['summary']} {product['search_text']}"
        for _, product in product_items
    ]
    query = " ".join(selected_interests)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([query, *product_texts])
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()

    offers = []
    for (key, product), score in zip(product_items, scores):
        offers.append({
            "key": key,
            **product,
            "relevance_score": float(score),
            "selected": GOAL_LABELS[product["goal"]] in selected_interests,
        })
    return sorted(offers, key=lambda offer: offer["relevance_score"], reverse=True)
