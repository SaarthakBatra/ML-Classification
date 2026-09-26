"""
Evaluation Package for Candidate Generation Quality and Statistical Distributions.
"""

from evaluation.metrics import evaluate_blocking
from evaluation.analyzer import analyze_candidate_file

__all__ = [
    "evaluate_blocking",
    "analyze_candidate_file",
]
