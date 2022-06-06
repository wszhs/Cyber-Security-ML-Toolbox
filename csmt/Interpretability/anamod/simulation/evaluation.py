"""Evaluate simulation results"""

import warnings

import anytree
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from synmod.constants import REGRESSOR

from anamod.core import constants
from anamod.core.constants import FDR, POWER, BASE_FEATURES_FDR, BASE_FEATURES_POWER
from anamod.core.constants import ORDERING_ALL_IMPORTANT_FDR, ORDERING_ALL_IMPORTANT_POWER
from anamod.core.constants import ORDERING_IDENTIFIED_IMPORTANT_FDR, ORDERING_IDENTIFIED_IMPORTANT_POWER
from anamod.core.constants import AVERAGE_WINDOW_FDR, AVERAGE_WINDOW_POWER, WINDOW_OVERLAP
from anamod.core.constants import WINDOW_IMPORTANT_FDR, WINDOW_IMPORTANT_POWER, WINDOW_ORDERING_IMPORTANT_FDR, WINDOW_ORDERING_IMPORTANT_POWER
from anamod.core.constants import OVERALL_SCORES_CORR, WINDOW_SCORES_CORR, OVERALL_RELEVANT_SCORES_CORR, WINDOW_RELEVANT_SCORES_CORR


def evaluate(args, sfeatures, afeatures):
    """Evaluate results"""
    if args.analysis_type == constants.HIERARCHICAL:
        return evaluate_hierarchical(args, sfeatures, afeatures)
    return evaluate_temporal(args, sfeatures, afeatures)


def get_precision_recall(true, inferred):
    """Get precision and recall given vectors of true and inferred values"""
    precision, recall, _, _ = precision_recall_fscore_support(true, inferred, average="binary", zero_division=1)
    return precision, recall


def evaluate_hierarchical(args, sfeatures, afeatures):
    """
    Evaluate hierarchical analysis results - obtain power/FDR measures for all nodes/base features
    """
    # pylint: disable = too-many-locals
    # Map features in hierarchy to original features and identify ground-truth importances/scores
    sfeatures_map = {sfeature.name: sfeature for sfeature in sfeatures}
    importance_map = {}
    score_map = {}
    for node in anytree.PostOrderIter(afeatures[0].root):
        if node.is_leaf:
            sfeature_name = str(node.idx[0])
            importance_map[node.name] = sfeatures_map[sfeature_name].important
            score_map[node.name] = sfeatures_map[sfeature_name].effect_size
        else:
            importance_map[node.name] = any(importance_map[child.name] for child in node.children)
    # Overall FDR/power
    important = np.zeros(len(afeatures))
    inferred_important = np.zeros(len(afeatures))
    for idx, afeature in enumerate(afeatures):
        important[idx] = importance_map[afeature.name]
        inferred_important[idx] = afeature.important
    imp_precision, imp_recall = get_precision_recall(important, inferred_important)
    # Base features FDR/power
    base_features = list(filter(lambda node: node.is_leaf, afeatures))
    base_important = np.zeros(len(base_features))
    inferred_base_important = np.zeros(len(base_features))
    for idx, base_feature in enumerate(base_features):
        base_important[idx] = importance_map[base_feature.name]
        inferred_base_important[idx] = base_feature.important
    base_imp_precision, base_imp_recall = get_precision_recall(base_important, inferred_base_important)
    # Importance scores for base features
    overall_scores_corr, overall_relevant_scores_corr = (1.0, 1.0)
    if args.model_type == REGRESSOR:
        scores = np.zeros(len(base_features))
        inferred_scores = np.zeros(len(base_features))
        for idx, base_feature in enumerate(base_features):
            scores[idx] = score_map[base_feature.name]
            inferred_scores[idx] = base_feature.importance_score
        relevant_base_features = list(filter(lambda node: importance_map[node.name], base_features))
        relevant_scores = np.zeros(len(relevant_base_features))
        relevant_inferred_scores = np.zeros(len(relevant_base_features))
        for idx, relevant_base_feature in enumerate(relevant_base_features):
            relevant_scores[idx] = score_map[relevant_base_feature.name]
            relevant_inferred_scores[idx] = relevant_base_feature.importance_score
        overall_scores_corr = pearsonr(scores, inferred_scores)[0] if len(scores) >= 2 else 1
        overall_relevant_scores_corr = pearsonr(relevant_scores, relevant_inferred_scores)[0] if len(relevant_scores) >= 2 else 1

    vals = {FDR: 1 - imp_precision, POWER: imp_recall,
            BASE_FEATURES_FDR: 1 - base_imp_precision, BASE_FEATURES_POWER: base_imp_recall,
            OVERALL_SCORES_CORR: overall_scores_corr, OVERALL_RELEVANT_SCORES_CORR: overall_relevant_scores_corr}
    return {key: value if isinstance(value, dict) else round(value, 10) for key, value in vals.items()}  # Round values to avoid FP discrepancies


def evaluate_temporal(args, sfeatures, afeatures):
    """Evaluate results of temporal model analysis - obtain power/FDR measures for importance, temporal importance and windows"""
    # pylint: disable = protected-access, too-many-locals, invalid-name, too-many-statements
    # TODO: Refactor
    afeatures = list(filter(lambda afeature: len(afeature.idx) == 1, afeatures))  # Only evaluate base features, not feature groups
    # TODO: Measure power and FDR w.r.t. feature groups as well
    num_features = len(afeatures)

    def init_vectors():
        """Initialize vectors indicating importances"""
        important = np.zeros(num_features, dtype=bool)
        ordering_important = np.zeros(num_features, dtype=bool)
        windows = np.zeros((len(afeatures), args.sequence_length))
        window_important = np.zeros(num_features, dtype=bool)
        window_ordering_important = np.zeros(num_features, dtype=bool)
        return important, ordering_important, windows, window_important, window_ordering_important

    # Populate importance vectors (ground truth and inferred)
    afeatures = sorted(afeatures, key=lambda afeature: afeature.idx[0])  # To ensure features are ordered by their index in the feature vector
    important, ordering_important, windows, window_important, window_ordering_important = init_vectors()
    inferred_important, inferred_ordering_important, inferred_windows, inferred_window_important, inferred_window_ordering_important = init_vectors()
    for idx, afeature in enumerate(afeatures):
        assert idx == afeature.idx[0]
        sfeature = sfeatures[idx]
        # Ground truth values
        if sfeature.important:
            important[idx] = sfeature.important
            window_important[idx] = sfeature.window_important
            assert sfeature.window_important  # All relevant features have windows
            left, right = sfeature.window
            windows[idx][left: right + 1] = 1
            window_ordering_important[idx] = sfeature.window_ordering_important
            ordering_important[idx] = sfeature.ordering_important
        # Inferred values
        if afeature.important:
            inferred_important[idx] = afeature.important  # Overall importance after performing FDR control
            inferred_window_important[idx] = afeature.window_important
            if afeature.temporal_window is not None:
                left, right = afeature.temporal_window
                inferred_windows[idx][left: right + 1] = 1
            inferred_window_ordering_important[idx] = afeature.window_ordering_important
            inferred_ordering_important[idx] = afeature.ordering_important

    imp_precision, imp_recall = get_precision_recall(important, inferred_important)
    ordering_all_precision, ordering_all_recall = get_precision_recall(ordering_important, inferred_ordering_important)
    tidx = [idx for idx, afeature in enumerate(afeatures) if afeature.important]  # Features tested for temporal properties
    ordering_identified_precision, ordering_identified_recall = get_precision_recall(ordering_important[tidx], inferred_ordering_important[tidx])
    window_imp_precision, window_imp_recall = get_precision_recall(window_important[tidx], inferred_window_important[tidx])
    window_ordering_precision, window_ordering_recall = get_precision_recall(window_ordering_important[tidx],
                                                                             inferred_window_ordering_important[tidx])

    # Window metrics for relevant features
    window_results = {}
    for idx, afeature in enumerate(afeatures):
        if not (afeature.important and sfeatures[idx].important):
            # Ignore features that were not important or not identified as important
            # Motivation: to evaluate temporal localization conditioned on correct identification of overall relevance
            continue
        window_precision, window_recall = get_precision_recall(windows[idx], inferred_windows[idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Avoid warning if the two vectors have no common values
            # TODO: Is balanced accuracy score the best metric for measuring window overlap?
            # Leads to mismatch w.r.t. window power
            window_overlap = balanced_accuracy_score(windows[idx], inferred_windows[idx])
        window_results[idx] = {"precision": window_precision, "recall": window_recall, "overlap": window_overlap}
    avg_window_precision = np.mean([result["precision"] for result in window_results.values()]) if window_results else 1.
    avg_window_recall = np.mean([result["recall"] for result in window_results.values()]) if window_results else 0.
    window_overlaps = {idx: result["overlap"] for idx, result in window_results.items()}

    # Importance scores
    scores = [sfeature.effect_size for sfeature in sfeatures]
    inferred_scores = [afeature.overall_effect_size for afeature in afeatures]
    window_scores = [sfeature.effect_size for idx, sfeature in enumerate(sfeatures) if afeatures[idx].window_important]
    inferred_window_scores = [afeature.window_effect_size for afeature in afeatures if afeature.window_important]
    overall_scores_corr, window_scores_corr = (1.0, 1.0)
    relevant_scores = [sfeature.effect_size for sfeature in sfeatures if sfeature.important]
    relevant_inferred_scores = [afeature.overall_effect_size for idx, afeature in enumerate(afeatures) if sfeatures[idx].important]
    relevant_window_scores = [sfeature.effect_size for idx, sfeature in enumerate(sfeatures)
                              if sfeature.important and afeatures[idx].window_important]
    relevant_inferred_window_scores = [afeature.window_effect_size for idx, afeature in enumerate(afeatures)
                                       if sfeatures[idx].important and afeatures[idx].window_important]
    overall_relevant_scores_corr, window_relevant_scores_corr = (1.0, 1.0)
    if args.model_type == REGRESSOR:
        overall_scores_corr = pearsonr(scores, inferred_scores)[0] if len(scores) >= 2 else 1
        window_scores_corr = pearsonr(window_scores, inferred_window_scores)[0] if len(window_scores) >= 2 else 1
        overall_relevant_scores_corr = pearsonr(relevant_scores, relevant_inferred_scores)[0] if len(relevant_scores) >= 2 else 1
        window_relevant_scores_corr = pearsonr(relevant_window_scores, relevant_inferred_window_scores)[0] if len(relevant_window_scores) >= 2 else 1

    vals = {FDR: 1 - imp_precision, POWER: imp_recall,
            ORDERING_ALL_IMPORTANT_FDR: 1 - ordering_all_precision, ORDERING_ALL_IMPORTANT_POWER: ordering_all_recall,
            ORDERING_IDENTIFIED_IMPORTANT_FDR: 1 - ordering_identified_precision, ORDERING_IDENTIFIED_IMPORTANT_POWER: ordering_identified_recall,
            AVERAGE_WINDOW_FDR: 1 - avg_window_precision, AVERAGE_WINDOW_POWER: avg_window_recall,
            WINDOW_OVERLAP: window_overlaps,
            WINDOW_IMPORTANT_FDR: 1 - window_imp_precision, WINDOW_IMPORTANT_POWER: window_imp_recall,
            WINDOW_ORDERING_IMPORTANT_FDR: 1 - window_ordering_precision, WINDOW_ORDERING_IMPORTANT_POWER: window_ordering_recall,
            OVERALL_SCORES_CORR: overall_scores_corr, WINDOW_SCORES_CORR: window_scores_corr,
            OVERALL_RELEVANT_SCORES_CORR: overall_relevant_scores_corr, WINDOW_RELEVANT_SCORES_CORR: window_relevant_scores_corr}
    return {key: value if isinstance(value, dict) else round(value, 10) for key, value in vals.items()}  # Round values to avoid FP discrepancies
