"""Constant definitions"""

# TODO: Remove redundant constants

# Arguments
HIERARCHICAL = "hierarchical"

# Features
NAME = "Name"
IMPORTANT = "Important"
IMPORTANCE_SCORE = "Importance Score"
WINDOW_IMPORTANT = "Window Important"
WINDOW_IMPORTANCE_SCORE = "Window Importance Score"
ORDERING_IMPORTANT = "Ordering Important"

# Hierarchy
CLUSTER_FROM_DATA = "cluster_from_data"
RANDOM = "random"
FLAT = "flat"

# Temporal analysis
EFFECT_SIZE = "effect_size"
IMPORTANCE_TEST = "importance_test"
CHOICES_WINDOW_SEARCH_ALGORITHM = {EFFECT_SIZE}  # {EFFECT_SIZE, IMPORTANCE_TEST}

# Condor
POLL_BASED_TRACKING = "poll_based_tracking"
EVENT_LOG_TRACKING = "event_log_tracking"
CONDOR_MAX_RUNNING_TIME = 72 * 3600
CONDOR_MAX_WAIT_TIME = 1200  # Time to wait for job to start running before retrying
CONDOR_MAX_RETRIES = 50
# Reference: https://htcondor.readthedocs.io/en/latest/classad-attributes/job-classad-attributes.html
CONDOR_HOLD_RETRY_CODES = set([6, 7, 8, 9, 10, 11, 12, 13, 14])
# List of hosts to avoid (if causing issues):
CONDOR_AVOID_HOSTS = ["mastodon-5.biostat.wisc.edu", "qlu-1.biostat.wisc.edu",
                      "e269.chtc.wisc.edu", "e1039.chtc.wisc.edu",  # jobs freeze indefinitely
                      "chief.biostat.wisc.edu", "mammoth-1.biostat.wisc.edu", "nebula-7.biostat.wisc.edu",
                      "mastodon-1.biostat.wisc.edu"]   # don't seem to be on shared filesystem
CONDOR_MAX_ERROR_COUNT = 100  # Maximum number of condor errors to tolerate before aborting (for a variety of possible reasons)
QUEUE, REMOVE, QUERY, HISTORY = ("queue", "remove", "query", "history")  # Condor scheduler actions

# Master I/O
MODEL_FILENAME = "model.cpkl"
DATA_FILENAME = "data.hdf5"
FEATURE_IMPORTANCE = "feature_importance"
FEATURE_IMPORTANCE_HIERARCHY = f"{FEATURE_IMPORTANCE}_hierarchy"
FEATURE_IMPORTANCE_WINDOWS = f"{FEATURE_IMPORTANCE}_windows"

# Worker I/O
INPUT_FEATURES_FILENAME = "{}/input_features_worker_{}.cpkl"
OUTPUT_FEATURES_FILENAME = "{}/output_features_worker_{}.cpkl"
RESULTS_FILENAME = "{}/results_worker_{}.hdf5"

# Hypothesis testing
PVALUE = "p-value"
PAIRED_TTEST = "paired-t-test"
WILCOXON_TEST = "wilcoxon-test"
PVALUES_FILENAME = "pvalues.csv"
LESS = "less"
GREATER = "greater"
TWOSIDED = "two-sided"
# Permutation test statistics
MEAN_LOSS = "mean_loss"
MEAN_LOG_LOSS = "mean_log_loss"
MEDIAN_LOSS = "median_loss"
RELATIVE_MEAN_LOSS = "relative_mean_loss"
SIGN_LOSS = "sign_loss"
CHOICES_TEST_STATISTICS = [MEAN_LOSS, MEAN_LOG_LOSS, MEDIAN_LOSS, RELATIVE_MEAN_LOSS, SIGN_LOSS]


# Interactions
INTERACTIONS_PVALUES_FILENAME = "interaction_pvalues.csv"
DUMMY_ROOT = "dummy_root"
INTERACTIONS_FDR_DIR = "interaction_fdr_results"

# HDF5
LOSSES = "losses"
PREDICTIONS = "predictions"
RECORD_IDS = "record_ids"
TARGETS = "targets"
TABULAR = "tabular"
TEMPORAL = "temporal"
DATA = "data"

# Simulation
RELEVANT = "relevant"
IRRELEVANT = "irrelevant"
FDR = "FDR"
POWER = "Power"
OUTER_NODES_FDR = "Outer_Nodes_FDR"
OUTER_NODES_POWER = "Outer_Nodes_Power"
BASE_FEATURES_FDR = "Base_Features_FDR"
BASE_FEATURES_POWER = "Base_Features_Power"
INTERACTIONS_FDR = "Interactions_FDR"
INTERACTIONS_POWER = "Interactions_Power"
ORDERING_ALL_IMPORTANT_FDR = "Ordering_All_Important_FDR"
ORDERING_ALL_IMPORTANT_POWER = "Ordering_All_Important_Power"
ORDERING_IDENTIFIED_IMPORTANT_FDR = "Ordering_Identified_Important_FDR"
ORDERING_IDENTIFIED_IMPORTANT_POWER = "Ordering_Identified_Important_Power"
AVERAGE_WINDOW_FDR = "Average_Window_FDR"
AVERAGE_WINDOW_POWER = "Average_Window_Power"
WINDOW_OVERLAP = "Window_Overlap"
WINDOW_IMPORTANT_FDR = "Window_Important_FDR"
WINDOW_IMPORTANT_POWER = "Window_Important_Power"
WINDOW_ORDERING_IMPORTANT_FDR = "Window_Ordering_Important_FDR"
WINDOW_ORDERING_IMPORTANT_POWER = "Window_Ordering_Important_Power"
OVERALL_SCORES_CORR = "overall_scores_corr"
WINDOW_SCORES_CORR = "window_scores_corr"
OVERALL_RELEVANT_SCORES_CORR = "overall_relevant_scores_corr"
WINDOW_RELEVANT_SCORES_CORR = "window_relevant_scores_corr"
SIMULATION_RESULTS = "simulation_results"
SIMULATION_SUMMARY_FILENAME = "simulation_summary.json"
SYNTHESIZED_FEATURES_FILENAME = "synthesized_features.cpkl"
ANALYZED_FEATURES_FILENAME = "analyzed_features.cpkl"
MODEL_WRAPPER_FILENAME = "model_wrapper.cpkl"
TARGETS_FILENAME = "targets.npy"

# Trial (multiple simulations)
DEFAULT = "DEFAULT"
INSTANCE_COUNTS = "instance_counts"
NOISE_LEVELS = "noise_levels"
FEATURE_COUNTS = "feature_counts"
PERMUTATION_COUNTS = "permutation_counts"
SEQUENCE_LENGTHS = "sequence_lengths"
WINDOW_SEQUENCE_DEPENDENCE = "window_sequence_dependence"
MODEL_TYPES = "model_types"
TEST = "test"
ALL_SIMULATIONS_SUMMARY = "all_simulations_summary"
CONFIG_HIERARCHICAL = "config_run_simulations_hierarchical.ini"
CONFIG_TEMPORAL = "config_run_simulations_temporal.ini"
ALL_TRIALS_SUMMARY_FILENAME = "all_trials_summary.json"
CONFIG, MODEL, RESULTS = ("config", "model", "results")  # simulation output JSON categories
MAX_ATTEMPTS = 10

# Hierarchical FDR
HIERARCHICAL_FDR_DIR = "hierarchical_fdr_results"
HIERARCHICAL_FDR_OUTPUTS = "hierarchical_fdr_outputs"
ADJUSTED_PVALUE = "adjusted_p-value"
REJECTED_STATUS = "rejected_status"
# Dependence assumptions
POSITIVE = "positive"
ARBITRARY = "arbitrary"
# Procedures
YEKUTIELI = "yekutieli"
LYNCH_GUO = "lynch_guo"
TREE = "tree"

# Perturbation
ZEROING = "zeroing"
PERMUTATION = "permutation"
RNG_SEED = "rng_seed"
ACROSS_INSTANCES = "across"
WITHIN_INSTANCE = "within"  # TODO: rename to ACROSS_TIMESTEPS
PVALUE_THRESHOLD = 0.05
QUADRATIC_LOSS = "quadratic_loss"
ABSOLUTE_DIFFERENCE_LOSS = "absolute_difference_loss"
BINARY_CROSS_ENTROPY = "binary_cross_entropy"
ZERO_ONE_LOSS = "zero_one_loss"
CHOICES_LOSS_FUNCTIONS = {None, QUADRATIC_LOSS, ABSOLUTE_DIFFERENCE_LOSS, BINARY_CROSS_ENTROPY, ZERO_ONE_LOSS}
LABELS = "labels"
BASELINE_PREDICTIONS = "baseline_predictions"

# Default values
DEFAULT_OUTPUT_DIR = "anamod_outputs"
DEFAULT_NUM_PERMUTATIONS = 50
AUTO = "auto"
AUTO_R2 = 0.9  # R2 to aim for when auto-selecting noise

# Miscellaneous
BASELINE = "baseline"
SEED = 13997
NEAR_ZERO = 1e-10  # To handle floating-point errors
