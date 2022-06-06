"""
anamod master pipeline
Given test samples, a trained model and a feature hierarchy,
computes the effect on the model's output loss after perturbing the features/feature groups
in the hierarchy.
"""

import importlib
import csv
import os
import pickle
import sys

import cloudpickle
import numpy as np

from anamod.core import constants, utils
from anamod.core.pipelines import CondorPipeline, SerialPipeline
from anamod.visualization.analysis import visualize_hierarchical, visualize_temporal


def main(args):
    """Parse arguments from command-line"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.rng = np.random.default_rng(args.seed)
    args.logger = utils.get_logger(__name__, f"{args.output_dir}/anamod.log")
    validate_args(args)
    return pipeline(args)


def pipeline(args):
    """Master pipeline"""
    # TODO: 'args' is now an object. Change to reflect that and figure out way to print object attributes
    args.logger.info(f"Begin anamod master pipeline with args: {args}")
    # Perturb features
    worker_pipeline = CondorPipeline(args) if args.condor else SerialPipeline(args)
    analyzed_features = worker_pipeline.run()
    write_outputs(args, analyzed_features)
    visualize(args, analyzed_features)
    args.logger.info("End anamod master pipeline")
    return analyzed_features


def write_outputs(args, features):
    """Write outputs to file"""
    features_filename = f"{args.output_dir}/{constants.FEATURE_IMPORTANCE}.cpkl"
    with open(features_filename, "wb") as features_file:
        cloudpickle.dump(features, features_file, protocol=pickle.DEFAULT_PROTOCOL)
    csv_filename = f"{args.output_dir}/{constants.FEATURE_IMPORTANCE}.csv"
    attributes = ["name", "important", "importance_score", "pvalue"]
    if args.analysis_type == constants.TEMPORAL:
        attributes += ["ordering_important", "ordering_pvalue",
                       "window", "window_important", "window_importance_score", "window_pvalue",
                       "window_ordering_important", "window_ordering_pvalue"]
    with open(csv_filename, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(attributes)
        for feature in features:
            writer.writerow([getattr(feature, attribute) for attribute in attributes])
    print(f"Summary of important features: {csv_filename}")


def visualize(args, features):
    """Visualize outputs"""
    if not args.visualize:
        return
    if args.analysis_type == constants.TEMPORAL:
        sequence_length = args.data.shape[2]
        visualize_temporal(args, features, sequence_length)
    elif args.analysis_type == constants.HIERARCHICAL:
        visualize_hierarchical(args, features)


def validate_args(args):
    """Validate arguments"""
    if args.condor:
        try:
            importlib.import_module("htcondor")
        except ModuleNotFoundError:
            print("htcondor module not found. "
                  "Use 'pip install htcondor' to install htcondor on a compatible platform, or "
                  "disable condor", file=sys.stderr)
            raise
