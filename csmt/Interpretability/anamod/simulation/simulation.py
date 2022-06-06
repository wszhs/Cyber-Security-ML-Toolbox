"""Generates simulated data and model to test anamod algorithm"""

import argparse
import copy
from distutils.util import strtobool
import json
import os
import pickle
import pprint
import sys

import anytree
from anytree.importer.jsonimporter import JsonImporter
import cloudpickle
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import r2_score
import synmod.master
from synmod.constants import CLASSIFIER, REGRESSOR, FEATURES_FILENAME, MODEL_FILENAME, INSTANCES_FILENAME

from anamod.core import constants, utils, ModelAnalyzer, TemporalModelAnalyzer
from anamod.core.master import validate_args
from anamod.core.utils import CondorJobWrapper
from anamod.simulation.model_wrapper import ModelWrapper
from anamod.simulation import evaluation


def main():
    """Main"""
    # pylint: disable = too-many-statements
    parser = argparse.ArgumentParser("python anamod.simulation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Required arguments
    required = parser.add_argument_group("Required parameters")
    required.add_argument("-output_dir", help="Name of output directory")
    # Optional common arguments
    common = parser.add_argument_group("Optional common parameters")
    common.add_argument("-analysis_type", help="Type of model analysis to perform",
                        default=constants.TEMPORAL, choices=[constants.TEMPORAL, constants.HIERARCHICAL])
    common.add_argument("-seed", type=int, default=constants.SEED)
    common.add_argument("-visualize", type=strtobool, default=False)
    common.add_argument("-num_instances", type=int, default=200)
    common.add_argument("-num_features", type=int, default=10)
    common.add_argument("-fraction_relevant_features", type=float, default=.2)
    common.add_argument("-loss_target_values", choices=[constants.LABELS, constants.BASELINE_PREDICTIONS], default=constants.LABELS,
                        help=("Target values to compare perturbed values to while computing losses. "
                              "Note: baseline predictions here refer to oracle's noise-free predictions. "
                              "If noise multiplier is non-zero, noise is added when computing baseline losses, "
                              "else all baseline losses would be zero "
                              "while all perturbed losses would be positive (for quadratic loss, default for non-label predictions)"))
    common.add_argument("-num_interactions", type=int, default=0, help="number of interaction pairs in model")
    common.add_argument("-include_interaction_only_features", help="include interaction-only features in model"
                        " in addition to linear + interaction features (default enabled)", type=strtobool, default=True)
    common.add_argument("-condor", help="Use condor for parallelization", type=strtobool, default=False)
    common.add_argument("-shared_filesystem", type=strtobool, default=False)
    common.add_argument("-features_per_worker", type=int, default=10)
    common.add_argument("-cleanup", type=strtobool, default=True, help="Clean data and model files after completing simulation")
    common.add_argument("-condor_cleanup", type=strtobool, default=True, help="Clean condor cmd/out/err/log files after completing simulation")
    common.add_argument("-avoid_bad_hosts", type=strtobool, default=True)
    common.add_argument("-retry_arbitrary_failures", type=strtobool, default=True)
    common.add_argument("-synthesis_dir", help="Directory for synthesized data. If none provided, will be set as {output_dir}/synthesis")
    common.add_argument("-synthesize_only", type=strtobool, default=False, help="Synthesize data and stop (skip analysis/evaluation)")
    common.add_argument("-evaluate_only", type=strtobool, default=False, help="Assume results already exist and rerun evaluation")
    # Hierarchical feature importance analysis arguments
    hierarchical = parser.add_argument_group("Hierarchical feature analysis arguments")
    hierarchical.add_argument("-noise_multiplier", default=0.,
                              help=("Multiplicative factor for noise added to polynomial computation for irrelevant features; "
                                    f"if '{constants.AUTO}', selected automatically to get R^2 value of 0.9 (regressor only)"))
    hierarchical.add_argument("-hierarchy_filename", help="If provided, will be passed to anamod instead of creating "
                              "a flat/random/clustering-based hierarchy", type=str)
    hierarchical.add_argument("-hierarchy_type", help="Choice of hierarchy to generate", default=constants.FLAT,
                              choices=[constants.CLUSTER_FROM_DATA, constants.RANDOM, constants.FLAT])
    hierarchical.add_argument("-contiguous_node_names", type=strtobool, default=False, help="enable to change node names in hierarchy "
                              "to be contiguous for better visualization (but creating mismatch between node names and features indices)")
    hierarchical.add_argument("-analyze_interactions", help="enable analyzing interactions", type=strtobool, default=False)
    hierarchical.add_argument("-perturbation", default=constants.PERMUTATION, choices=[constants.PERMUTATION])
    hierarchical.add_argument("-num_permutations", type=int, default=constants.DEFAULT_NUM_PERMUTATIONS,
                              help="Number of permutations to perform in permutation test")
    # Temporal model analysis arguments
    temporal = parser.add_argument_group("Temporal model analysis arguments")
    temporal.add_argument("-sequence_length", help="sequence length for temporal models", type=int, default=20)
    temporal.add_argument("-model_type", default=REGRESSOR, choices=[CLASSIFIER, REGRESSOR])
    temporal.add_argument("-sequences_independent_of_windows", type=strtobool, dest="window_independent")
    temporal.add_argument("-standardize_features", type=strtobool, default=True)
    temporal.add_argument("-feature_type_distribution", nargs=3, type=float, default=[0.25, 0.25, 0.50])
    temporal.set_defaults(window_independent=False)

    args, pass_args = parser.parse_known_args()
    validate_args(args)
    if args.evaluate_only:
        assert args.analysis_type != constants.HIERARCHICAL, "-evaluate_only not currently supported with hierarchical analysis"
    if not args.output_dir:
        args.output_dir = (f"sim_outputs_inst_{args.num_instances}_feat_{args.num_features}_noise_{args.noise_multiplier:.3f}_"
                           f"relfraction_{args.fraction_relevant_features:.3f}_pert_{args.perturbation}_shufftrials_{args.num_permutations}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not args.synthesis_dir:
        args.synthesis_dir = f"{args.output_dir}/synthesis"
    args.rng = np.random.default_rng(args.seed)
    args.logger = utils.get_logger(__name__, f"{args.output_dir}/simulation.log")
    return pipeline(args, pass_args)


def pipeline(args, pass_args):
    """Simulation pipeline"""
    args.logger.info(f"Begin anamod simulation with args: {args}")
    synthesized_features, data, model_wrapper, targets = synthesize(args)
    analyzed_features = analyze(args, pass_args, synthesized_features, data, model_wrapper, targets)
    results, model_wrapper = evaluate(args, synthesized_features, model_wrapper, analyzed_features)
    summary = write_summary(args, model_wrapper, results)
    args.logger.info("End anamod simulation")
    return summary


def synthesize(args):
    """Synthesize data and model"""
    if args.evaluate_only:
        return (None, None, None, None)
    try:
        # Load synthesized/intermediate files if already generated
        # TODO: maybe add option to toggle reusing old generated files
        synthesized_features, data, _ = read_synthesized_inputs(args.synthesis_dir)
        model_wrapper, targets = read_intermediate_inputs(args.synthesis_dir)
    except FileNotFoundError:
        synthesized_features, data, model = run_synmod(args)
        targets = model.predict(data, labels=True) if args.loss_target_values == constants.LABELS else model.predict(data)
        noise_multiplier = noise_selection(args, data, targets, model)
        # Create wrapper around ground-truth model
        model_wrapper = ModelWrapper(model, noise_multiplier)
        write_intermediate_inputs(args.synthesis_dir, model_wrapper, targets)
    args.noise_multiplier = model_wrapper.noise_multiplier
    return synthesized_features, data[:args.num_instances], model_wrapper, targets[:args.num_instances]


def read_synthesized_inputs(synthesis_dir):
    """Read inputs for model analysis"""
    with open(f"{synthesis_dir}/{FEATURES_FILENAME}", "rb") as data_file:
        synthesized_features = cloudpickle.load(data_file)
    data = np.load(f"{synthesis_dir}/{INSTANCES_FILENAME}")
    with open(f"{synthesis_dir}/{MODEL_FILENAME}", "rb") as model_file:
        model = cloudpickle.load(model_file)
    return synthesized_features, data, model


def read_intermediate_inputs(synthesis_dir):
    """Read intermediate inputs"""
    with open(f"{synthesis_dir}/{constants.MODEL_WRAPPER_FILENAME}", "rb") as model_wrapper_file:
        model_wrapper = cloudpickle.load(model_wrapper_file)
    targets = np.load(f"{synthesis_dir}/{constants.TARGETS_FILENAME}")
    return model_wrapper, targets


def write_intermediate_inputs(synthesis_dir, model_wrapper, targets):
    """Write intermediate inputs"""
    with open(f"{synthesis_dir}/{constants.MODEL_WRAPPER_FILENAME}", "wb") as model_wrapper_file:
        cloudpickle.dump(model_wrapper, model_wrapper_file)
    targets = np.save(f"{synthesis_dir}/{constants.TARGETS_FILENAME}", targets)


def analyze(args, pass_args, synthesized_features, data, model_wrapper, targets):
    """Analyze model"""
    # pylint: disable = too-many-arguments
    if args.synthesize_only or args.evaluate_only:
        return (None, None)
    hierarchy_root = None
    if args.hierarchy_filename:
        # Load hierarchy from file
        with open(args.hierarchy_filename, encoding="utf-8") as hierarchy_file:
            hierarchy_root = JsonImporter().read(hierarchy_file)
    elif args.analysis_type == constants.HIERARCHICAL:
        # Generate hierarchy if required
        hierarchy_root, _ = gen_hierarchy(args, data)
        if hierarchy_root:
            # Update hierarchy descriptions for future visualization
            update_hierarchy_descriptions(hierarchy_root, model_wrapper.ground_truth_model.relevant_feature_map, synthesized_features)
    # Invoke feature importance algorithm
    analyzed_features = run_anamod(args, pass_args, data, model_wrapper, targets, hierarchy_root)
    return analyzed_features


def evaluate(args, synthesized_features, model_wrapper, analyzed_features):
    """Evaluate results of analysis"""
    # pylint: disable = too-many-arguments
    if args.synthesize_only:
        return (None, model_wrapper)
    if args.evaluate_only:
        synthesized_features, model_wrapper, analyzed_features = read_outputs(args.output_dir)
    else:
        write_outputs(args.output_dir, synthesized_features, model_wrapper, analyzed_features)
    # Evaluate anamod outputs - power/FDR, importance score correlations
    results = evaluation.evaluate(args, synthesized_features, analyzed_features)
    return results, model_wrapper


def read_outputs(output_dir):
    """Read simulation outputs assuming already generated, for evaluation"""
    with open(f"{output_dir}/{constants.SYNTHESIZED_FEATURES_FILENAME}", "rb") as synthesized_features_file:
        synthesized_features = cloudpickle.load(synthesized_features_file)
    with open(f"{output_dir}/{constants.MODEL_WRAPPER_FILENAME}", "rb") as model_wrapper_file:
        model_wrapper = cloudpickle.load(model_wrapper_file)
    with open(f"{output_dir}/{constants.ANALYZED_FEATURES_FILENAME}", "rb") as analyzed_features_file:
        analyzed_features = cloudpickle.load(analyzed_features_file)
    return synthesized_features, model_wrapper, analyzed_features


def write_outputs(output_dir, synthesized_features, model_wrapper, analyzed_features):
    """Write simulation inputs and outputs (model and features)"""
    with open(f"{output_dir}/{constants.SYNTHESIZED_FEATURES_FILENAME}", "wb") as synthesized_features_file:
        cloudpickle.dump(synthesized_features, synthesized_features_file, protocol=pickle.DEFAULT_PROTOCOL)
    with open(f"{output_dir}/{constants.MODEL_WRAPPER_FILENAME}", "wb") as model_wrapper_file:
        cloudpickle.dump(model_wrapper, model_wrapper_file, protocol=pickle.DEFAULT_PROTOCOL)
    with open(f"{output_dir}/{constants.ANALYZED_FEATURES_FILENAME}", "wb") as analyzed_features_file:
        cloudpickle.dump(analyzed_features, analyzed_features_file, protocol=pickle.DEFAULT_PROTOCOL)


def run_synmod(oargs):
    """Synthesize data and model"""
    args = configure_synthesis_args(oargs)
    args.logger.info("Begin running synmod")
    if args.condor:
        # Spawn condor job to synthesize data
        # Compute size requirements
        data_size = args.num_instances * args.num_features * args.sequence_length // (8 * (2 ** 30))  # Data size in GB
        memory_requirement = f"{1 + data_size}GB"
        disk_requirement = f"{4 + data_size}GB"
        # Set up command-line arguments
        args.sequences_independent_of_windows = args.window_independent
        cmd = "python3 -m synmod"
        job_dir = args.output_dir
        args.output_dir = os.path.abspath(args.output_dir) if args.shared_filesystem else os.path.basename(args.output_dir)
        for arg in ["output_dir", "num_features", "num_instances", "synthesis_type",
                    "fraction_relevant_features", "num_interactions", "include_interaction_only_features", "seed", "write_outputs",
                    "sequence_length", "sequences_independent_of_windows", "model_type", "standardize_features", "feature_type_distribution"]:
            cmd += f" -{arg} {args.__getattribute__(arg)}"
        args.logger.info(f"Running cmd: {cmd}")
        # Launch and monitor job
        job = CondorJobWrapper(cmd, [], job_dir, shared_filesystem=args.shared_filesystem, memory=memory_requirement, disk=disk_requirement,
                               avoid_bad_hosts=args.avoid_bad_hosts, retry_arbitrary_failures=args.retry_arbitrary_failures)
        job.run()
        CondorJobWrapper.monitor([job], cleanup=args.condor_cleanup)
        synthesized_features, data, model = read_synthesized_inputs(job_dir)
    else:
        synthesized_features, data, model = synmod.master.pipeline(args)
    args.logger.info("End running synmod")
    return synthesized_features, data, model


def configure_synthesis_args(oargs):
    """Configure arguments for data/model synthesis"""
    args = copy.copy(oargs)
    args.output_dir = args.synthesis_dir
    args.write_outputs = True
    if args.analysis_type == constants.HIERARCHICAL:
        args.synthesis_type = constants.TABULAR
    else:
        args.synthesis_type = constants.TEMPORAL
    if args.noise_multiplier != constants.AUTO:
        try:
            args.noise_multiplier = float(args.noise_multiplier)
        except ValueError:
            print(f"Noise multiplier must be set to '{constants.AUTO}' or a non-negative float", file=sys.stderr)
            raise
    return args


def noise_selection(args, data, targets, model):
    """Select noise multiplier based on desired regressor performance if auto-selection invoked"""
    # pylint: disable = protected-access
    if args.noise_multiplier != constants.AUTO:
        return float(args.noise_multiplier)
    if args.model_type == REGRESSOR:
        targets_mean = np.mean(targets)
        sum_of_squares_total = sum((targets - targets_mean)**2)
        sum_of_residuals_scaled = sum((targets - model.predict(data, noise=1))**2)
        args.noise_multiplier = np.sqrt(sum_of_squares_total * (1 - constants.AUTO_R2) / sum_of_residuals_scaled)
        args.logger.info(f"Auto-selected noise multiplier {args.noise_multiplier} "
                         f"to yield R2 score {r2_score(targets, model.predict(data, noise=args.noise_multiplier))}")
    else:
        agg_data_t = model._aggregator.operate(data).transpose()
        targets = model.predict(data, labels=True) if args.loss_target_values != constants.LABELS else targets
        noise_multipliers = np.arange(0.01, 1, 0.01)
        accuracies = np.zeros(noise_multipliers.shape[0])
        for idx, noise_multiplier in enumerate(noise_multipliers):
            accuracies[idx] = sum(targets == (model._polynomial_fn(agg_data_t, noise_multiplier) - model._threshold > 0)) / args.num_instances
        acc_diff = np.abs(accuracies - constants.AUTO_R2)
        best_idx = np.argmin(acc_diff)
        args.noise_multiplier = noise_multipliers[best_idx]
        args.logger.info(f"Auto-selected noise multiplier {args.noise_multiplier} "
                         f"to yield accuracy {accuracies[best_idx]}")
    return args.noise_multiplier


def gen_hierarchy(args, clustering_data):
    """
    Generate hierarchy over features

    Args:
        args: Command-line arguments
        clustering_data: Data potentially used to cluster features
                         (depending on hierarchy generation method)

    Returns:
        hierarchy_root: root fo resulting hierarchy over features
    """
    # TODO: Get rid of possibly redundant hierarchy attributes e.g. vidx
    # Generate hierarchy
    hierarchy_root = None
    if args.hierarchy_type == constants.FLAT:
        args.contiguous_node_names = False  # Flat hierarchy should be automatically created; do not re-index hierarchy
    elif args.hierarchy_type == constants.CLUSTER_FROM_DATA:
        clusters = cluster_data(clustering_data)
        hierarchy_root = gen_hierarchy_from_clusters(args, clusters)
    elif args.hierarchy_type == constants.RANDOM:
        hierarchy_root = gen_random_hierarchy(args)
    else:
        raise NotImplementedError("Need valid hierarchy type")
    # Improve visualization - contiguous feature names
    feature_id_map = {}  # mapping from visual feature ids to original ids
    if args.contiguous_node_names:
        for idx, node in enumerate(anytree.PostOrderIter(hierarchy_root)):
            node.vidx = idx
            if node.is_leaf:
                node.min_child_vidx = idx
                node.max_child_vidx = idx
                node.num_base_features = 1
                node.name = str(idx)
                feature_id_map[idx] = node.idx[0]
            else:
                node.min_child_vidx = min([child.min_child_vidx for child in node.children])
                node.max_child_vidx = max([child.vidx for child in node.children])
                node.num_base_features = sum([child.num_base_features for child in node.children])
                node.name = f"[{node.min_child_vidx}-{node.max_child_vidx}] (size: {node.num_base_features})"
    return hierarchy_root, feature_id_map


def gen_random_hierarchy(args):
    """Generates balanced random hierarchy"""
    args.logger.info("Begin generating hierarchy")
    nodes = [anytree.Node(str(idx), idx=[idx]) for idx in range(args.num_features)]
    args.rng.shuffle(nodes)
    node_count = len(nodes)
    while len(nodes) > 1:
        parents = []
        for left_idx in range(0, len(nodes), 2):
            parent = anytree.Node(str(node_count))
            node_count += 1
            nodes[left_idx].parent = parent
            right_idx = left_idx + 1
            if right_idx < len(nodes):
                nodes[right_idx].parent = parent
            parents.append(parent)
        nodes = parents
    hierarchy_root = nodes[0]
    args.logger.info("End generating hierarchy")
    return hierarchy_root


def cluster_data(data):
    """Cluster data using hierarchical clustering with Hamming distance"""
    # Cluster data
    clusters = linkage(data.transpose(), metric="hamming", method="complete")
    return clusters


def gen_hierarchy_from_clusters(args, clusters):
    """
    Organize clusters into hierarchy

    Args:
        clusters: linkage matrix (num_features-1 X 4)
                  rows indicate successive clustering iterations
                  columns, respectively: 1st cluster index, 2nd cluster index, distance, sample count
    Returns:
        hierarchy_root: root of resulting hierarchy over features
    """
    # Generate hierarchy from clusters
    nodes = [anytree.Node(str(idx), idx=[idx]) for idx in range(args.num_features)]
    for idx, cluster in enumerate(clusters):
        cluster_idx = idx + args.num_features
        left_idx, right_idx, _, _ = cluster
        left_idx = int(left_idx)
        right_idx = int(right_idx)
        cluster_node = anytree.Node(str(cluster_idx))
        nodes[left_idx].parent = cluster_node
        nodes[right_idx].parent = cluster_node
        nodes.append(cluster_node)
    hierarchy_root = nodes[-1]
    return hierarchy_root


def update_hierarchy_descriptions(hierarchy_root, relevant_feature_map, features):
    """
    Add feature relevance information to nodes of hierarchy
    """
    relevant_features = set()
    for key in relevant_feature_map:
        relevant_features.update(key)
    for node in anytree.PostOrderIter(hierarchy_root):
        node.description = constants.IRRELEVANT
        if node.is_leaf:
            idx = node.idx[0]
            coeff = relevant_feature_map.get(frozenset([idx]))
            if coeff:
                node.description = f"{constants.RELEVANT} feature:\nPolynomial coefficient: {coeff}\nSummary: {features[idx].summary()}"
            elif idx in relevant_features:
                node.description = f"{constants.RELEVANT} feature\n(Interaction-only)\nSummary: {features[idx].summary()}"
        else:
            for child in node.children:
                if child.description != constants.IRRELEVANT:
                    node.description = constants.RELEVANT


def run_anamod(args, pass_args, data, model, targets, hierarchy=None):  # pylint: disable = too-many-arguments
    """Run analysis algorithms"""
    args.logger.info("Begin running anamod")
    # Add options
    options = {}
    options["feature_hierarchy"] = hierarchy
    options["output_dir"] = args.output_dir
    options["seed"] = args.seed
    options["visualize"] = args.visualize
    options["analysis_type"] = args.analysis_type
    options["condor"] = args.condor
    options["shared_filesystem"] = args.shared_filesystem
    options["features_per_worker"] = args.features_per_worker
    options["memory_requirement"] = 2 + (data.nbytes // (2 ** 30))
    options["disk_requirement"] = 3 + options["memory_requirement"]
    options["avoid_bad_hosts"] = args.avoid_bad_hosts
    options["retry_arbitrary_failures"] = args.retry_arbitrary_failures
    options["num_permutations"] = args.num_permutations
    options["cleanup"] = args.cleanup
    options["loss_function"] = constants.BINARY_CROSS_ENTROPY if args.model_type == CLASSIFIER else constants.QUADRATIC_LOSS
    if args.analysis_type == constants.HIERARCHICAL:
        options["perturbation"] = args.perturbation
        options["analyze_interactions"] = args.analyze_interactions
    args.logger.info(f"Passing the following arguments to anamod.master without parsing: {pass_args}")
    pass_args = process_pass_args(pass_args)
    options = {**pass_args, **options}  # Merge dictionaries
    # Create analyzer
    analyzer_class = ModelAnalyzer if args.analysis_type == constants.HIERARCHICAL else TemporalModelAnalyzer
    analyzer = analyzer_class(model, data, targets, **options)
    # Run analyzer
    args.logger.info(f"Analyzing model with options:\n{pprint.pformat(options)}")
    features = analyzer.analyze()
    cleanup(args, analyzer.data_filename, analyzer.model_filename)
    args.logger.info("End running anamod")
    return features


def process_pass_args(pass_args):
    """Process list of unrecognized arguments, to pass to anamod.master"""
    assert len(pass_args) % 2 == 0, f"Odd argument count in pass_args: {pass_args} ; is a value missing?"
    pass_args = {pass_args[idx].strip("-"): pass_args[idx + 1] for idx in range(0, len(pass_args), 2)}  # Make dict
    return pass_args


def cleanup(args, data_filename, model_filename):
    """Clean data and model files after completing simulation"""
    # TODO: clean up hierarchy file
    if not args.cleanup:
        return
    for filename in [data_filename, model_filename]:
        if filename is not None and os.path.exists(filename):
            os.remove(filename)


def write_summary(args, model_wrapper, results):
    """Write simulation summary"""
    model = model_wrapper.ground_truth_model
    config = dict(analysis_type=args.analysis_type,
                  num_instances=args.num_instances,
                  num_features=args.num_features,
                  sequence_length=args.sequence_length,
                  model_type=model.__class__.__name__,
                  num_permutations=args.num_permutations,
                  noise_multiplier=args.noise_multiplier,
                  sequences_independent_of_windows=args.window_independent)
    # pylint: disable = protected-access
    model_summary = {}
    if args.analysis_type == constants.TEMPORAL:
        model_summary["windows"] = [f"({window[0]}, {window[1]})" if window else None for window in model._aggregator._windows]
        model_summary["aggregation_fns"] = [agg_fn.__class__.__name__ for agg_fn in model._aggregator._aggregation_fns]
    model_summary["polynomial"] = model.sym_polynomial_fn.__repr__()
    summary = {constants.CONFIG: config, constants.MODEL: model_summary, constants.RESULTS: results}
    summary_filename = f"{args.output_dir}/{constants.SIMULATION_SUMMARY_FILENAME}"
    args.logger.info(f"Writing summary to {summary_filename}")
    with open(summary_filename, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    return summary


if __name__ == "__main__":
    main()
