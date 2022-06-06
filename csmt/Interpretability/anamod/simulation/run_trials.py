"""Run multiple trials of multiple simulations and aggregate results"""

import argparse
from collections import namedtuple, OrderedDict
from copy import copy
import configparser
from distutils.util import strtobool
import json
import os
import resource
import subprocess
import time

import numpy as np
from anamod.core import constants, utils

TestParam = namedtuple("TestParameter", ["key", "values"])


class Simulation():
    """Simulation helper class"""
    # pylint: disable = too-few-public-methods
    def __init__(self, cmd, output_dir, param, popen=None):
        self.cmd = cmd
        self.output_dir = output_dir
        self.param = param
        self.popen = popen
        self.complete = False
        self.attempt = 1

    def __hash__(self):
        return hash(self.cmd)


class Trial():  # pylint: disable = too-many-instance-attributes
    """Class that parametrizes, runs, monitors, and analyzes a group of simulations"""
    # pylint: disable = too-many-arguments, attribute-defined-outside-init
    def __init__(self, seed, sim_type, analysis_type, output_dir, summarize_only,
                 reevaluate, max_simulation_count, pass_args):
        self.seed = seed
        self.type = sim_type
        self.analysis_type = analysis_type
        self.output_dir = output_dir
        self.summarize_only = summarize_only
        self.reevaluate = reevaluate
        self.max_simulation_count = max_simulation_count
        self.pass_args = pass_args
        self.setup_simulations()

    def __hash__(self):
        return hash(self.seed)

    def setup_simulations(self):
        """Set up simulations"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = utils.get_logger(__name__, f"{self.output_dir}/run_simulations.log")
        self.config_filename = constants.CONFIG_HIERARCHICAL if self.analysis_type == constants.HIERARCHICAL else constants.CONFIG_TEMPORAL
        self.logger.info(f"Trial for seed {self.seed}: Begin running/analyzing simulations")
        self.config, self.test_param = self.load_config()
        self.synthesis_sim = None  # Simulation that synthesizes data/model, before performing analysis for different parameter values
        self.simulations = self.parametrize_simulations()
        self.error = False  # Flag to indicate if any simulation failed
        self.running_sims = set()  # Set of simulations running concurrently

    def load_config(self):
        """Load simulation configuration"""
        config_filename = f"{os.path.dirname(__file__)}/{self.config_filename}"
        assert os.path.exists(config_filename)
        config = configparser.ConfigParser()
        config.read(config_filename)
        assert self.type in config, f"Type not understood; choose one of {list(config.keys())}"
        dconfig = config[self.type]
        sconfig = ""
        test_param = None
        for option, value in dconfig.items():
            if "," in value:
                test_param = TestParam(option, [item.strip() for item in value.split(",")])
                continue
            sconfig += f"-{option} {value} "
        return sconfig, test_param

    def parametrize_simulations(self):
        """Parametrize simulations"""
        sims = []
        test_param = self.test_param
        if test_param is None:
            test_param = TestParam("", [constants.DEFAULT])
        key, values = test_param
        base_cmd = f"python -m anamod.simulation.simulation {self.config} -seed {self.seed} -evaluate_only {self.reevaluate} {self.pass_args}"
        synthesis_dir = ""
        if key == "num_instances" and not self.reevaluate:
            # Parameter corresponds to number of instances;
            # Fix model and generate data using largest instance count, then use this model for all values of the parameter
            max_instance_count = max([int(value) for value in values])
            synthesis_dir = f"{self.output_dir}/synthesis"
            cmd = base_cmd + f" -num_instances {max_instance_count} -output_dir {synthesis_dir} -synthesis_dir {synthesis_dir} -synthesize_only 1"
            self.synthesis_sim = Simulation(cmd, dir, f"{max_instance_count}-synthesis")
            if os.path.isfile(f"{synthesis_dir}/{constants.SIMULATION_SUMMARY_FILENAME}"):
                self.synthesis_sim.complete = True
        for value in values:
            output_dir = f"{self.output_dir}/{self.type}_{value}"
            cmd = base_cmd + f" -output_dir {output_dir}"
            cmd += f" -{key} {value}" if key else ""
            if synthesis_dir:
                cmd += f" -synthesis_dir {synthesis_dir} -analyze_only 1"
            sims.append(Simulation(cmd, output_dir, value))
            if os.path.isfile(f"{output_dir}/{constants.SIMULATION_SUMMARY_FILENAME}"):
                sims[-1].complete = True
        return sims

    def run_simulations(self):
        """Runs simulations in parallel"""
        if self.summarize_only and not self.reevaluate:
            return  # Skip running the simulations and proceed to analysis (assuming the results are already generated)
        if self.synthesis_sim and not self.synthesis_sim.complete:
            self.logger.info(f"Running synthesis simulation: '{self.synthesis_sim.cmd}'")
            self.synthesis_sim.popen = subprocess.Popen(self.synthesis_sim.cmd, shell=True)  # pylint: disable = consider-using-with
            self.running_sims.add(self.synthesis_sim)
        else:
            for sim in self.simulations:
                # TODO: Write this in a log file inside the trial directory instead of global log
                if self.reevaluate:
                    # Run serially
                    subprocess.run(sim.cmd, shell=True, check=True)
                elif not sim.complete:
                    self.logger.info(f"Running simulation: '{sim.cmd}'")
                    sim.popen = subprocess.Popen(sim.cmd, shell=True)  # pylint: disable = consider-using-with
                    self.running_sims.add(sim)

    def monitor_simulations(self, concurrent_simulation_count):
        """Monitor simulation progress and returns completion status"""
        sim_count = len(self.running_sims)
        for sim in copy(self.running_sims):
            returncode = sim.popen.poll()
            if returncode is not None:
                if returncode != 0:
                    assert not os.path.isfile(f"{sim.output_dir}/{constants.SIMULATION_SUMMARY_FILENAME}")
                    if sim.attempt < constants.MAX_ATTEMPTS:
                        sim.attempt += 1
                        self.logger.warning(f"Simulation {sim.cmd} failed; re-attempting ({sim.attempt} of {constants.MAX_ATTEMPTS})")
                        sim.popen = subprocess.Popen(sim.cmd, shell=True)  # pylint: disable = consider-using-with
                        continue
                    self.logger.error(f"Simulation {sim.cmd} failed; reached max attempts ({constants.MAX_ATTEMPTS}); see logs in {sim.output_dir}")
                    self.error = True
                self.running_sims.remove(sim)
                sim_count -= 1
        if not self.running_sims:
            if self.error:
                self.logger.error(f"run_simulations.py failed, see log at {self.output_dir}")
                return True, 0
            if self.synthesis_sim and not self.synthesis_sim.complete:
                if concurrent_simulation_count < self.max_simulation_count:
                    self.synthesis_sim.complete = True
                    self.run_simulations()
                    sim_count = len(self.running_sims)
            else:
                self.analyze_simulations()
                return True, 0
        return False, sim_count

    def analyze_simulations(self):
        """Collate and analyze simulation results"""
        results_filename = f"{self.output_dir}/{constants.ALL_SIMULATIONS_SUMMARY}_{self.type}.json"
        with open(results_filename, "w", encoding="utf-8") as results_file:
            root = OrderedDict()
            for sim in self.simulations:
                sim_results_filename = f"{sim.output_dir}/{constants.SIMULATION_SUMMARY_FILENAME}"
                with open(sim_results_filename, "r", encoding="utf-8") as sim_results_file:
                    data = json.load(sim_results_file)
                    data[constants.CONFIG]["output_dir"] = f"{os.path.basename(self.output_dir)}/{os.path.basename(sim.output_dir)}"
                    root[sim.param] = data
            json.dump(root, results_file, indent=2)
        self.logger.info(f"Trial for seed {self.seed}: End running/analyzing simulations")


def main(strargs=""):
    """Main"""
    args = parse_arguments(strargs)
    return pipeline(args)


def parse_arguments(strargs):
    """Parse arguments from input string or command-line"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-num_trials", type=int, default=3, help="number of trials to perform and average results over")
    parser.add_argument("-max_concurrent_simulations", type=int, default=4, help="number of simulations to run concurrently"
                        " (only increase if running on htcondor)")
    parser.add_argument("-trial_wait_period", type=int, default=60, help="time in seconds to wait before checking trial status")
    parser.add_argument("-start_seed", type=int, default=100000, help="randomization seed for first trial, incremented for"
                        " every subsequent trial.")
    parser.add_argument("-type", default=constants.DEFAULT, help="type of parameter to vary across simulations")
    parser.add_argument("-analysis_type", default=constants.TEMPORAL, choices=[constants.TEMPORAL, constants.HIERARCHICAL],
                        help="type of analysis to perform")
    parser.add_argument("-summarize_only", type=strtobool, default=False,
                        help="attempt to summarize results assuming they're already generated")
    parser.add_argument("-reevaluate", type=strtobool, default=False,
                        help="reevaluate metrics assuming results are already generated")
    parser.add_argument("-output_dir", required=True)
    args, pass_arglist = parser.parse_known_args(strargs.split(" ")) if strargs else parser.parse_known_args()
    args.pass_args = " ".join(pass_arglist)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.reevaluate:
        args.trial_wait_period = 0  # Serial
    args.logger = utils.get_logger(__name__, f"{args.output_dir}/run_trials.log")
    return args


def pipeline(args):
    """Pipeline"""
    args.logger.info(f"Begin running trials with config: {args}")
    setup(args)
    trials = gen_trials(args)
    run_trials(args, trials)
    return summarize_trials(args, trials)


def setup(args):
    """
    Setup: limit the number of threads used to load numpy libraries, since each subprocess will load its own
    Avoids this error: https://stackoverflow.com/questions/51256738/multiple-instances-of-python-running-simultaneously-limited-to-35
    """
    process_limit, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    process_limit -= (args.max_concurrent_simulations + 100)  # buffer
    assert args.max_concurrent_simulations < process_limit
    num_threads = process_limit // args.max_concurrent_simulations
    num_threads = min(max(1, num_threads), os.cpu_count())
    os.environ['OPENBLAS_NUM_THREADS'] = f"{num_threads}"


def gen_trials(args):
    """Generate multiple trials, each with multiple simulations"""
    trials = set()
    for seed in range(args.start_seed, args.start_seed + args.num_trials):
        output_dir = f"{args.output_dir}/trial_{args.type}_{seed}"
        trials.add(Trial(seed, args.type, args.analysis_type, output_dir, args.summarize_only,
                         args.reevaluate, args.max_concurrent_simulations, args.pass_args))
    return trials


def run_trials(args, trials):
    """Run multiple trials, each with multiple simulations"""
    # TODO: Maybe time trials as well, both actual and CPU time (which should exclude condor delays)
    running_trials = set()  # Trials currently running
    waiting_trials = copy(trials)  # Trials that haven't yet started
    unfinished_trials = copy(trials)  # Trials that haven't finished for whatever reason
    while unfinished_trials:
        concurrent_simulation_count = sum([len(trial.running_sims) for trial in running_trials])
        # Monitor running trials and determine available slots
        for trial in copy(running_trials):
            concurrent_simulation_count -= len(trial.running_sims)
            completed, sim_count = trial.monitor_simulations(concurrent_simulation_count)
            concurrent_simulation_count += sim_count
            if completed:
                running_trials.remove(trial)
                unfinished_trials.remove(trial)
        # Run trials if slots available
        for trial in copy(waiting_trials):
            if concurrent_simulation_count < args.max_concurrent_simulations:
                trial.run_simulations()
                waiting_trials.remove(trial)
                running_trials.add(trial)
                concurrent_simulation_count += len(trial.running_sims)
        if not unfinished_trials:
            break  # All trials completed, don't need to wait
        time.sleep(args.trial_wait_period)  # Wait before resuming monitoring/launching trials
    # Report errors for failed trials
    error = False
    for trial in trials:
        if trial.error:
            args.logger.error(f"Trial with seed {trial.seed} failed; see logs in {trial.output_dir}")
            error = True
    assert not error, f"run_trials.py failed, see log at {args.output_dir}"


def summarize_trials(args, trials):
    """
    Summarize outputs from trials.

    Returns dict containing
    (i) trials configuration, and
    (ii) for every simulation output variable (e.g. FDR), a mapping from test parameter values (e.g. instance counts)
         to a list of values that the variable took in the outputs of simulations with that test configuration
    and writes this dict as a JSON file
    """
    # pylint: disable = too-many-locals
    if not trials:
        return None
    some_trial = next(iter(trials))
    data = dict(run_trials_config=dict(num_trials=args.num_trials, type=args.type, test_values=some_trial.test_param.values,
                                       analysis_type=args.analysis_type, start_seed=args.start_seed, simulation_cmdline=some_trial.config))
    for tidx, trial in enumerate(trials):
        trial_results_filename = f"{trial.output_dir}/{constants.ALL_SIMULATIONS_SUMMARY}_{args.type}.json"
        with open(trial_results_filename, "r", encoding="utf-8") as trial_results_file:
            sim_data = json.load(trial_results_file, object_pairs_hook=OrderedDict)
            for param, sim in sim_data.items():  # Mapping from parameter values to corresponding simulation outputs
                for category, category_data in sim.items():  # Mapping from simulation output category (config/model/results) to its data
                    if category not in data:
                        data[category] = OrderedDict()
                    for key, value in category_data.items():  # Mapping from category-specific variable names to values
                        if key not in data[category]:
                            data[category][key] = OrderedDict()
                        if param not in data[category][key]:
                            data[category][key][param] = [None] * args.num_trials
                        # FIXME: Want to plot window accuracy by overlap (instead of average window overlap)
                        if key == constants.WINDOW_OVERLAP:
                            value = np.mean(list(value.values())) if value else 0.
                        data[category][key][param][tidx] = value
    data_filename = f"{args.output_dir}/{constants.ALL_TRIALS_SUMMARY_FILENAME}"
    with open(data_filename, "w", encoding="utf-8") as data_file:
        json.dump(data, data_file, indent=2)
    return data


if __name__ == "__main__":
    main()
