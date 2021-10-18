#!/usr/bin/env python3
"""

Copyright 2021 Nokia

Licensed under the Apache License 2.0
SPDX-License-Identifier: Apache-2.0

"""

import os
import time
import sys
import argparse
from pathlib import Path
from typing import Union, List
from distutils.version import LooseVersion
from dataclasses import dataclass, field
from time import sleep

from robot.api import TestSuiteBuilder
from robot.run import RobotFramework
from robot import rebot
from robot.version import get_version, get_full_version
from robot.running.model import TestSuite, TestCase  # single executable test case or test suite
from robot.result.model import TestCase as ExecutedTestCase  # Represents results of a single test case

from glob import glob

import multiprocess as multiprocessing  # type: ignore # pip3 install multiprocess

MINIMUM_ROBOT_VERSION = "4.0"
TERMINAL_SIGNAL = None
MIN_CLUSTER_ID = 1
MIN_WORKER_ID = 1


def worker_process_tasks(task_queue, process_nr: int, out_dir: str) -> bool:
    """
    This is the main entry point for a worker process

    :param task_queue: Multiprocess.Queue: queue holding all test cases to be executed
    :param process_nr: int: process id of this given process
    :return: True always
    """
    print(f"[process {process_nr}]: PID={os.getpid()} PARENTPID={os.getppid()}")
    while True:
        tpl = task_queue.get()
        if tpl is None:
            break
        if not worker_run_single_tc(task_queue, process_nr, tpl):
            break

    print(f"[process {process_nr}]: exiting")
    return True


def worker_run_single_tc(task_queue, worker_id: int, tpl) -> bool:
    """
    Executes a single testcase from a given suite
    :param task_queue: instance of JoinableQueue
    :param worker_id: worker number
    :param tpl: tuple holding (suite, testcase)
    :return: False if termination signal was received, True otherwise
    """
    suite, test = tpl

    print(f"[worker {worker_id}]: entry point PID={os.getpid()} PARENTPID={os.getppid()}")
    print(f"[worker {worker_id}]: ***************************TC name: {test.name}")
    root_suite = get_root_suite(suite)
    root_suite.filter(included_tests=[test.name])  # get rid of TC from all suites
    add_execution_tag_to_all_tcs(suite, test, worker_id)  # Only tag this single TC in this suite

    listener_instance = PythonListener(worker_id=worker_id, task_queue=task_queue)
    root_suite.name = "AllSuites"
    variables = [
        f'worker_id:{worker_id}',
    ]

    # This is the starting point of robot execution in the parallel worker
    root_suite.run(variable=variables, loglevel='TRACE', output=f"output-worker-{worker_id}",
                   listener=listener_instance)  # run defined in src/robot/running/model.py

    # to close chrome drivers:
    sleep(3)

    if listener_instance.shutdown_worker:
        return False
    else:
        return True


def get_root_suite(suite: TestSuite) -> TestSuite:
    """
    Walk through recursively the suite's parent suite until the root suite is found
    :param suite:
    :return: root suite
    """
    if suite.parent:
        parent_suite = suite.parent
        while parent_suite.parent:
            parent_suite = parent_suite.parent
        return parent_suite
    else:
        return suite


def add_execution_tag_to_tc(testcase, worker_id: int) -> None:
    """
    Adds tags to executed test cases; useful in the final report
    :param testcase: TestCase
    :param worker_id: id of worker executing in this process
    :return: None
    """
    testcase.tags.add(f"run_by_worker{worker_id}")


def add_execution_tag_to_all_tcs(all_suites: TestSuite, testcase_to_run, worker_id: int) -> None:
    """
    Walks through all child suites and test cases and tags test cases; useful in the final report
    :param all_suites: TestSuite
    :param testcase_to_run: TestCase
    :param worker_id: id of worker executing in this process
    :return: None
    """
    if len(all_suites.suites) > 0:
        for suite in all_suites.suites:
            for test in suite.tests:
                if test.id == testcase_to_run.id:
                    add_execution_tag_to_tc(test, worker_id)
                    return
    else:
        suite = all_suites
        for test in suite.tests:
            if test.id == testcase_to_run.id:
                add_execution_tag_to_tc(test, worker_id)
                return


class PythonListener:
    """
    Class to handle Robot events during execution especially new_suite and end_testcase
    """
    ROBOT_LISTENER_API_VERSION = 3
    current_suite = None
    shutdown_worker = False

    def __init__(self, worker_id, task_queue, *args, **kwargs):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.ROBOT_LIBRARY_LISTENER = self

    def start_suite(self, suite: TestSuite, *args, **kwargs) -> None:
        """
        Called when robot starts the execution of a new suite
        :param suite: TestSuite
        :param args: ignored
        :param kwargs: ignored
        :return: None
        """
        # save current suite so that we can modify it later
        self.current_suite = suite

    def end_test(self, testcase: TestCase, executed_testcase_object: ExecutedTestCase, *args, **kwargs) -> None:
        """
        Called when robot finishes the execution of a test case
        :param testcase: robot.running.model.TestCase (without execution data)
        :param executed_testcase_object: robot.result.model.TestCase (with execution data)
        :param args: ignored
        :param kwargs: ignored
        :return: None
        """
        # Mark this job as done
        self.task_queue.task_done()

        if getattr(testcase, "name"):
            print(f"\n[worker {self.worker_id}]: ***************************end_test {testcase.name}: {executed_testcase_object.status} ")
        else:
            print(f"\n[worker {self.worker_id}]: ***************************end_test {testcase}: {executed_testcase_object.status} ")

        # Get a new job
        tpl = self.task_queue.get()
        if tpl is None:
            self.shutdown_worker = True
            return

        suite, test = tpl
        print(f"\n[worker {self.worker_id}]: *************************** RECEIVED NEW TC name: {test.name} from suite: {suite.name}")
        if suite.name == self.current_suite.name or get_root_suite(self.current_suite) == self.current_suite:  # type: ignore
            # this is the same suite, only add TC
            self.current_suite.tests.append(test)  # type: ignore
        else:
            suite.filter(included_tests=[test.name])
            get_root_suite(self.current_suite).suites.append(suite)  # type: ignore

        add_execution_tag_to_tc(test, worker_id=self.worker_id)


############################################################################
############################################################################
############################################################################
############################################################################

class OrderedSet(list):
    """
    Implents a deduplicated list that has only unique elements
    """

    def add(self, element) -> None:
        """
        Add an element to the OrderedSet
        Add action will be skipped if the OrderedSet has already such an element
        :param element: any
        :return:
        """
        if element not in self:
            self.append(element)


@dataclass
class ConfigData:
    """
    Data class storing the configuration needs of a test case
    Each field denotes a configuration attribute except for fields starting with underscore

    NOTE: each field should have a corresponding set_{fieldname} function eg. param1 -> set_param1
    """
    param1: str = ""
    param2: str = ""
    param3: str = ""
    param4: str = ""
    _freedom: int = -1
    _cluster: int = -1
    _config_fields: List[str] = field(default_factory=list)

    def get_all_config_fields(self) -> list:
        """
        Returns all configuration fields that are not private (no starting underscore in the name)
        :return: list of field names
        """
        fields = [item for item in self.__dict__ if not item.startswith('_')]
        return fields

    def calculate_freedom(self) -> None:
        """
        Calculates the freedom ie. how many config parameters have a don't care (== "") value
        and stores this value in self._freedom
        :return: None
        """
        self._config_fields = self.get_all_config_fields()
        freedom = 0
        for item in self._config_fields:
            if self.__dict__[item] == "":
                freedom += 1
        self._freedom = freedom

    def is_compatible_config(self, another_config) -> bool:
        """
        Compares current ConfigData item with another ConfigData item given in the parameter
        Returns true if they match or their don't care values make them compatible
        :param another_config: ConfigData
        :return: bool
        """
        self._config_fields = self.get_all_config_fields()
        identical = 0
        for item in self._config_fields:
            if self.__dict__[item] == "" or another_config.__dict__[item] == "" or self.__dict__[item] == another_config.__dict__[item]:
                # this config field is identical
                identical += 1
        return len(self._config_fields) == identical

    def render_robot_config_tc(self, suite):
        test_timeout = int((len(self._config_fields) - self._freedom) * 1.25)
        if test_timeout < 1:
            test_timeout = 1
        test_case = suite.tests.create(f"Configuration for cluster {self._cluster}", timeout=f'{test_timeout}m')

        for item in self._config_fields:
            if getattr(self, item) != "":
                test_case.body.create_keyword(name=f'Set {item}', args=[getattr(self, item)])
        # test_case.body is never None in Robot 4.0+, it is an empty Body object with len 0 like Body(item_class=BodyItem, items=[])
        if len(test_case.body) == 0:
            # Do this only if really there is nothing the config cluster should do
            # Actually this should never happen, only in very rare circumstances e.g. like all test cases have "don't care" for all config possibilities
            test_case.body.create_keyword(name='Pass Execution', args=['Config executes no keywords'])

    def __repr__(self):
        out_str = f"ConfigData(_freedom={self._freedom}, _cluster={self._cluster}, "

        for key in self._config_fields:
            if getattr(self, key) != "":
                out_str += f"{key}={getattr(self, key)}, "
        out_str += ")"
        return out_str


class ConfigDataList:
    """
    Data class storing list of ConfigData entries
    """
    configs: List[ConfigData]
    max_assigned_cluster: int = -1

    def __init__(self, configs=None):
        self.configs = []
        if configs is not None:
            for config_dict in configs:
                conf_data = ConfigData(**config_dict)
                conf_data.calculate_freedom()
                if conf_data not in self.configs:
                    self.configs.append(conf_data)

    def add_config(self, config_dict: dict) -> int:
        """
        Adds a new ConfigData entry to this ConfigDataList from the input dictionary (config_dict)

        :param config_dict: dict: key-value pairs of configuration parameters
        :return: ConfigData to be stored to the TestCase object
        """
        conf_data = ConfigData(**config_dict)
        conf_data.calculate_freedom()
        if conf_data not in self.configs:
            self.configs.append(conf_data)

        return self.configs.index(conf_data)

    def get_configs_in_cluster(self, cluster: int) -> list:
        """
        Returns a list of ConfigData items that belong to the cluster given as parameter
        :param cluster: int: id of cluster for which to return all belonging ConfigData items
        :return: list of ConfigData
        """
        return [config for config in self.configs if config._cluster == cluster]

    def determine_clusters(self):
        """
        Determines the config clusters which contain compatible ConfigData items
        :return: None
        """
        max_assigned_cluster = -1
        for config in sorted(self.configs, key=lambda config_item: config_item._freedom, reverse=True):
            if max_assigned_cluster == -1:
                max_assigned_cluster = MIN_CLUSTER_ID
                config._cluster = max_assigned_cluster
            else:
                for cluster in range(MIN_CLUSTER_ID, max_assigned_cluster + 1):
                    not_compatible = 0
                    existing_configs_in_cluster = self.get_configs_in_cluster(cluster)
                    for a in existing_configs_in_cluster:
                        if not a.is_compatible_config(config):
                            not_compatible += 1
                            break
                    else:
                        # full compatible
                        config._cluster = cluster
                        break

                else:
                    if config._cluster == -1:
                        max_assigned_cluster += 1
                        # there was no break
                        config._cluster = max_assigned_cluster

        self.max_assigned_cluster = max_assigned_cluster

    def get_most_restrictive_config_in_cluster(self, cluster: int) -> ConfigData:
        """
        Returns a ConfigData item that represents this cluster ie. contains all config parameters that need to be set
        :param cluster: int: id of cluster for which to return the representative ConfigData item
        :return: ConfigData
        """
        configs = self.get_configs_in_cluster(cluster)
        representative_config = ConfigData()
        if len(configs) == 0:
            return representative_config  # in this case it is empty, nothing is set
        for _field in configs[0]._config_fields:
            for conf in configs:
                if conf.__dict__[_field]:
                    representative_config.__dict__[_field] = conf.__dict__[_field]
                    break
        representative_config._cluster = cluster
        representative_config.calculate_freedom()
        return representative_config


class ParallelRunner:
    """
    Implements the parallel runner logic:
    - creating config clusters
    - setting them up
    - and running corresponding test cases in parallel
    """
    dir_of_parallel_runner = Path(__file__).parent.absolute()

    def __init__(self, suites_directory: str = "./", worker_processes: int = 4):
        self.robot_args = {}
        self.out_dir = ''
        self.suites = None
        self.WORKER_PROCESSES = worker_processes
        self.suites_directory = suites_directory
        self.task_queue = multiprocessing.JoinableQueue()
        self.processes = []

    def main(self, robot_args: Union[dict, None] = None) -> None:
        """
        Main entry point, call this to do the parallel execution
        :param robot_args: dict of robot cli arguments
        :return: None
        """
        if robot_args is not None:
            self.robot_args = robot_args

        self.parse_robot_files_and_print()
        self.allocate_configs_to_clusters()
        self.execute_all_clusters()
        self.finalize()

    def collect_testfiles(self, subdir: str) -> list:
        """
        Collects all robot test files in the current and child directories

        :param subdir: str: the directory from which to start the search relative to the parallel_runner.py file location
        :return: sorted list of robot file names
        """
        test_files = []
        for (dir_name, _, files) in os.walk(self.dir_of_parallel_runner / subdir):
            test_files.extend(
                [os.path.join(dir_name, f) for f in files if (f.endswith('.robot'))]
            )

        return sorted(test_files)

    def parse_suite_for_configs(self, suite: TestSuite) -> None:
        """
        Walk through the given TestSuite and parse the configuration tags of each test case to ConfigData objects
        :param suite:
        :return: None
        """
        print(f"[*] Testsuite: {suite.name}")
        print(f"    - Setup: {suite.setup}")
        print(f"    - Teardown: {suite.teardown}")

        for test in suite.tests:
            print(f'    - Testcase: {test.name}')
            print(f'      Tags: {test.tags}')
            config_dict = {}
            for tag in test.tags:
                if "config-" in tag and "=" in tag:
                    key, value = tag.split("=", 1)
                    config_dict[key.replace('config-', '')] = value
            configdata_idx = self.test_configs.add_config(config_dict)
            test.tags.add(f"configdata={configdata_idx}")

    def parse_robot_files_and_print(self) -> None:
        """
        Loads robot files and generates ConfigData objects attached to the test cases and added to self.test_configs
        self.all_suites: TestSuite will be the suite containing all test cases
        """
        self.test_configs = ConfigDataList()
        tc_files = self.collect_testfiles(self.suites_directory)

        print("[*] Robot files found:")
        for item in tc_files:
            print(f"    - {item}")
        print("")

        self.all_suites = TestSuiteBuilder().build(*tc_files)
        self.all_suites.filter(
            included_suites=self.robot_args.pop('suite', None),
            included_tests=self.robot_args.pop('test', None),
            included_tags=self.robot_args.pop('include', None),
            excluded_tags=self.robot_args.pop('exclude', None)
        )

        print("[*] Parsing all test suites")

        for suite in self.all_suites.suites:  # this contains all suites if more than one suite was found and loaded
            self.parse_suite_for_configs(suite)
        self.parse_suite_for_configs(self.all_suites)  # this contains the suite if only one suite was found and loaded

    def allocate_configs_to_clusters(self) -> None:
        """
        Implements the logic that groups different ConfigData items into clusters
        The clusters contain such ConfigData elements that are compatible with each other
        :return: None
        """
        self.suites = OrderedSet(suite.name for suite in self.all_suites.suites if suite.tests)
        self.suites.add(self.all_suites.name)

        print("")
        print("SUITES:")
        for item in self.suites:
            print(f" * {item}")
        print("")

        self.test_configs.determine_clusters()
        print("----------------------------")
        print("[*] Assigning testcase configs to clusters")
        for cluster in range(MIN_CLUSTER_ID, self.test_configs.max_assigned_cluster + 1):
            print(f"    - Config cluster {cluster}")
            for item in self.test_configs.get_configs_in_cluster(cluster):
                print(f"      * {repr(item)}")
            print("")

    def put_tcs_into_queue_from_a_suite(self, suite: TestSuite, cluster: int, is_passed: bool) -> None:
        """
        Puts the test cases of suite belonging to cluster (id) to the task_queue
        :param suite: TestSuite
        :param cluster: id of cluster
        :param is_passed: test status
        :return: None
        """
        for test in suite.tests:
            config_idx = -1
            tag = ""
            for tag in test.tags:
                if "configdata" in tag:
                    key, value = tag.split("=")
                    config_idx = int(value)
                    break
            # The test belonging to an already executed cluster do not have anymore the configdata tag, so config_idx = -1
            if config_idx >= 0 and self.test_configs.configs[config_idx]._cluster == cluster:
                print(f"    - Testcase: {test.name}")
                print(f'      Config: {self.test_configs.configs[config_idx]}')
                test.tags.add(f"CONFIG_CLUSTER={cluster}")
                test.tags.remove(tag)  # get rid of configdata=... tag
                self.task_queue.put((suite, test))

    def put_tcs_into_queue(self, cluster: int, is_passed: bool) -> None:
        """
        Walk through all suites that were loaded and put the test cases to task_queue that belong to cluster (id)
        :param cluster: id of cluster
        :param is_passed: cluster config result: passed/failed
        :return: None
        """
        for suite in self.all_suites.suites:
            self.put_tcs_into_queue_from_a_suite(suite, cluster, is_passed)

        self.put_tcs_into_queue_from_a_suite(self.all_suites, cluster, is_passed)

    def setup_parallel_workers(self) -> None:
        """
        Starts the parallel worker processes
        :return: None
        """
        print("==============================================================================")
        print("[*] Starting parallel workers")

        print(f'Running with {self.WORKER_PROCESSES} processes!')
        for n in range(MIN_WORKER_ID, self.WORKER_PROCESSES + MIN_WORKER_ID):
            p = multiprocessing.Process(
                target=worker_process_tasks, args=(self.task_queue, n, self.out_dir), daemon=True)
            self.processes.append(p)
            p.start()
        print("Processes started")

    def execute_all_clusters(self) -> None:
        """
        Sets up parallel worker processes, walks through all clusters and executes all config and real test cases
        After that worker processes are terminated
        :return: None
        """
        self.setup_parallel_workers()

        print("==============================================================================")
        print("[*] Execution of config clusters")
        self.start_all = time.time()

        for cluster in range(MIN_CLUSTER_ID, self.test_configs.max_assigned_cluster + 1):
            self.execute_cluster(cluster)
        self.terminate_parallel_workers()

    def merge_outputs(self) -> None:
        """
        Merges reports from workers into one single report using rebot
        :return: None
        """
        output_workers = glob('output-worker-*.xml')
        output_configs = glob('output-config_suite-cluster_*.xml')

        # These config cluster xmls do not have a common root suite, so we merge them under AllSuites
        for output_config in output_configs:
            rebot(*output_configs, merge=False, name='AllSuites', output="output-config_suite-all.xml", log="NONE")

        # All other test suites should have already the AllSuites as their common root suite
        if os.path.isfile("output-config_suite-all.xml"):
            # Config logs should be the first items
            outputs = ["output-config_suite-all.xml"] + output_workers
        else:
            # There is no such file if we skip the config suites
            outputs = output_workers

        if os.path.isfile("output_seq_ft.xml"):
            # add the first FT run result to merge it
            print("Adding first FT run xml to the file list for merge.")
            outputs += ["output_seq_ft.xml"]

        rebot(*outputs, output='output', log='log', name='AllSuites', merge=True)  # This is the final report containing all executed TCs and suites
        if os.path.isfile('log.html'):
            output_xml_files = glob('output*.xml')
            for out_xml in output_xml_files:
                if out_xml != 'output.xml':
                    os.remove(out_xml)

    def terminate_parallel_workers(self) -> None:
        """
        Enqueues the termination signal (None) tp the task queue for all processes
        and waits until processes exit after processing the termination signal
        :return: None
        """
        print("[*] Terminating parallel workers")
        for n in range(MIN_WORKER_ID, self.WORKER_PROCESSES + MIN_WORKER_ID):
            self.task_queue.put(TERMINAL_SIGNAL)
        for p in self.processes:
            p.join()

    def create_dynamic_testcase_for_config(self, cluster: int) -> TestSuite:
        """
        Generates a config suite based on ./config_suite/config_suite.robot
        and adds test cases for each config field that needs to be set
        :param cluster: id of cluster
        :return: generated TestSuite object
        """
        representative_config_of_cluster = self.test_configs.get_most_restrictive_config_in_cluster(cluster)

        config_suite = TestSuiteBuilder().build(self.dir_of_parallel_runner / "config_suite/config_suite.robot")
        config_suite.name = f"Config Suite for cluster {cluster}"
        config_suite.resource.variables.create("${NUMBER_OF_THREADS}", (str(self.WORKER_PROCESSES),))
        config_suite.filter(excluded_tags=["remove"])
        representative_config_of_cluster.render_robot_config_tc(config_suite)
        return config_suite

    def execute_cluster(self, cluster: int) -> None:
        """
        Execute a given cluster:
        - creates the config suite
        - executes the config suite
        - enqueues the test cases belonging to the given cluster
        - waits for completion
        :param cluster: id of cluster to execute
        :return: None
        """
        print("==============================================================================")
        print("==============================================================================")
        print("==============================================================================")
        print(f"  Config nr. {cluster}")
        print("------------------------------------------------------------------------------")
        start = time.time()

        config_suite = self.create_dynamic_testcase_for_config(cluster)
        variables: List[str] = self.robot_args.get('variable', [])

        print(f"[*] Execute config suite for cluster {cluster}")
        result = config_suite.run(variable=variables,
                                  loglevel="TRACE",
                                  output=f"output-config_suite-cluster_{cluster}",
                                  currdir=self.suites_directory)

        print("***", result.suite.status)

        execution_time = time.time() - start
        print(f'Time taken for config = {execution_time:.10f}')
        self.suites.add(config_suite.name)
        config_suite_passed = result.suite.passed
        print("------------------------------------------------------------------------------")
        print(f"[*] Enqueuing testcases of cluster {cluster} for execution:")
        self.put_tcs_into_queue(cluster, config_suite_passed)
        self.task_queue.join()
        print("Tasks done, processes will be reused")
        execution_time = time.time() - start
        print(f'Time taken = {execution_time:.10f}')
        print("==============================================================================")
        print("==============================================================================")

    def finalize(self) -> None:
        """
        Prints total execution time and merges reports from workers into one single report
        :return: None
        """
        total_execution_time = time.time() - self.start_all
        print(f"TOTAL EXECUTION TIME: {total_execution_time:.10f} sec")
        self.merge_outputs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="poc_parallel_runner.py",
                                     description="Parallel runner for Robot test cases")
    parser.add_argument('--parallel-workers', help="Numer of worker processes that run test cases in parallel", metavar="<INT>", default=4, required=False)
    parser.add_argument('--suites-dir', help="Folder where to look for robot files", metavar="<DIR>", default="./suites", required=False)

    args, probably_robot_args = parser.parse_known_args()
    print(f"probably_robot_args = {probably_robot_args}")
    cli_args = sys.argv[1:]
    robot_args_from_command_line, _ = RobotFramework().parse_arguments(probably_robot_args + ['dummy.file.name'])

    print(f"Robot version: {get_full_version()}")
    print(f"parallel-workers: {args.parallel_workers}")
    print(f"suites-dir: {args.suites_dir}")
    print("ROBOT ARGS:", robot_args_from_command_line)

    if LooseVersion(MINIMUM_ROBOT_VERSION) > LooseVersion(get_version()):
        print("")
        print(f"ERROR: This parallel_runner.py implementation requires Robot framework v{MINIMUM_ROBOT_VERSION} or later")
        print("")
        sys.exit(2)

    ParallelRunner(suites_directory=args.suites_dir, worker_processes=int(args.parallel_workers)).main(robot_args_from_command_line)
