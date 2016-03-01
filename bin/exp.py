#!/usr/bin/env python

__copyright__ = "Copyright 2015, http://radical.rutgers.edu"
__license__   = "MIT"

import os

os.environ['RADICAL_SAGA_LOG_TGT'] = 'exp_saga.log'
os.environ['RADICAL_PILOT_LOG_TGT'] = 'exp_rp.log'

import sys
import time
import radical.pilot as rp
import random
import pprint
import inspect
import argparse
import tempfile

import radical.utils as ru
report = ru.LogReporter(name='radical.pilot')

# The agent wait until all units have been submitted
BARRIER_AGENT_LAUNCH= 'barrier_agent_launch'
# Wait until the agent has launched before we submit units
BARRIER_CLIENT_SUBMIT='barrier_client_submit'
# Submission happens in bulks of generation
BARRIER_GENERATION='generation'

# Whether and how to install new RP remotely
RP_VERSION = "local" # debug, installed, local
VIRTENV_MODE = "create" # create, use, update

# Schedule CUs directly to a Pilot, assumes single Pilot
SCHEDULER = rp.SCHED_DIRECT_SUBMISSION

resource_config = {
    #
    # XE nodes have 2 "Interlagos" Processors with 8 "Bulldozer" cores each.
    # Every "Bulldozer" core consists of 2 schedualable integer cores.
    # XE nodes therefore have a PPN=32.
    #
    # Depending on the type of application,
    # one generally chooses to have 16 or 32 instances per node.

    'LOCAL': {
        'RESOURCE': 'local.localhost',
        'TASK_LAUNCH_METHOD': 'FORK',
        'AGENT_SPAWNER': 'POPEN',
        'TARGET': 'local',
        'PPN': 8
    },
    'BW_APRUN': {
        'TARGET': 'local',
        'RESOURCE': 'ncsa.bw_aprun',
        'TASK_LAUNCH_METHOD': 'APRUN',
        'PROJECT': 'gkd',
        'AGENT_SPAWNER': 'POPEN',
        'QUEUE': 'normal', # Maximum 30 minutes
        'PPN': 32
    },
    'BW_CCM': {
        'TARGET': 'local',
        'RESOURCE': 'ncsa.bw_ccm',
        #'TASK_LAUNCH_METHOD': 'SSH',
        'TASK_LAUNCH_METHOD': 'MPIRUN',
        'AGENT_SPAWNER': 'POPEN',
        'QUEUE': 'normal', # Maximum 30 minutes
        'PROJECT': 'gkd',
        'PPN': 32,
        'PRE_EXEC_PREPEND': [
            'module use --append /u/sciteam/marksant/privatemodules',
            'module load use.own',
            'module load openmpi/1.8.4_ccm'
        ]
    },
    'BW_ORTELIB': {
        'RESOURCE': 'ncsa.bw_lib',
        'NETWORK_INTERFACE': 'ipogif0',
        'TASK_LAUNCH_METHOD': "ORTE_LIB",
        'MPI_LAUNCH_METHOD': "ORTE_LIB",
        'TARGET': 'node',
        'AGENT_SPAWNER': 'ORTE',
        'PROJECT': 'gkd',
        'QUEUE': 'normal', # Maximum 30 minutes
        'PPN': 32
    },
    'BW_ORTE': {
        'RESOURCE': 'ncsa.bw',
        'NETWORK_INTERFACE': 'ipogif0',
        'TASK_LAUNCH_METHOD': "ORTE",
        'TARGET': 'node',
        'AGENT_SPAWNER': 'POPEN',
        'PROJECT': 'gkd',
        'QUEUE': 'normal', # Maximum 30 minutes
        'PPN': 32
    },
    'TITAN': {
        'RESOURCE': 'ornl.titan',
        'TARGET': 'node',
        'SCHEMA': 'local',
        'TASK_LAUNCH_METHOD': "ORTE",
        'AGENT_SPAWNER': 'POPEN',
        #'QUEUE': 'debug', # Maximum 60 minutes
        'NETWORK_INTERFACE': 'ipogif0',
        'PROJECT': 'csc168',
        'PPN': 16,
        'PRE_EXEC_PREPEND': [
            #'module use --append /u/sciteam/marksant/privatemodules',
            #'module load use.own',
            #'module load openmpi/git'
        ]
    },
    'STAMPEDE_SSH': {
        'RESOURCE': 'xsede.stampede',
        #'SCHEMA': 'local',
        #'TASK_LAUNCH_METHOD': "ORTE",
        'AGENT_SPAWNER': 'POPEN',
        'TARGET': 'local',
        #'QUEUE': 'development',
        'QUEUE': 'normal',
        'PROJECT': 'TG-MCB090174', # RADICAL
        'PPN': 16,
        'PRE_EXEC_PREPEND': [
        ]
    },
    'STAMPEDE_ORTE': {
        'RESOURCE': 'xsede.stampede_orte',
        #'SCHEMA': 'local',
        #'TASK_LAUNCH_METHOD': "ORTE",
        'AGENT_SPAWNER': 'POPEN',
        'TARGET': 'local',
        #'QUEUE': 'development',
        'QUEUE': 'normal',
        'PROJECT': 'TG-MCB090174', # RADICAL
        'PPN': 16,
        'PRE_EXEC_PREPEND': [
        ]
    },
    'STAMPEDE_ORTELIB': {
        'RESOURCE': 'xsede.stampede_ortelib',
        #'SCHEMA': 'local',
        'TARGET': 'local',
        #'QUEUE': 'development',
        'QUEUE': 'normal',
        'PROJECT': 'TG-MCB090174', # RADICAL
        'PPN': 16,
        'PRE_EXEC_PREPEND': [
        ]
    },
    'COMET': {
        'RESOURCE': 'xsede.comet',
        #'TASK_LAUNCH_METHOD': "MPIRUN_RSH",
        #'AGENT_LAUNCH_METHOD': "SSH",
        'AGENT_SPAWNER': 'POPEN',
        'TARGET': 'node',
        'QUEUE': 'compute', # Maximum 72 nodes (1728 cores) / 48 hours
        'PROJECT': 'TG-MCB090174', # RADICAL
        'PPN': 24,
        'PRE_EXEC_PREPEND': [
        ]
    },
    'COMET_ORTE': {
        'RESOURCE': 'xsede.comet_orte',
        #'TASK_LAUNCH_METHOD': "MPIRUN_RSH",
        #'AGENT_LAUNCH_METHOD': "SSH",
        'AGENT_SPAWNER': 'POPEN',
        'TARGET': 'node',
        'QUEUE': 'compute', # Maximum 72 nodes (1728 cores) / 48 hours
        'PROJECT': 'TG-MCB090174', # RADICAL
        'PPN': 24,
        'PRE_EXEC_PREPEND': [
        ]
    },
    'ARCHER': {
        'RESOURCE': 'epsrc.archer',
        'TARGET': 'node',
        'TASK_LAUNCH_METHOD': "ORTE",
        'QUEUE': 'short', # Jobs can range from 1-8 nodes (24-192 cores) and can have a maximum walltime of 20 minutes.
        'NETWORK_INTERFACE': 'ipogif0',
        'PROJECT': 'e290',
        'PPN': 24,
        'PRE_EXEC_PREPEND': [
        ]
    },
}

#------------------------------------------------------------------------------
#
def pilot_state_cb(pilot, state):

    if not pilot:
        return

    report.warn("\n[Callback]: ComputePilot '%s' state: %s." % (pilot.uid, state))

    #if state == rp.FAILED:
    #    raise rp.PilotException("Pilot %s failed (CB)" % pilot.uid)

    #if state == rp.CANCELED:
    #    raise rp.PilotException("Pilot %s canceled (CB)" % pilot.uid)

CNT = 0
#------------------------------------------------------------------------------
#
def unit_state_cb(unit, state):

    if not unit:
        return

    global CNT

    report.info("[Callback]: unit %s on %s: %s.\n" % (unit.uid, unit.pilot_id, state))

    if state in [rp.FAILED, rp.DONE, rp.CANCELED]:
        CNT += 1
        report.info("[Callback]: # %6d\n" % CNT)

    if state == rp.FAILED:
        report.error("stderr: %s\n" % unit.stderr)


#------------------------------------------------------------------------------
#
def wait_queue_size_cb(umgr, wait_queue_size):
    report.info("[Callback]: wait_queue_size: %s.\n" % wait_queue_size)
#------------------------------------------------------------------------------


def construct_agent_config(num_sub_agents, num_exec_instances_per_sub_agent,
        target, network_interface=None, clone_factor=1):

    config = {

        # directory for staging files inside the agent sandbox
        "staging_area"         : "staging_area",

        # url scheme to indicate the use of staging_area
        "staging_scheme"       : "staging",

        # max number of cu out/err chars to push to db
        "max_io_loglength"     : 1024,

        # max time period to collect db notifications into bulks (seconds)
        "bulk_collection_time" : 1.0,

        # time to sleep between database polls (seconds)
        "db_poll_sleeptime"    : 0.1,

        # time between checks of internal state and commands from mothership (seconds)
        "heartbeat_interval"   : 10,

        # factor by which the number of units are increased at a certain step.  Value of
        # "1" will leave the units unchanged.  Any blowup will leave on unit as the
        # original, and will then create clones with an changed unit ID (see blowup()).
        "clone" : {
            "AgentWorker"                 : {"input" : 1, "output" : clone_factor},
            "AgentStagingInputComponent"  : {"input" : 1, "output" : 1},
            "AgentSchedulingComponent"    : {"input" : 1, "output" : 1},
            "AgentExecutingComponent"     : {"input" : 1, "output" : 1},
            "AgentStagingOutputComponent" : {"input" : 1, "output" : 1}
        },

        # flag to drop all blown-up units at some point in the pipeline.  The units
        # with the original IDs will again be left untouched, but all other units are
        # silently discarded.
        # 0: drop nothing
        # 1: drop clones
        # 2: drop everything
        "drop" : {
            "AgentWorker"                 : {"input" : 0, "output" : 0},
            "AgentStagingInputComponent"  : {"input" : 0, "output" : 0},
            "AgentSchedulingComponent"    : {"input" : 0, "output" : 0},
            "AgentExecutingComponent"     : {"input" : 0, "output" : 0},
            "AgentStagingOutputComponent" : {"input" : 0, "output" : 1}
        }
    }

    # interface for binding zmq to
    if network_interface:
        config["network_interface"] = "ipogif0"

    layout =  {
        "agent_0"   : {
            "target": "local",
            "pull_units": True,
            "sub_agents": [],
            "bridges" : [
                # Leave the bridges on agent_0 for now
                "agent_staging_input_queue",
                "agent_scheduling_queue",
                "agent_executing_queue",
                "agent_staging_output_queue",

                "agent_unschedule_pubsub",
                "agent_reschedule_pubsub",
                "agent_command_pubsub",
                "agent_state_pubsub"
            ],
            "components" : {
                # We'll only toy around with the AgentExecutingComponent for now
                "AgentStagingInputComponent": 1,
                "AgentSchedulingComponent": 1,
                "AgentExecutingComponent": 0,
                "AgentStagingOutputComponent" : 1
            }
        }
    }

    for sub_agent_id in range(1, num_sub_agents+1):

        sub_agent_name = "agent_%d" % sub_agent_id

        layout[sub_agent_name] = {
            "components": {
                "AgentExecutingComponent": num_exec_instances_per_sub_agent,
            },
            "target": target
        }

        # Add sub-agent to list of sub-agents
        layout["agent_0"]["sub_agents"].append(sub_agent_name)

    # Add the complete constructed layout to the agent config now
    config["agent_layout"] = layout

    return config
#
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
def run_experiment(backend, pilot_cores, pilot_runtime, cu_runtime, cu_cores, cu_count, generations, cu_mpi, profiling, agent_config, cancel_on_all_started=False, barriers=[], metadata=None):

    # Profiling
    if profiling:
        os.environ['RADICAL_PILOT_PROFILE'] = 'TRUE'
    else:
        os.environ.pop('RADICAL_PILOT_PROFILE', None)

    if not metadata:
        metadata = {}

    metadata.update({
        'backend': backend,
        'pilot_cores': pilot_cores,
        'pilot_runtime': pilot_runtime,
        'cu_runtime': cu_runtime,
        'cu_cores': cu_cores,
        'cu_count': cu_count,
        'generations': generations,
        'barriers': barriers,
        'profiling': profiling
    })

    # Create a new session. No need to try/except this: if session creation
    # fails, there is not much we can do anyways...
    session = rp.Session()
    report.info("session id: %s\n" % session.uid)
    report.info("Experiment - Backend:%s, PilotCores:%d, PilotRuntime:%d, CURuntime:%d, CUCores:%d, CUCount:%d\n" % \
        (backend, pilot_cores, pilot_runtime, cu_runtime, cu_cores, cu_count))

    cfg = session.get_resource_config(resource_config[backend]['RESOURCE'])

    # create a new config based on the old one, and set a different queue
    new_cfg = rp.ResourceConfig(resource_config[backend]['RESOURCE'], cfg)

    # Insert pre_execs at the beginning in reverse order
    if 'PRE_EXEC_PREPEND' in resource_config[backend]:
        for entry in resource_config[backend]['PRE_EXEC_PREPEND'][::-1]:
            new_cfg.pre_bootstrap_1.insert(0, entry)

    # Change task launch method
    if 'TASK_LAUNCH_METHOD' in resource_config[backend]:
        new_cfg.task_launch_method = resource_config[backend]['TASK_LAUNCH_METHOD']

    # Change MPI launch method
    if 'MPI_LAUNCH_METHOD' in resource_config[backend]:
        new_cfg.mpi_launch_method = resource_config[backend]['MPI_LAUNCH_METHOD']

    # Change agent launch method
    if 'AGENT_LAUNCH_METHOD' in resource_config[backend]:
        new_cfg.agent_launch_method = resource_config[backend]['AGENT_LAUNCH_METHOD']

    # Change method to spawn tasks
    if 'AGENT_SPAWNER' in resource_config[backend]:
        new_cfg.agent_spawner = resource_config[backend]['AGENT_SPAWNER']

    # Don't install a new version of RP
    new_cfg.rp_version = RP_VERSION
    new_cfg.virtenv_mode = VIRTENV_MODE

    # Barrier
    if BARRIER_AGENT_LAUNCH in barriers:
        new_cfg.pre_bootstrap_1.append("export RADICAL_PILOT_BARRIER=$PWD/staging_area/%s" % 'start_barrier')

    # now add the entry back.  As we did not change the config name, this will
    # replace the original configuration.  A completely new configuration would
    # need a unique name.
    session.add_resource_config(new_cfg)

    # all other pilot code is now tried/excepted.  If an exception is caught, we
    # can rely on the session object to exist and be valid, and we can thus tear
    # the whole RP stack down via a 'session.close()' call in the 'finally'
    # clause...
    try:

        pmgr = rp.PilotManager(session=session)
        pmgr.register_callback(pilot_state_cb)

        pdesc = rp.ComputePilotDescription()
        pdesc.resource = resource_config[backend]['RESOURCE']
        pdesc.cores = pilot_cores
        if 'QUEUE' in resource_config[backend]:
            pdesc.queue = resource_config[backend]['QUEUE']
        if 'SCHEMA' in resource_config[backend]:
            pdesc.access_schema = resource_config[backend]['SCHEMA']
        if 'PROJECT' in resource_config[backend]:
            pdesc.project = resource_config[backend]['PROJECT']
        pdesc.runtime = pilot_runtime
        pdesc.cleanup = False

        pdesc._config = agent_config

        pilot = pmgr.submit_pilots(pdesc)

        umgr = rp.UnitManager(session=session, scheduler=SCHEDULER)
        #umgr.register_callback(unit_state_cb, rp.UNIT_STATE)
        umgr.register_callback(wait_queue_size_cb, rp.WAIT_QUEUE_SIZE)
        umgr.add_pilots(pilot)

        # Wait until the pilot is active before we start submitting things
        if BARRIER_CLIENT_SUBMIT in barriers:
            report.info("Waiting for pilot %s to become active ...\n" % pilot.uid)
            pilot.wait(state=[rp.ACTIVE, rp.FAILED, rp.CANCELED])

        cuds = []
        for generation in range(generations):
            for unit_count in range(0, cu_count):
                cud = rp.ComputeUnitDescription()
                cud.executable     = "/bin/sh"
                cud.arguments      = ["-c", "date && hostname -f && sleep %d && date" % cu_runtime]
                cud.cores          = cu_cores
                cud.mpi            = cu_mpi
                cuds.append(cud)

            # Switch behavior based on barrier type
            if BARRIER_GENERATION in barriers:
                # We want to the submission below to happen for this set of units only
                report.info("Submitting %d units for generation %d.\n" % (len(cuds), generation))
                pass
            elif BARRIER_GENERATION not in barriers and generation == generations-1:
                report.info("Submitting %d units for all %d generations.\n" % (len(cuds), generations))
                pass
            elif BARRIER_GENERATION not in barriers and generation != generations-1:
                # We will add all the generations at once and only fall through with the last generation only
                continue
            else:
                raise("Unexpected condition. Barriers: %s, generation: %d, generations: %d" % (barriers, generation, generations))

            units = umgr.submit_units(cuds)

            # With the current un-bulkyness of the client <-> mongodb interaction,
            # it is difficult to really test the agent if queuing time is too short.
            # This barrier waits with starting the agent until all units have
            # reached the database.
            if BARRIER_AGENT_LAUNCH in barriers:
                umgr.wait_units(state=rp.AGENT_STAGING_INPUT_PENDING)
                tmp_fd, tmp_name = tempfile.mkstemp()
                os.close(tmp_fd)
                sd_pilot = {
                    'source': 'file://%s' % tmp_name,
                    'target': 'staging:///%s' % 'start_barrier',
                    'action': rp.TRANSFER
                }
                pilot.stage_in(sd_pilot)
                os.remove(tmp_name)

            # If we are only interested in startup times, we can cancel once that
            # has been achieved, which might save us some cpu hours.
            if cancel_on_all_started:
                wait_states = [rp.EXECUTING, rp.DONE, rp.FAILED, rp.CANCELED]
            else:
                wait_states = [rp.DONE, rp.FAILED, rp.CANCELED]
            umgr.wait_units(unit_ids=[cu.uid for cu in units], state=wait_states)

            for cu in units:
                report.plain("* Task %s state %s, exit code: %s, started: %s, finished: %s" \
                    % (cu.uid, cu.state, cu.exit_code, cu.start_time, cu.stop_time))

            # reset list for next generation
            cuds = []


    except rp.PilotException as e:
        session._logger.exception("Caught a Pilot Exception, cleaning up ...")

    except Exception as e:
        session._logger.exception("caught exception")
        raise

    except (KeyboardInterrupt, SystemExit) as e:
        # the callback called sys.exit(), and we can here catch the
        # corresponding KeyboardInterrupt exception for shutdown.  We also catch
        # SystemExit (which gets raised if the main threads exits for some other
        # reason).
        report.error("need to exit now: %s\n" % e)

    finally:

        if metadata:
            report.info("Inserting meta data into session ...\n")
            rp.utils.inject_metadata(session, metadata)

        session.close(cleanup=False, terminate=True)

        return session._uid, metadata
#
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Pilot generally operates on nodes, and not on cores here.
#
# Iterable: [cu_cores_var, cu_duration_var, num_sub_agents_var, num_exec_instances_per_sub_agent_var, nodes_var]
# Quantitative: repetitions, [cu_count | generations], pilot_runtime
# Config: backend, exclusive_agent_nodes, label, sort_nodes, skip_few_nodes, profiling, barriers
# Static: cu=/bin/sleep
#
def iterate_experiment(
        backend,
        label,
        repetitions=1,
        exclusive_agent_nodes=True,
        barriers=[],
        clone=False,
        cu_cores_var=[1], # Number of cores per CU to iterate over
        cu_duration_var=[0], # Duration of the payload
        cancel_on_all_started=False, # Quit once everything is started.
        cu_count=None, # By default calculate the number of cores based on cores
        cu_mpi=False, # Which launch method to use
        generations=1, # Multiple the number of
        num_sub_agents_var=[1], # Number of sub-agents to iterate over
        num_exec_instances_per_sub_agent_var=[1], # Number of workers per sub-agent to iterate over
        nodes_var=[1], # The number of nodes to allocate for running CUs
        sort_nodes_var=True,
        skip_few_nodes=False, # skip if nodes < cu_cores
        pilot_runtime=30, # Maximum walltime for experiment TODO: guesstimate?
        profiling=True # Enable/Disable profiling
):

    f = open('%s.txt' % label, 'a')

    # While it technically works, it is less useful to use this combination,
    # and might therefore mean a misunderstanding.
    if generations > 1 and cancel_on_all_started:
        raise Exception("cancel_on_all_started not supported for multiple generations")

    # Shuffle some of the input parameters for statistical sanity
    random.shuffle(cu_cores_var)
    random.shuffle(cu_duration_var)
    random.shuffle(num_sub_agents_var)
    random.shuffle(num_exec_instances_per_sub_agent_var)

    # Allows to skip sorting the number of nodes,
    # so that the smallest pilots runs first.
    if sort_nodes_var:
        random.shuffle(nodes_var)

    # Variable to keep track of sessions
    sessions = {}

    for iter in range(repetitions):

        for nodes in nodes_var:

            for cu_cores in cu_cores_var:

                # Allow to specify FULL node, that translates into the PPN
                if cu_cores == 'FULL':
                    cu_cores = int(resource_config[backend]['PPN'])

                for num_sub_agents in num_sub_agents_var:

                    for num_exec_instances_per_sub_agent in num_exec_instances_per_sub_agent_var:

                        if exclusive_agent_nodes:
                            # Allocate some extra nodes for the sub-agents
                            pilot_nodes = nodes + num_sub_agents
                        else:
                            # "steal" from the nodes that are available for CUs
                            pilot_nodes = nodes

                        for cu_duration in cu_duration_var:

                            # Pilot Desc takes cores, so we translate from nodes here
                            pilot_cores = int(resource_config[backend]['PPN']) * pilot_nodes

                            # Number of cores available for CUs
                            effective_cores = int(resource_config[backend]['PPN']) * nodes

                            # Don't need full node experiments for low number of nodes,
                            # as we have no equivalent in single core experiments
                            if skip_few_nodes and nodes < cu_cores:
                                continue


                            # Check if fixed cu_count was specified
                            # Note: make a copy because of the loop
                            if cu_count:
                                this_cu_count = cu_count
                            else:
                                # keep core consumption equal
                                this_cu_count = effective_cores / cu_cores

                            if cu_duration == 'GUESSTIMATE':
                                cus_per_gen = effective_cores / cu_cores
                                cu_duration = 60 + cus_per_gen / num_sub_agents
                                report.warn("CU_DURATION GUESSTIMATED at %d seconds.\n" % cu_duration)
                            # Clone
                            if clone:
                                clone_factor = this_cu_count
                                this_cu_count = 1
                            else:
                                clone_factor = 1

                            # Create and agent layout
                            agent_config = construct_agent_config(
                                num_sub_agents=num_sub_agents,
                                num_exec_instances_per_sub_agent=num_exec_instances_per_sub_agent,
                                target=resource_config[backend]['TARGET'],
                                network_interface=resource_config[backend].get('NETWORK_INTERFACE'),
                                clone_factor=clone_factor
                            )


                            # Fire!!
                            sid, meta = run_experiment(
                                backend=backend,
                                barriers=barriers,
                                pilot_cores=pilot_cores,
                                pilot_runtime=pilot_runtime,
                                cu_runtime=cu_duration,
                                cancel_on_all_started=cancel_on_all_started,
                                cu_cores=cu_cores,
                                cu_count=this_cu_count,
                                generations=generations,
                                cu_mpi=cu_mpi,
                                profiling=profiling,
                                agent_config=agent_config,
                                metadata={
                                    'label': label,
                                    'repetitions': repetitions,
                                    'iteration': iter,
                                    'exclusive_agent_nodes': exclusive_agent_nodes,
                                    'num_sub_agents': num_sub_agents,
                                    'num_exec_instances_per_sub_agent': num_exec_instances_per_sub_agent,
                                    'effective_cores': effective_cores,
                                }
                            )

                            # Append session id to return value
                            sessions[sid] = meta

                            # Record session id to file
                            f.write('%s - %s - %s\n' % (sid, time.ctime(), str(meta)))
                            f.flush()

    f.close()
    return sessions
#
#-------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Variable CU duration (0, 1, 10, 30, 60, 120)
# Fixed CU count (1024)
# Fixed CU cores (1)
# CU = /bin/sleep
# Fixed Pilot cores (256)
#
# Goal: investigate the relative overhead of LM in relation to the runtime of the CU
#
def exp1(backend):

    sessions = iterate_experiment(
        backend=backend,
        label=inspect.currentframe().f_code.co_name,
        repetitions=1,
        cu_count=16,
        cu_duration_var=[0, 1, 10], #, 30, 60, 120]
    )
    return sessions

#
#-------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Single backend
# Fixed CU duration (60)
# Variable CU cores (1-256)
# Variable CU count (1024-4)
# Fixed Pilot nodes (8)
#
# Goal: Investigate the relative overhead of small tasks compared to larger tasks
#
def exp2(backend):

    sessions = iterate_experiment(
        repetitions=1,
        backend=backend,
        label=inspect.currentframe().f_code.co_name,
        cu_duration_var=[60],
        cu_cores_var=[1,2,4,8,16,32,64,128,256],
        generations=1,
        nodes_var=[16]
    )
    return sessions
#
#-------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
# Goal: Investigate the effect of number of number of ExecWorkers for a single sub-agent
#
# Single backend
# Fixed CU duration (60s)
# Fixed CU cores (1)
# CU = /bin/sleep
# Fixed Pilot nodes (16)
# Variable number of exec workers (1-8)
#
def exp3(backend):

    sessions = iterate_experiment(
        repetitions=1,
        backend=backend,
        label=inspect.currentframe().f_code.co_name,
        cu_duration_var=[0, 60],
        generations=5,
        cu_cores_var=[1],
        nodes_var=[16],
        num_exec_instances_per_sub_agent_var=[1, 2, 4, 8, 16, 24],
        num_sub_agents_var=[1]
    )

    return sessions
#
#-------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
# Fixed CU duration (60)
# Fixed backend (ORTE)
# Variable CU count (5 generations)
# Variable CU cores (1, 32)
# CU = /bin/sleep
# Variable Pilot cores (256, 512, 1024, 2048, 4096, 8192)
#
# Goals: A) Investigate the scale of things. 
#        B) Investigate the effect of 1 per node vs 32 per node
#
def exp4(repeat):

    f = open('exp4.txt', 'a')
    f.write('%s\n' % time.ctime())

    agent_config = {}
    agent_config['number_of_workers'] = {}
    agent_config['number_of_workers']['ExecWorker'] = 1

    sessions = {}

    # Enable/Disable profiling
    profiling=True

    backend = 'ORTE'

    cu_sleep = 60

    generations = 5

    # The number of cores to acquire on the resource
    nodes_var = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    random.shuffle(nodes_var)

    # Single core and multicore
    cu_cores_var = [1, 32]
    random.shuffle(cu_cores_var)

    # Maximum walltime for experiment
    pilot_runtime = 30 # should we guesstimate this?

    for iter in range(repeat):

        for nodes in nodes_var:

            pilot_cores = int(resource_config[backend]['PPN']) * nodes

            for cu_cores in cu_cores_var:

                # Don't need full node experiments for low number of nodes,
                # as we have no equivalent in single core experiments
                if nodes < cu_cores:
                    continue

                # keep core consumption equal (4 generations)
                cu_count = (generations * pilot_cores) / cu_cores

                sid = run_experiment(
                    backend=backend,
                    pilot_cores=pilot_cores,
                    pilot_runtime=pilot_runtime,
                    cu_runtime=cu_sleep,
                    cu_cores=cu_cores,
                    cu_count=cu_count,
                    profiling=profiling,
                    agent_config=agent_config
                )

                sessions[sid] = {
                    'backend': backend,
                    'pilot_cores': pilot_cores,
                    'pilot_runtime': pilot_runtime,
                    'cu_runtime': cu_sleep,
                    'cu_cores': cu_cores,
                    'cu_count': cu_count,
                    'profiling': profiling,
                    'iteration': iter,
                    'number_of_workers': agent_config['number_of_workers']['ExecWorker']
                }
                f.write('%s - %s\n' % (sid, str(sessions[sid])))
                f.flush()

    f.close()
    return sessions
#
#-------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
# Variable CU duration (60)
# Fixed backend (ORTE)
# Variable CU count (5 generations)
# Variable CU cores (1, 32)
# CU = /bin/sleep
# Variable Pilot cores (256, 512, 1024, 2048, 4096, 8192)
#
# Goals: A) Investigate the scale of things. 
#        B) Investigate the effect of 1 per node vs 32 per node
#
def exp5(repeat):

    f = open('exp5.txt', 'a')
    f.write('%s\n' % time.ctime())

    agent_config = {}
    agent_config['number_of_workers'] = {}
    agent_config['number_of_workers']['ExecWorker'] = 8

    sessions = {}

    # Enable/Disable profiling
    profiling=True

    backend = 'TITAN'

    generations = 1

    # The number of cores to acquire on the resource
    #nodes_var = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    #nodes_var = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    nodes_var = [256]
    #random.shuffle(nodes_var)

    # Single core and multicore
    #cu_cores_var = [1, resource_config[backend]['PPN']]
    #random.shuffle(cu_cores_var)
    cu_cores_var = [1]
	
    # Maximum walltime for experiment
    pilot_runtime = 180 # should we guesstimate this?

    cu_sleep = 3600

    for iter in range(repeat):

        for nodes in nodes_var:
            
            pilot_cores = int(resource_config[backend]['PPN']) * nodes

            for cu_cores in cu_cores_var:
                
                # Don't need full node experiments for low number of nodes,
                # as we have no equivalent in single core experiments
                if nodes < cu_cores:
                    continue

                # keep core consumption equal
                cu_count = (generations * pilot_cores) / cu_cores

                #cu_sleep = max(60, cu_count / 5)

                sid = run_experiment(
                    backend=backend,
                    pilot_cores=pilot_cores,
                    pilot_runtime=pilot_runtime,
                    cu_runtime=cu_sleep,
                    cu_cores=cu_cores,
                    cu_count=cu_count,
                    profiling=profiling,
                    agent_config=agent_config
                )

                sessions[sid] = {
                    'backend': backend,
                    'pilot_cores': pilot_cores,
                    'pilot_runtime': pilot_runtime,
                    'cu_runtime': cu_sleep,
                    'cu_cores': cu_cores,
                    'cu_count': cu_count,
                    'profiling': profiling,
                    'iteration': iter,
                    'number_of_workers': agent_config['number_of_workers']['ExecWorker']
                }
                f.write('%s - %s\n' % (sid, str(sessions[sid])))
                f.flush()

    f.close()
    return sessions
#
#-------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# Variable CU duration (0, 600, 3600)
# Single backend
# Variable CU count (1 generations)
# Variable CU cores = pilot cores
# CU = /bin/sleep
# Variable Pilot cores (32(2), 128(4), 512(32), 1024(64), 2048(128), 4096, 8192)
#
# Goals: A) Investigate the scale of things. 
#        B) Investigate the effect of 1 per node vs 32 per node
#
def exp6(repeat):

    f = open('exp6.txt', 'a')
    f.write('%s\n' % time.ctime())

    agent_config = {}
    agent_config['number_of_workers'] = {}
    agent_config['number_of_workers']['ExecWorker'] = 8

    sessions = {}

    # Enable/Disable profiling
    profiling=True

    backend = 'TITAN'

    generations = 1

    # The number of cores to acquire on the resource
    #nodes_var = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    nodes_var = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    #cores_var = [32, 128, 512, 1024, 2048, 4096]
    # Disable nodes_var shuffle to get the some results quickly because of queuing time
    #random.shuffle(nodes_var)

    # Single core and multicore
    #cu_cores_var = [1, resource_config[backend]['PPN']]
    #random.shuffle(cu_cores_var)
    cu_cores_var = [1]
	
    # Maximum walltime for experiment
    pilot_runtime = 60 # should we guesstimate this?

    cu_sleep = 3600

    for iter in range(repeat):

        for nodes in nodes_var:
            
            pilot_cores = int(resource_config[backend]['PPN']) * nodes

            for cu_cores in cu_cores_var:
                
                # Don't need full node experiments for low number of nodes,
                # as we have no equivalent in single core experiments
                if nodes < cu_cores:
                    continue

                # keep core consumption equal
                cu_count = (generations * pilot_cores) / cu_cores

                #cu_sleep = max(60, cu_count / 5)

                sid = run_experiment(
                    backend=backend,
                    pilot_cores=pilot_cores,
                    pilot_runtime=pilot_runtime,
                    cu_runtime=cu_sleep,
                    cu_cores=cu_cores,
                    cu_count=cu_count,
                    profiling=profiling,
                    agent_config=agent_config
                )

                sessions[sid] = {
                    'backend': backend,
                    'pilot_cores': pilot_cores,
                    'pilot_runtime': pilot_runtime,
                    'cu_runtime': cu_sleep,
                    'cu_cores': cu_cores,
                    'cu_count': cu_count,
                    'profiling': profiling,
                    'iteration': iter,
                    'number_of_workers': agent_config['number_of_workers']['ExecWorker']
                }
                f.write('%s - %s\n' % (sid, str(sessions[sid])))
                f.flush()

    f.close()
    return sessions
#
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
# Single resource experiment.
#
# Investigate the performance of RP with different SUB-AGENT setups.
#
def exp7(backend):

    sessions = iterate_experiment(
        backend=backend,
        label=inspect.currentframe().f_code.co_name,
        repetitions=1,
        generations=5,
        cu_mpi=True,
        cu_duration_var=[10],
        num_sub_agents_var=[1], # Number of sub-agents to iterate over
        num_exec_instances_per_sub_agent_var=[1], # Number of workers per sub-agent to iterate over
        nodes_var=[10], # The number of nodes to allocate for running CUs
        sort_nodes_var=False # Disable nodes_var shuffle to get the some results quickly because of queuing time
    )
    return sessions
#
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#
# Single resource experiment.
#
# Goal: determine degree of scaling with adding execworkers/sub-agents
#
# I think scaling with exec workers, and specifically to what problem size
# this behaves kinda linear and also that this scaling is independent
# from CU size (scalar / mpi)
#
def exp8(backend):

    sessions = iterate_experiment(
        backend=backend,
        label=inspect.currentframe().f_code.co_name,
        repetitions=1,
        generations=5,
        #cu_count=1,
        #cu_duration_var=['GUESSTIMATE'],
        barriers=[BARRIER_AGENT_LAUNCH],
        cu_duration_var=[60],
        num_sub_agents_var=[10], # Number of sub-agents to iterate over
        #num_sub_agents_var=[1, 2, 4, 8, 16, 32], # Number of sub-agents to iterate over
        #num_exec_instances_per_sub_agent_var=[1, 2, 4, 8, 16, 24], # Number of workers per sub-agent to iterate over
        num_exec_instances_per_sub_agent_var=[1],
        #nodes_var=[1, 2, 4, 8, 16, 32] # The number of nodes to allocate for running CUs
        nodes_var=[16],
    )
    return sessions
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
# Single resource experiment.
#
# Goal: determine degree of scaling with adding execworkers/sub-agents
#
# I think scaling with exec workers, and specifically to what problem size
# this behaves kinda linear and also that this scaling is independent
# from CU size (scalar / mpi)
#
def exp9(backend):

    sessions = iterate_experiment(
        backend=backend,
        label=inspect.currentframe().f_code.co_name,
        repetitions=1,
        generations=5,
        #cu_duration_var=['GUESSTIMATE'],
        barriers=[BARRIER_CLIENT_SUBMIT, BARRIER_GENERATION],
        cu_duration_var=[60],
        num_sub_agents_var=[10], # Number of sub-agents to iterate over
        #exclusive_agent_nodes=False,
        #num_sub_agents_var=[1, 2, 4, 8, 16, 32], # Number of sub-agents to iterate over
        #num_exec_instances_per_sub_agent_var=[1, 2, 4, 8, 16, 24], # Number of workers per sub-agent to iterate over
        num_exec_instances_per_sub_agent_var=[1],
        #nodes_var=[1, 2, 4, 8, 16, 32, 48] # The number of nodes to allocate for running CUs
        nodes_var=[16, 48],
    )
    return sessions
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
# Single resource experiment.
#
# Goal: determine degree of scaling with adding execworkers/sub-agents
#
# I think scaling with exec workers, and specifically to what problem size
# this behaves kinda linear and also that this scaling is independent
# from CU size (scalar / mpi)
#
def exp10(backend):

    sessions = iterate_experiment(
        backend=backend,
        label=inspect.currentframe().f_code.co_name,
        repetitions=1,
        generations=5,
        #cu_duration_var=['GUESSTIMATE'],
        barriers=[BARRIER_CLIENT_SUBMIT],
        cu_duration_var=[60],
        num_sub_agents_var=[1], # Number of sub-agents to iterate over
        #exclusive_agent_nodes=False,
        #num_sub_agents_var=[1, 2, 4, 8, 16, 32], # Number of sub-agents to iterate over
        #num_exec_instances_per_sub_agent_var=[1, 2, 4, 8, 16, 24], # Number of workers per sub-agent to iterate over
        num_exec_instances_per_sub_agent_var=[1],
        #nodes_var=[1, 2, 4, 8, 16, 32, 48] # The number of nodes to allocate for running CUs
        nodes_var=[1]
    )
    return sessions
#
#-------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
if __name__ == "__main__":

    report.title('Experiment Driver (RP version %s)' % rp.version)

    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', dest = 'backend', help = 'Backend to run on', default = "LOCAL")
    parser.add_argument('--run', nargs = '*', dest = 'experiments', help = 'Experiments to run')
    parser.add_argument('--list', action="store_true", help = 'List experiments')

    args = parser.parse_args()

    experiments = [exp for exp in locals().copy() if exp.startswith('exp')]

    if args.list:
        report.info('Implemented experiments: %s\n' % experiments)
        exit(0)

    if not args.experiments:
        report.error('Must specify which experiment(s) to run!\n')
        exit(1)
    requested = args.experiments

    for r in requested:
        if r not in experiments:
            report.error("Experiment %s not implemented!\n" % r)
            exit(1)

    if args.backend not in resource_config:
        report.error('Backend "%s" not in resource config!\n' % args.backend)
        exit(1)
    backend = args.backend

    report.info('Running on backend "%s"\n' % backend)
    report.info('Requested experiments: %s\n' % requested)

    for r in requested:
        sessions = locals()[r](backend)
        pprint.pprint(sessions)
#
#------------------------------------------------------------------------------
