import os
import glob

EXPERIMENT_DIR = '/home/marksant/ipdps2017'

CLIENT_DIR = os.path.join(EXPERIMENT_DIR, 'client')
TARGET_DIR = os.path.join(EXPERIMENT_DIR, 'data/profiling')
JSON_DIR = os.path.join(EXPERIMENT_DIR, 'data/json')
HDF5_DIR   = os.path.join(EXPERIMENT_DIR, 'data/hdf5')
PICKLE_DIR = os.path.join(EXPERIMENT_DIR, 'data/pickle')

RADICAL_PILOT_DBURL = None

#
# Various barrier constants
#
BARRIER_AGENT_LAUNCH= 'barrier_agent_launch'
BARRIER_CLIENT_SUBMIT='barrier_client_submit'
BARRIER_GENERATION='barrier_generation'
barrier_legend = {
    BARRIER_AGENT_LAUNCH: 'Agent-barrier',
    BARRIER_GENERATION: 'Generation-barrier',
    BARRIER_CLIENT_SUBMIT: 'Application-barrier'
}
barrier_colors = {
    BARRIER_AGENT_LAUNCH: 'green',
    BARRIER_GENERATION: 'red',
    BARRIER_CLIENT_SUBMIT: 'blue'
}
barrier_marker = {
    BARRIER_AGENT_LAUNCH: 'x',
    BARRIER_GENERATION: '+',
    BARRIER_CLIENT_SUBMIT: '*'
}
BARRIER_FONTSIZE = 6
LEGEND_FONTSIZE = 6
TICK_FONTSIZE = 6
LABEL_FONTSIZE = 9
TITLE_FONTSIZE = 4
BARRIER_LINEWIDTH = 2.0
LINEWIDTH = 0.5
BORDERWIDTH = 0.5

resource_legend = {
    'titan': 'Titan',
    'comet': 'Comet',
    'bw': 'Blue Waters',
    'orte': 'ORTE CLI',
    'orte_lib': 'ORTE LIB',
    'ssh': 'SSH',
    'aprun': 'ALPS',
    'ccm': 'CCM'
}
resource_colors = {
    'titan': 'green',
    'comet': 'red',
    'bw': 'blue',
    'orte': 'green',
    'orte_lib': 'red',
    'ssh': 'blue',
    'aprun': 'orange',
    'ccm': 'blue',
}
resource_marker = {
    'titan': 'x',
    'comet': '+',
    'bw': '*',
    'orte': '+',
    'orte_lib': '+',
    'ssh': '*',
    'aprun': '+',
    'ccm': '+'

}

###############################################################################
# Get the pilots ppn values for this session
def get_ppn(unit_info_df, pilot_info_df, sid):

    ppn_values = {}

    # Get all units and all pilots for session
    unit_info = unit_info_df[unit_info_df['sid'] == sid]
    pilot_info = pilot_info_df[pilot_info_df['sid'] == sid]

    pilots_in_session = unit_info['pilot'].unique()

    for pilot_id in pilots_in_session:
        pilot = pilot_info.loc[pilot_id]
        ppn = pilot['agent_config.cores_per_node']

        ppn_values[pilot_id] = ppn

    return ppn_values


###############################################################################
#
def find_preprocessed_sessions():

    dir = PICKLE_DIR

    session_paths = glob.glob('%s/rp.session.*' % dir)
    if not session_paths:
        raise Exception("No session files found in directory %s" % dir)

    session_files = [os.path.basename(e) for e in session_paths]

    session_ids = [e.rsplit('.json')[0] for e in session_files]

    print "Found sessions in %s: %s" % (dir, session_ids)

    return session_ids


###############################################################################
# Get the pilots resource labels for this session
def get_resources(unit_info_df, pilot_info_df, sid):

    resources = {}

    # Get all units and all pilots for session
    unit_info = unit_info_df[unit_info_df['sid'] == sid]
    pilot_info = pilot_info_df[pilot_info_df['sid'] == sid]

    pilots_in_session = unit_info['pilot'].unique()

    for pilot_id in pilots_in_session:
        pilot = pilot_info.loc[pilot_id]
        label = pilot['description.resource']

        resources[pilot_id] = label

    return resources


###############################################################################
# Get the pilots spawners for this session
def get_spawners(unit_info_df, pilot_info_df, sid):

    spawners = {}

    # Get all units and all pilots for session
    unit_info = unit_info_df[unit_info_df['sid'] == sid]
    pilot_info = pilot_info_df[pilot_info_df['sid'] == sid]

    pilots_in_session = unit_info['pilot'].unique()

    for pilot_id in pilots_in_session:
        pilot = pilot_info.loc[pilot_id]
        spawner = pilot['agent_config.spawner']

        spawners[pilot_id] = spawner

    return spawners


###############################################################################
# Get the pilots launch methods for this session
def get_lm(unit_info_df, pilot_info_df, sid, mpi):

    lms = {}

    # Get all units and all pilots for session
    unit_info = unit_info_df[unit_info_df['sid'] == sid]
    pilot_info = pilot_info_df[pilot_info_df['sid'] == sid]

    pilots_in_session = unit_info['pilot'].unique()

    for pilot_id in pilots_in_session:
        pilot = pilot_info.loc[pilot_id]
        if mpi:
            lm = pilot['agent_config.mpi_launch_method']
        else:
            lm = pilot['agent_config.task_launch_method']
        lms[pilot_id] = lm

    return lms


###############################################################################
# Get the value of MPI for the first CU (assuming all equal)
def get_mpi(unit_info_df, sid):

    # Get all units and all pilots for session
    unit_info = unit_info_df[unit_info_df['sid'] == sid]
    return unit_info.iloc[0]['description.mpi']
