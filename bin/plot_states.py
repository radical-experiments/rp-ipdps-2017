import os
import sys
import time
import glob
import pandas as pd

from common import JSON_DIR, PICKLE_DIR, get_resources, get_spawners, get_lm, get_mpi, find_preprocessed_sessions

# Global Pandas settings
pd.set_option('display.width', 180)
pd.set_option('io.hdf.default_format','table')

import matplotlib as mp


###############################################################################
#
# TODO: add concurrent CUs on right axis
def plot(tr_unit_prof_df, info_df, unit_info_df, pilot_info_df, sid):

    print "Plotting %s ..." % sid

    labels = []

    # Legend info
    info = info_df.loc[sid]

    mpi = get_mpi(unit_info_df, sid)
    #mpi = True
    # For this call assume that there is only one pilot per session
    lms = get_lm(unit_info_df, pilot_info_df, sid, mpi)
    assert len(lms) == 1
    launch_method = lms.values()[0]

    # For this call assume that there is only one pilot per session
    spawners = get_spawners(unit_info_df, pilot_info_df, sid)
    assert len(spawners) == 1
    spawner = spawners.values()[0]

    # For this call assume that there is only one pilot per session
    resources = get_resources(unit_info_df, pilot_info_df, sid)
    assert len(resources) == 1
    resource_label = resources.values()[0]

    # Get only the entries for this session
    uf = unit_info_df[unit_info_df['sid'] == sid]

    result = pd.value_counts(uf['state'].values, sort=False)
    print result

    ax = result.plot(kind='pie', autopct='%.2f%%')
    ax.set_aspect('equal')

    print info
    #mp.pyplot.legend(labels, loc='upper left', fontsize=5)
    mp.pyplot.title("%s (%s)\n"
                    "%d CUs of %d core(s) with a %ds payload on a %d core pilot on %s.\n"
                    "%d sub-agent(s) with %d ExecWorker(s) each. All times are per CU.\n"
                    "RP: %s - RS: %s - RU: %s"
                   % (sid, time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(info['created'])),
                      info['metadata.cu_count'], info['metadata.cu_cores'], info['metadata.cu_runtime'], info['metadata.pilot_cores'], resource_label,
                      info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
                      info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
                      ), fontsize=8)

    mp.pyplot.savefig('%s_plot_states.pdf' % sid)
    mp.pyplot.close()

###############################################################################
#
def find_sessions(json_dir):

    session_paths = glob.glob('%s/rp.session.*json' % json_dir)
    if not session_paths:
        raise Exception("No session files found in directory %s" % json_dir)

    session_files = [os.path.basename(e) for e in session_paths]

    session_ids = [e.rsplit('.json')[0] for e in session_files]

    print "Found sessions in %s: %s" % (json_dir, session_ids)

    return session_ids


###############################################################################
#
if __name__ == '__main__':

    session_ids = []

    # Read from file if specified, otherwise read from stdin
    f = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    for line in f:
        session_ids.append(line.strip())

    if not session_ids:
        session_ids = find_preprocessed_sessions()

    for sid in session_ids:
        session_dir = os.path.join(PICKLE_DIR, sid)

        unit_info_df = pd.read_pickle(os.path.join(session_dir, 'unit_info.pkl'))
        pilot_info_df = pd.read_pickle(os.path.join(session_dir, 'pilot_info.pkl'))
        tr_unit_prof_df = pd.read_pickle(os.path.join(session_dir, 'tr_unit_prof.pkl'))
        session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))

        plot(tr_unit_prof_df, session_info_df, unit_info_df, pilot_info_df, sid)
