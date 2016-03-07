import os
import sys
import time
import glob
import pandas as pd
import radical.pilot as rp

from common import PICKLE_DIR, find_preprocessed_sessions, get_resources

# Global Pandas settings
pd.set_option('display.width', 180)
pd.set_option('io.hdf.default_format','table')

import matplotlib as mp

###############################################################################
#
# TODO: add concurrent CUs on right axis
def plot(unit_prof_df, tr_unit_prof_df, info_df, unit_info_df, pilot_info_df, sid):

    print "Plotting %s ..." % sid

    labels = []

    # # Legend info
    info = info_df.loc[sid]

    # For this call assume that there is only one pilot per session
    resources = get_resources(unit_info_df, pilot_info_df, sid)
    assert len(resources) == 1
    resource_label = resources.values()[0]

    df = pd.DataFrame()

    #
    # Pulling in
    #
    populating_df = unit_prof_df[
        (unit_prof_df.cc_populating >= 0) &
        (unit_prof_df.event == 'advance') &
        (unit_prof_df.sid == sid)
        ][['time', 'cc_populating']]

    #
    # Staging in
    #
    stage_in_df = unit_prof_df[
        (unit_prof_df.cc_stage_in >= 0) &
        (unit_prof_df.event == 'advance') &
        (unit_prof_df.sid == sid)
        ][['time', 'cc_stage_in']]

    #
    # Scheduling
    #
    sched_df = unit_prof_df[
        (unit_prof_df.cc_sched >= 0) &
        (unit_prof_df.event == 'advance') &
        (unit_prof_df.sid == sid)
        ][['time', 'cc_sched']]

    #
    # Executing
    #
    exec_df = unit_prof_df[
        (unit_prof_df.cc_exec >= 0) &
        (unit_prof_df.event == 'advance') &
        (unit_prof_df.sid == sid)
        ][['time', 'cc_exec']]

    #
    # Staging out
    #
    stage_out_df = unit_prof_df[
        (unit_prof_df.cc_stage_out >= 0) &
        (unit_prof_df.event == 'advance') &
        (unit_prof_df.sid == sid)
        ][['time', 'cc_stage_out']]

    print sched_df.head()

    df = populating_df
    labels.append("Populating MongoDB")
    df = pd.merge(df, stage_in_df,  on='time', how='outer')
    labels.append("Staging Input Data")
    df = pd.merge(df, sched_df,     on='time', how='outer')
    labels.append("Scheduling")
    df = pd.merge(df, exec_df,      on='time', how='outer')
    labels.append("Executing")
    df = pd.merge(df, stage_out_df, on='time', how='outer')
    labels.append("Staging Output Data")

    df.set_index('time', inplace=True)
    print df.head()

    df.plot(colormap='Paired', drawstyle='steps-post')

    mp.pyplot.legend(labels, loc='upper left', fontsize=5)
#    mp.pyplot.title("Concurrent Compute Units per Component.\n"
#                    "%d CUs of %d core(s) with a %ss payload on a %d core pilot on %s.\n"
#                    "%d sub-agent(s) with %d ExecWorker(s) each. All times are per CU.\n"
#                    "RP: %s - RS: %s - RU: %s"
#                   % (info['metadata.cu_count'], info['metadata.cu_cores'], info['metadata.cu_runtime'], info['metadata.pilot_cores'], resource_label,
#                      info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
#                      info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
#                      ), fontsize=8)
    mp.pyplot.xlabel("Time (s)")
    mp.pyplot.ylabel("Concurrent Compute Units")

    #mp.pyplot.ylim(0,100)
    #mp.pyplot.xlim(1200, 1500)

    mp.pyplot.savefig('%s_plot4.pdf' % sid)
    mp.pyplot.close()


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
        unit_prof_df = pd.read_pickle(os.path.join(session_dir, 'unit_prof.pkl'))
        tr_unit_prof_df = pd.read_pickle(os.path.join(session_dir, 'tr_unit_prof.pkl'))
        session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))

        plot(unit_prof_df, tr_unit_prof_df, session_info_df, unit_info_df, pilot_info_df, sid)
