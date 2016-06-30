import os
import sys
import time
import pandas as pd

from common import PICKLE_DIR, find_preprocessed_sessions, get_resources, get_spawners, get_lm, get_mpi, LABEL_FONTSIZE, LEGEND_FONTSIZE, TICK_FONTSIZE, LINEWIDTH, BORDERWIDTH

# Global Pandas settings
pd.set_option('display.width', 180)
pd.set_option('io.hdf.default_format','table')

import matplotlib as mp

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

from matplotlib import pyplot as plt
import numpy as np
cmap = plt.get_cmap('jet')

###############################################################################
#
# TODO: add concurrent CUs on right axis
def plot(tr_unit_prof_df, info_df, unit_info_df, pilot_info_df, sid, values, plot_mean, paper):

    print "Plotting %s ..." % sid

    labels = []

    colors = [cmap(i) for i in np.linspace(0, 1, len(values))]
    c = 0

    # Legend info
    info = info_df.loc[sid]

    # mpi = get_mpi(unit_info_df, sid)
    mpi = False
    # For this call assume that there is only one pilot per session
    # lms = get_lm(unit_info_df, pilot_info_df, sid, mpi)
    # assert len(lms) == 1
    # launch_method = lms.values()[0]
    launch_method = 'ORTE_LIB'

    # For this call assume that there is only one pilot per session
    # spawners = get_spawners(unit_info_df, pilot_info_df, sid)
    # assert len(spawners) == 1
    # spawner = spawners.values()[0]
    spawner = 1

    #exit()

    # For this call assume that there is only one pilot per session
    # resources = get_resources(unit_info_df, pilot_info_df, sid)
    # assert len(resources) == 1
    # resource_label = resources.values()[0]
    resource_label = 'bogus'

    # Get only the entries for this session
    tuf = tr_unit_prof_df[tr_unit_prof_df['sid'] == sid]

    #tuf = tuf[tuf['Done'].notnull()]

    # We sort the units based on the order ...
    #tufs = tuf.sort('awo_get_u_pend') # they arrived at the agent
    #tufs = tuf.sort('aec_work_u_pend') # they are picked up by an EW
    tufs = tuf.sort('asc_put_u_pend') # they are scheduled
    #tufs = tuf.sort('asc_get_u_pend') # the are picked up by the scheduler
    #tufs = tuf.sort()
    #tufs = tuf
    print tufs.head()

    if 'core-occ':
        df = tufs['asc_released'] - tufs['asc_allocated'] - info['metadata.cu_runtime']
        ax = df.plot(kind='line', color=colors[c], linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
        labels.append("Core Occupation")
        c += 1
        if plot_mean:
            mean = df.mean()
            mp.pyplot.plot((0,len(df)),(mean, mean), 'r--', linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
            labels.append("Mean Core Occupation overhead")

    if 'sched' in values:
        # ax = (tufs['asc_put_u_pend'] - tufs['asc_work_u_pend']).plot(kind='line', color='cyan')
        df = tufs['asc_allocated'] - tufs['asc_try']
        ax = df.plot(kind='line', color=colors[c], linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
        labels.append("Scheduling")
        c += 1
        if plot_mean:
            mean = df.mean()
            mp.pyplot.plot((0,len(df)),(mean, mean), '--', color='cyan', linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
            labels.append("Mean Scheduling")

    if 'sched2execQ' in values:
        df = tufs['aec_get_u_pend'] - tufs['asc_put_u_pend']
        ax = df.plot(kind='line', color=colors[c], linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
        labels.append("Executor Pickup Delay")
        c += 1
        if plot_mean:
            mean = df.mean()
            mp.pyplot.plot((0,len(df)),(mean, mean), '--', color='yellow', linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
            labels.append("Mean Scheduler2Executor Q")

    if 'pre-spawn' in values:
        df = tufs['aec_handover'] - tufs['asc_allocated']
        ax = df.plot(kind='line', color=colors[c], linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
        labels.append("Pre-spawn slot occupation")
        c += 1
        if plot_mean:
            mean = df.mean()
            mp.pyplot.plot((0,len(df)),(mean, mean), '--', color='orange', linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
            labels.append("Mean Pre-spawn slot occupation")

    if 'spawner' in values:
        df = tufs['aec_start_script'] - tufs['aec_handover']
        ax = df.plot(kind='line', color=colors[c], linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
        #labels.append("Spawner (%s)" % spawner)
        labels.append("Spawning")
        c += 1
        if plot_mean:
            mean = df.mean()
            mp.pyplot.plot((0,len(df)),(mean, mean), '--', color='brown', linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
            labels.append("Mean Spawner (%s)" % spawner)

    if 'launch_cb' in values:
        df = tufs['aec_pickup'] - tufs['aec_start_script']
        ax = df.plot(kind='line', color=colors[c], linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
        labels.append("Launch notification")
        c += 1
        if plot_mean:
            mean = df.mean()
            mp.pyplot.plot((0,len(df)),(mean, mean), '--', color='black', linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
            labels.append("Mean Launch notification")

    if 'runtime' in values:
        df = tufs['aec_after_exec'] - tufs['aec_start_script'] - info['metadata.cu_runtime']
        ax = df.plot(kind='line', color=colors[c], linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
        labels.append('Runtime overhead')
        c += 1
        if plot_mean:
            mean = df.mean()
            mp.pyplot.plot((0,len(df)),(mean, mean), '--', color='orange', linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
            labels.append("Mean Runtime overhead")

    # if False:
        # (tufs['asc_get_u_pend'] - tufs['asic_put_u_pend']).plot(kind='line', color='blue')
        # labels.append("Scheduler Queue")
        # ax = (tufs['aec_work_u_pend'] - tufs['asc_put_u_pend']).plot(kind='line', color='green')
        # labels.append("ExecWorker Queue")

    # ax = (tufs['aec_work_u_pend'] - tufs['aec_get_u_pend']).plot(kind='line', color='green')
    # labels.append("ExecWorker pre-stuff")

    if 'finish_cb' in values:
        df = tufs['aec_complete'] - tufs['aec_after_exec']
        ax = df.plot(kind='line', color=colors[c], linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
        labels.append('Completion Notification Delay')
        c += 1
        if plot_mean:
            mean = df.mean()
            mp.pyplot.plot((0,len(df)),(mean, mean), '--', color='blue', linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
            labels.append('Mean Completion notification')

    if 'post-cb' in values:
        df = tufs['asc_released'] - tufs['aec_complete']
        ax = df.plot(kind='line', color=colors[c], linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
        labels.append("Unscheduling")
        c += 1
        if plot_mean:
            mean = df.mean()
            mp.pyplot.plot((0,len(df)),(mean, mean), '--', color='magenta', linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
            labels.append("Mean Post-callback Resource Release")

    #location = 'upper left'
    location = 'upper right'
    #location = 'lower right'
    legend = mp.pyplot.legend(labels, loc=location, fontsize=LEGEND_FONTSIZE, labelspacing=0)
    legend.get_frame().set_linewidth(BORDERWIDTH)

    #mp.pyplot.legend.get_frame().set_linewidth(0.1)
    #ax.get_frame().set_linewidth(0.1)
    if not paper:
        mp.pyplot.title("%s (%s)\n"
                    "%d CUs of %d core(s) with a %ds payload on a %d core pilot on %s.\n"
                    "%d sub-agent(s) with %d ExecWorker(s) each. All times are per CU.\n"
                    "RP: %s - RS: %s - RU: %s"
                   % (sid, time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(info['created'])),
                      info['metadata.cu_count'], info['metadata.cu_cores'], info['metadata.cu_runtime'], info['metadata.pilot_cores'], resource_label,
                      info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
                      info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
                      ), fontsize=8)
    mp.pyplot.xlabel("Units (ordered by agent scheduling)", fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylabel("Time (s)", fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylim(0.00001, 2000)
    ax.get_xaxis().set_ticks([])
    ax.set_yscale('log', basey=10)

    [i.set_linewidth(BORDERWIDTH) for i in ax.spines.itervalues()]
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)

    # width = 3.487
    width = 3.3
    # height = width / 1.618
    height = 2.7
    fig = mp.pyplot.gcf()
    fig.set_size_inches(width, height)
    #fig.gca().get_frame().set_linewidth(2)
    #plt.gca().get_frame().set_linewidth(2)
    print dir(fig.gca())
    # fig.subplots_adjust(left=0, right=1, top=1, bottom=1)

    # fig.tight_layout(w_pad=0.0, h_pad=0.0, pad=0.1)
    fig.tight_layout(pad=0.1)

    mp.pyplot.savefig('%s_plot1.pdf' % sid)
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
        tr_unit_prof_df = pd.read_pickle(os.path.join(session_dir, 'tr_unit_prof.pkl'))
        session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))

        values = [
            'core-occ', # aggregate
            'sched',
            'sched2execQ',
            #'pre-spawn',
            'spawner',
            #'launch_cb',
            #'runtime',
            'finish_cb',
            'post-cb',

        ]
        plot(tr_unit_prof_df, session_info_df, unit_info_df, pilot_info_df, sid, values, plot_mean=False, paper=True)
