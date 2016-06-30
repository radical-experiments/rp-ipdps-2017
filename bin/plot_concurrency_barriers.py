import os
import sys
import time
import glob
import pandas as pd

from common import PICKLE_DIR, get_ppn, get_resources, \
    BARRIER_AGENT_LAUNCH, BARRIER_CLIENT_SUBMIT, BARRIER_GENERATION, \
    barrier_legend, barrier_colors, LINEWIDTH, LABEL_FONTSIZE, LEGEND_FONTSIZE, TICK_FONTSIZE, BORDERWIDTH

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
def plot(sids, value, label='', paper=False):

    labels = []
    #colors = []
    colors = [cmap(i) for i in np.linspace(0, 1, len(sids))]
    #c = 0

    first = True

    for sid in sids:

        session_dir = os.path.join(PICKLE_DIR, sid)

        unit_info_df = pd.read_pickle(os.path.join(session_dir, 'unit_info.pkl'))
        pilot_info_df = pd.read_pickle(os.path.join(session_dir, 'pilot_info.pkl'))
        unit_prof_df = pd.read_pickle(os.path.join(session_dir, 'unit_prof.pkl'))
        session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))

        # Legend info
        info = session_info_df.loc[sid]

        # For this call assume that there is only one pilot per session
        resources = get_resources(unit_info_df, pilot_info_df, sid)
        assert len(resources) == 1
        resource_label = resources.values()[0]

        # Get only the entries for this session
        #uf = unit_prof_df[unit_prof_df['sid'] == sid]

        # We sort the units based on the order they arrived at the agent
        #ufs = uf.sort('awo_get_u_pend')

        cores = info['metadata.effective_cores']

        if value == 'sched':
            #
            # Scheduling
            #
            df = unit_prof_df[
                (unit_prof_df.cc_sched >= 0) &
                (unit_prof_df.event == 'advance') &
                (unit_prof_df.sid == sid)
                ][['time', 'cc_sched']]

        elif value == 'exec':
            #
            # Scheduling
            #
            df = unit_prof_df[
                (unit_prof_df.cc_exec >= 0) &
                (unit_prof_df.event == 'advance') &
                (unit_prof_df.sid == sid)
                ][['time', 'cc_exec']]

        else:
            raise Exception("Value %s unknown" % value)

        if 'metadata.barriers' in info and 'barrier_generation' in info['metadata.barriers']:
            barrier = BARRIER_GENERATION
        elif 'metadata.barriers' in info and 'barrier_client_submit' in info['metadata.barriers']:
            barrier = BARRIER_CLIENT_SUBMIT
        elif 'metadata.barriers' in info and 'barrier_agent_launch' in info['metadata.barriers']:
            barrier = BARRIER_AGENT_LAUNCH
        else:
            raise Exception("No barrier info found")

        df.columns = ['time', barrier]
        df['time'] -= df['time'].min()

        if first:
            df_all = df
        else:
            df_all = pd.merge(df_all, df,  on='time', how='outer')

        labels.append(barrier_legend[barrier])
        #colors.append(barrier_colors[barrier])

        first = False

    df_all.set_index('time', inplace=True)
    print df_all.head()
    #df_all.plot(colormap='Paired')
    ax = df_all.plot(color=colors, fontsize=TICK_FONTSIZE, linewidth=LINEWIDTH)

    # For this call assume that there is only one pilot per session
    ppn_values = get_ppn(unit_info_df, pilot_info_df, sid)
    assert len(ppn_values) == 1
    ppn = ppn_values.values()[0]

    legend = mp.pyplot.legend(labels, loc='upper right', fontsize=LEGEND_FONTSIZE, labelspacing=0)
    legend.get_frame().set_linewidth(BORDERWIDTH)
    if not paper:
        mp.pyplot.title("Concurrent number of CUs in stage '%s'.\n"
            "%d generations of a variable number of 'concurrent' CUs of %d core(s) with a %ss payload on a variable core pilot on %s.\n"
            "Constant number of %d sub-agent with %d ExecWorker(s) each.\n"
            "RP: %s - RS: %s - RU: %s"
           % (value,
              info['metadata.generations'], info['metadata.cu_cores'], info['metadata.cu_runtime'], resource_label,
              info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
              info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
              ), fontsize=8)
    mp.pyplot.xlabel("Time (s)", fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylabel("Concurrent Units", fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylim(-10, 1500)
    mi = df_all.index.min()
    ma = df_all.index.max()
    mp.pyplot.xlim(mi - 0.01 * ma, ma * 1.01)
    #ax.get_xaxis().set_ticks([])

    [i.set_linewidth(BORDERWIDTH) for i in ax.spines.itervalues()]
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)

    if paper:
        # width = 3.487
        width = 3.3
        height = width / 1.618
        #height = 2.7
        fig = mp.pyplot.gcf()
        fig.set_size_inches(width, height)
        # fig.subplots_adjust(left=0, right=1, top=1, bottom=1)

        # fig.tight_layout(w_pad=0.0, h_pad=0.0, pad=0.1)
        fig.tight_layout(pad=0.1)
        mp.pyplot.savefig('plot_cc_ew_barriers.pdf')
    else:
        mp.pyplot.savefig('plot6_%s%s.pdf' % (value, label))
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

    session_ids = [
        # Single client barrier
        #"rp.session.ip-10-184-31-85.santcroos.016762.0000", # 1
        # "rp.session.ip-10-184-31-85.santcroos.016760.0007", # 1
        # "rp.session.ip-10-184-31-85.santcroos.016760.0005", # 2
        # "rp.session.ip-10-184-31-85.santcroos.016760.0003", # 4
        # #"rp.session.ip-10-184-31-85.santcroos.016760.0009", # 8
        # "rp.session.ip-10-184-31-85.santcroos.016762.0003", # 8
        # "rp.session.ip-10-184-31-85.santcroos.016760.0004", # 16
        # "rp.session.ip-10-184-31-85.santcroos.016760.0008", # 32
        "rp.session.ip-10-184-31-85.santcroos.016760.0006", # 48

        # Comet generation barrier / exp9
        # "rp.session.ip-10-184-31-85.santcroos.016760.0010", # 24 / 1
        # "rp.session.ip-10-184-31-85.santcroos.016760.0015", # 48 / 2
        # "rp.session.ip-10-184-31-85.santcroos.016760.0014", # 96 / 4
        # "rp.session.ip-10-184-31-85.santcroos.016760.0013", # 192 / 8
        # "rp.session.ip-10-184-31-85.santcroos.016762.0001", # 384 / 16
        # "rp.session.ip-10-184-31-85.santcroos.016760.0016", # 768 / 32
        "rp.session.ip-10-184-31-85.santcroos.016762.0002", # 1152 / 48

        # pilot barrier Comet after scheduler fix2
        # "rp.session.ip-10-184-31-85.santcroos.016747.0011", # 1 node
        # "rp.session.ip-10-184-31-85.santcroos.016747.0009", # 2 node
        # "rp.session.ip-10-184-31-85.santcroos.016747.0008", # 4 node
        #"rp.session.ip-10-184-31-85.santcroos.016747.0010", # 8 nodes
        # "rp.session.ip-10-184-31-85.santcroos.016747.0013", # 16 nodes
        # "rp.session.ip-10-184-31-85.santcroos.016747.0000", # 32 nodes
        "rp.session.ip-10-184-31-85.santcroos.016747.0001", # 48 nodes
    ]

    label = '_10sa_1ew'

    for value in ['exec']:
        plot(session_ids, value, label, paper=True)
