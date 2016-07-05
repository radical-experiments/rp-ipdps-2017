import os
import sys
import time
import glob
import pandas as pd
#from radical.pilot import utils as rpu
from radical.pilot import states as rps

import numpy as np

from common import PICKLE_DIR, get_ppn, get_resources, LABEL_FONTSIZE, LEGEND_FONTSIZE, LINEWIDTH, TICK_FONTSIZE, TITLE_FONTSIZE, BORDERWIDTH

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

# ------------------------------------------------------------------------------
#
def add_frequency(frame, tgt, window, spec):
    """
    This method will add a row 'tgt' to the given data frame, which will contain
    a contain the frequency (1/s) of the events specified in 'spec'.

    We first will filter the given frame by spec, and then apply a rolling
    window over the time column, counting the rows which fall into the window.
    The result is *not* divided by window size, so normalization is up to the
    caller.

    The method looks backwards, so the resulting frequency column contains the
    frequency which applied *up to* that point in time.
    """

    # --------------------------------------------------------------------------
    def _freq(t, _tmp, _window):
        # get sequence of frame which falls within the time window, and return
        # length of that sequence
        return len(_tmp.uid[(_tmp.time > t-_window) & (_tmp.time <= t)])
    # --------------------------------------------------------------------------

    # filter the frame by the given spec
    tmp = frame
    for key,val in spec.iteritems():
        tmp = tmp[tmp[key].isin([val])]
    frame[tgt] = tmp.time.apply(_freq, args=[tmp, window])

    # frame[tgt] = frame[tgt].fillna(0)

    return frame

###############################################################################
#
# TODO: add concurrent CUs on right axis
def plot(sid, values, label='', paper=False, window=1.0, plot_mean=False):

    labels = []
    means = {}

    colors = [cmap(i) for i in np.linspace(0, 1, len(values))]
    c = 0

    first = True

    if sid.startswith('rp.session'):
        rp = True
    else:
        rp = False

    session_dir = os.path.join(PICKLE_DIR, sid)

    pilot_info_df = pd.read_pickle(os.path.join(session_dir, 'pilot_info.pkl'))
    session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))
    unit_info_df = pd.read_pickle(os.path.join(session_dir, 'unit_info.pkl'))
    unit_prof_df = pd.read_pickle(os.path.join(session_dir, 'unit_prof.pkl'))

    # Legend info
    info = session_info_df.loc[sid]

    if rp:
        # For this call assume that there is only one pilot per session
        resources = get_resources(unit_info_df, pilot_info_df, sid)
        assert len(resources) == 1
        resource_label = resources.values()[0]
    else:
        resource_label = "bogus"

    # Get only the entries for this session
    #uf = unit_prof_df[unit_prof_df['sid'] == sid]

    # We sort the units based on the order they arrived at the agent
    #ufs = uf.sort('awo_get_u_pend')

    cores = info['metadata.effective_cores']

    for value in values:

        if 'stagein_freq' == value:
            spec = {'state': rps.AGENT_STAGING_INPUT, 'event': 'advance'}

        elif 'sched_freq' == value:
            spec = {'state': rps.EXECUTING_PENDING, 'event': 'advance'}

        elif 'exec_freq' == value:
            spec = {'state' : rps.EXECUTING, 'event' : 'advance'}

        elif 'fork_freq' == value:
            spec = {'info' : 'aec_start_script'}

        elif 'exit_freq' == value:
            spec = {'info' : 'aec_after_exec'}

        elif 'stageout_pend_freq' == value:
            spec = {'state' : rps.AGENT_STAGING_OUTPUT_PENDING, 'event' : 'advance'}

        elif 'stageout_freq' == value:
            spec = {'state': rps.AGENT_STAGING_OUTPUT, 'event': 'advance'}

        else:
            raise Exception("Value %s unknown" % value)

        #print unit_prof_df.head()

        add_frequency(unit_prof_df, value, window, spec)
        df = unit_prof_df[
            (unit_prof_df[value] >= 0) &
            #(unit_prof_df.event == 'advance') &
            (unit_prof_df.sid == sid)
            ][['time', value]]
        means[value] = df[value].mean()

        #df.columns = ['time', value]
        #df['time'] -= df['time'].min()
        df.time = pd.to_datetime(df.time, unit='s')
        df.set_index('time', inplace=True, drop=True, append=False)

        #print ("Head of %s before resample" % value)
        #print df.head()

        def _mean(array_like):
            return np.mean(array_like)/window
        df = df.resample('%dL' % int(1000.0*window), how=_mean)[value]
        df = df.fillna(0)

        #print ("Head of %s after resample" % value)
        #print df.head()
        if first:
            df_all = df
        else:

            #df_all = pd.merge(df_all, df,  on='time', how='outer')
            #df_all = pd.merge(df_all, df,  on='time')
            #df_all = pd.merge(df_all, df)
            df_all = pd.concat([df_all, df], axis=1)
            #df_all.append(df)

        #print ("Head of df_all")
        #print df_all.head()

        if value == 'exec_freq':
            labels.append("Launching")
        elif value == 'sched_freq':
            labels.append("Scheduling")
        elif value == 'fork_freq':
            labels.append("Forking")
        elif value == 'stageout_pend_freq':
            labels.append("Completing")
        else:
            labels.append("%s" % value)

        first = False

        # df.plot(drawstyle='steps-pre')

    c = 0
    for value in values:
        mean = df_all[value].mean()
        print "%s mean: %f" % (value, mean)
        # df_all['mean_%s' % value] = mean
        #labels.append("Mean %s" % value)
    print 'means:', means

    my_colors = colors
    if plot_mean:
        my_colors *= 2

    my_styles = []
    for x in range(len(values)):
        my_styles.append('-')
    if plot_mean:
        for x in range(len(values)):
            my_styles.append('--')

    #df_all.set_index('time', inplace=True)
    #print df_all.head(500)
    #df_all.plot(colormap='Paired')
    #df_all.plot(drawstyle='steps-post')
    ax = df_all.plot(drawstyle='steps-pre', color=my_colors, style=my_styles, linewidth=LINEWIDTH, fontsize=TICK_FONTSIZE)
    # df_all.plot(drawstyle='steps')
    #df_all.plot()

    # Vertial reference
    # x_ref = info['metadata.generations'] * info['metadata.cu_runtime']
    # mp.pyplot.plot((x_ref, x_ref),(0, 1000), 'k--')
    # labels.append("Optimal")

    mp.pyplot.legend(labels, loc='upper right', fontsize=LEGEND_FONTSIZE, labelspacing=0)
    if not paper:
        mp.pyplot.title("Rate of various components: %s'.\n"
                "%d generations of %d 'concurrent' units of %d core(s) with a %ss payload on a variable core pilot on %s.\n"
                "Constant number of %d sub-agent with %d ExecWorker(s) each.\n"
                "RP: %s - RS: %s - RU: %s"
               % (values,
                  info['metadata.generations'], cores, info['metadata.cu_cores'], info['metadata.cu_runtime'], resource_label,
                  info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
                  info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
                  ), fontsize=TITLE_FONTSIZE)
    mp.pyplot.xlabel("Time (s)", fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylabel("Rate (Unit/s)", fontsize=LABEL_FONTSIZE)
    #mp.pyplot.ylim(-1, 400)
    #mp.pyplot.xlim(-1,)
    #mp.pyplot.xlim(['1/1/2000', '1/1/2000'])
    #mp.pyplot.xlim('03:00', '04:00')
    #mp.pyplot.xlim(380, 400)
    #mp.pyplot.xlim(675, 680)
    #ax.get_xaxis().set_ticks([])
    # ax.set_yscale('log', basey=10)

    #mp.pyplot.xlim((291500.0, 1185200.0))
    #mp.pyplot.xlim((474000.0, 2367600.0))

    print "xlim:", ax.get_xlim()

    [i.set_linewidth(BORDERWIDTH) for i in ax.spines.itervalues()]
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)

    #width = 3.487
    width = 3.3
    height = width / 1.618
    # height = 2.7
    fig = mp.pyplot.gcf()
    fig.set_size_inches(width, height)
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=1)

    #fig.tight_layout(w_pad=0.0, h_pad=0.0, pad=0.1)
    fig.tight_layout(pad=0.1)
    #fig.tight_layout()

    mp.pyplot.savefig('plot_more_rates-%s.pdf' % sid)
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

    sids = [
    # BW
    # "rp.session.radical.marksant.016855.0006", # 1024
    # "rp.session.radical.marksant.016855.0008", # 2048
    # session_id = "rp.session.radical.marksant.016855.0005" # 4096
    # session_id = "rp.session.radical.marksant.016855.0007" # 8192

    # Stampede
    # "rp.session.radical.marksant.016860.0037",
    # "rp.session.radical.marksant.016860.0014",
    # session_id = "rp.session.radical.marksant.016861.0008" # 4096

    # Stampede, generation barrier
    # session_id = "rp.session.radical.marksant.016861.0006" # 256
    # session_id = "rp.session.radical.marksant.016861.0007" # 4096


    #
    # ORTE ONLY
    #

    # Stampede
    # session_id = 'mw.session.login3.stampede.tacc.utexas.edu.marksant.016863.0010'

    # Blue Waters
    # 'mw.session.h2ologin3.marksant.016863.0003' # 64 x 3
    # 'mw.session.h2ologin2.marksant.016863.0006', # 4096 x 3
    # 'mw.session.nid25431.marksant.016863.0009' # 8192x3
    # 'mw.session.nid25263.marksant.016864.0000' # 8192 cores 10k x
    # 'mw.session.login3.stampede.tacc.utexas.edu.marksant.016864.0001' # stampede 8k x 3
    # 'mw.session.login3.stampede.tacc.utexas.edu.marksant.016864.0002' # 100k 0s interrupted at ~75k

    # 'rp.session.radical.marksant.016864.0001' # stampede, 8k, 5gen

    # 'mw.session.netbook.mark.016865.0044',
    # 'rp.session.radical.marksant.016865.0039' # 8k
    # 'rp.session.radical.marksant.016865.0040'#  4k - 512s - 3 gen
    # 'rp.session.radical.marksant.016865.0002'#  4k - 512s - 3 gen
    # 'rp.session.radical.marksant.016861.0007'#  4k - 512s - 3 gen


    # 'mw.session.nid25429.marksant.016865.0005', # 4k

    # 'rp.session.radical.marksant.016868.0011'#  4k - 60s - nogen
    # 'rp.session.radical.marksant.016868.0015'#  4k - 60s - nogen

    # "mw.session.nid25337.marksant.016869.0007", # 4k, 3 gen, no bar, 64s ### PAPER? ###


    #  "rp.session.radical.marksant.016870.0004"
    #     'mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0008', # 32s
    #     'mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0009', # 16s
    #     'mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0010', # 16s

        #'rp.session.radical.marksant.016884.0063',
        #'rp.session.radical.marksant.016885.0011', # stampede ssh 3 gen 8k 64s
        # 'rp.session.radical.marksant.016868.0010', # bw 4k 60s 3gen gen_bar
        'rp.session.radical.marksant.016868.0011', # bw 4k 60s 3gen no_bar ### USED IN PAPER ###
        #'rp.session.radical.marksant.016868.0017', # bw 8k 60s 3gen gen_bar
        #'rp.session.radical.marksant.016868.0014', # bw 8k 60s 3gen no_bar

        # 'rp.session.radical.marksant.016884.0022',
        # 'rp.session.radical.marksant.016895.0006',

        # Stampede SSH 64s
        # 'rp.session.radical.marksant.016895.0000',  # 16
        # 'rp.session.radical.marksant.016895.0005',  # 32
        # 'rp.session.radical.marksant.016895.0003',  # 64
        # 'rp.session.radical.marksant.016895.0004',  # 256
        # 'rp.session.radical.marksant.016884.0040',  # 512
        # 'rp.session.radical.marksant.016884.0058',  # 128
        # 'rp.session.radical.marksant.016884.0022',  # 1024
        # 'rp.session.radical.marksant.016895.0001',  # 2048
        # 'rp.session.radical.marksant.016895.0006',  # 4096
        # 'rp.session.radical.marksant.016895.0007',  # 8192

        # 'rp.session.radical.marksant.016884.0063',
        # 'rp.session.radical.marksant.016884.0064',
        # 'rp.session.radical.marksant.016884.0010',
        # 'rp.session.radical.marksant.016884.0058',
        # 'rp.session.radical.marksant.016884.0062',
        # 'rp.session.radical.marksant.016884.0057',
        # 'rp.session.radical.marksant.016884.0059',
        # 'rp.session.radical.marksant.016884.0040',
        # 'rp.session.radical.marksant.016884.0037',
        # 'rp.session.radical.marksant.016884.0060',
        # 'rp.session.radical.marksant.016884.0022',
        # 'rp.session.radical.marksant.016884.0021',
        # 'rp.session.radical.marksant.016884.0017',
        # 'rp.session.radical.marksant.016884.0018',
        # 'rp.session.radical.marksant.016884.0007',
        # 'rp.session.radical.marksant.016884.0006',
        # 'rp.session.radical.marksant.016884.0005',

        'rp.session.radical.marksant.016929.0000',  # 4k, 4 EW MOM NODE

        'rp.session.radical.marksant.016983.0046',
    ]
    label = ''
    #values = ['stagein_freq']
    #values = ['sched_freq']
    # values = ['exec_freq']
    # #values = ['fork_freq']
    # values = ['exec_freq', 'fork_freq']
    #values = ['exit_freq']
    # #values = ['stageout_pend_freq']
    # #values = ['stageout_freq']
    #values = ['stagein_freq', 'sched_freq', 'exec_freq', 'fork_freq', 'exit_freq', 'stageout_pend_freq', 'stageout_freq']
    #values = ['sched_freq', 'exec_freq', 'fork_freq', 'exit_freq', 'stageout_pend_freq']
    #values = ['sched_freq', 'exec_freq', 'exit_freq']
    # values = ['sched_freq', 'exec_freq', 'fork_freq', 'exit_freq']

    values = ['sched_freq', 'exec_freq', 'stageout_pend_freq']

    #values = ['exec_freq', 'fork_freq', 'stageout_pend_freq']
    # values = ['exec_freq', 'stageout_pend_freq']

    for sid in sids:
        plot(sid, values, label, paper=True, window=1, plot_mean=False)
