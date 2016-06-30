import os
import sys
import time
import glob
import pandas as pd
#from radical.pilot import utils as rpu
from radical.pilot import states as rps

import numpy as np

from common import PICKLE_DIR, get_ppn, get_resources, LEGEND_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE, LINEWIDTH, TICK_FONTSIZE, BORDERWIDTH

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
def plot(sids, value, label='', paper=False, window=1.0, plot_mean=False, compare=None, micro=False):

    labels = []
    colors = [cmap(i) for i in np.linspace(0, 1, len(sids))]

    first = True

    values = []

    counter = 0

    for sid in sids:

        print "sid: %s" % sid

        if sid.startswith('rp.session'):
            rp = True
        else:
            rp = False

        session_dir = os.path.join(PICKLE_DIR, sid)

        unit_info_df = pd.read_pickle(os.path.join(session_dir, 'unit_info.pkl'))
        pilot_info_df = pd.read_pickle(os.path.join(session_dir, 'pilot_info.pkl'))
        session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))
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

        if not compare:
            raise Exception("Need to specify 'compare' parameter!")
        elif 'metadata.%s' % compare in info:
             metric = info['metadata.%s' % compare]
        else:
            #raise Exception("'%s' not found in info!" % compare)
            metric = counter
            counter += 1
        values.append(metric)

        if value == 'sched_freq':
            plot_type = 'sched'
            plot_label = 'Scheduling'

            spec = {'state': rps.ALLOCATING, 'event' : 'advance'}
            add_frequency(unit_prof_df, 'sched_freq', window, spec)
            print unit_prof_df.state.unique()

            #
            # scheduling frequency
            #
            df = unit_prof_df[
                (unit_prof_df.sched_freq >= 0) &
                (unit_prof_df.event == 'advance') &
                (unit_prof_df.sid == sid)
                ][['time', 'sched_freq']]

        elif value == 'exec_freq':
            plot_type = 'exec'
            plot_label = 'Executing'

            spec = {'state': 'Executing', 'event' : 'advance'}
            add_frequency(unit_prof_df, 'exec_freq', window, spec)

            #
            # feq
            #
            df = unit_prof_df[
                (unit_prof_df.exec_freq >= 0) &
                (unit_prof_df.event == 'advance') &
                (unit_prof_df.sid == sid)
                ][['time', 'exec_freq']]

        elif 'fork_freq' == value:
            spec = {'info': 'aec_start_script'}
            add_frequency(unit_prof_df, value, window, spec)

            #
            # fork - start_script
            #
            df = unit_prof_df[
                (unit_prof_df[value] >= 0) &
                #(unit_prof_df.event == 'advance') &
                (unit_prof_df.sid == sid)
                ][['time', value]]

        elif value == 'done_freq':

            spec = {'state' : rps.AGENT_STAGING_OUTPUT_PENDING, 'event' : 'advance'}
            add_frequency(unit_prof_df, 'done_freq', 1, spec)

            #
            # feq
            #
            df = unit_prof_df[
                (unit_prof_df.done_freq >= 0) &
                (unit_prof_df.event == 'advance') &
                (unit_prof_df.sid == sid)
                ][['time', 'done_freq']]

        else:
            raise Exception("Value %s unknown" % value)

        df.columns = ['time', metric]
        df['time'] -= df['time'].min()
        df.time = pd.to_datetime(df.time, unit='s')
        df.set_index('time', inplace=True)


        def _mean(array_like):
            return np.mean(array_like)/window
        df = df.resample('%dL' % int(1000.0*window), how=_mean)
        df = df.fillna(0)

        print df.head()

        if first:
            df_all = df
        else:
            #df_all = pd.merge(df_all, df,  on='time', how='outer')
            #df_all = pd.merge(df_all, df, how='outer')
            df_all = pd.concat([df_all, df], axis=1)

        labels.append("%d" % metric)
        #labels.append("%d - %s" % (cores, 'RP' if rp else 'ORTE'))
        #labels.append(sid[-4:])

        first = False


    c = 0
    for value in values:
        mean = df_all[value].mean()
        stddev = df_all[value].std(ddof=0)
        print "Mean value for %d: %f (%f)" % (value, mean, stddev)
        if plot_mean:
            df_all['mean_%s' % value] = mean
        # labels.append("Mean %s" % value)
    my_colors = colors
    my_styles = []
    for x in range(len(values)):
        my_styles.append('-')
    if plot_mean:
        my_colors *= 2
        for x in range(len(values)):
            my_styles.append('--')


    #df_all.set_index('time', inplace=True)
    # print df_all.head(500)
    #df_all.plot(colormap='Paired')
    #df_all.plot(drawstyle='steps-post')
    ax = df_all.plot(color=my_colors, style=my_styles, drawstyle='steps-pre', fontsize=TICK_FONTSIZE, linewidth=LINEWIDTH)
    # df_all.plot(drawstyle='steps')
    # df_all.plot()


    mp.pyplot.legend(labels, loc='upper right', fontsize=LEGEND_FONTSIZE, labelspacing=0)
    if not paper:
        mp.pyplot.title("Rate of CUs transitioning in stage '%s'.\n"
                "%d generations of a variable number of 'concurrent' CUs of %d core(s) with a %ss payload on a variable core pilot on %s.\n"
                "Constant number of %d sub-agent with %d ExecWorker(s) each.\n"
                "RP: %s - RS: %s - RU: %s"
               % (value,
                  info['metadata.generations'], info['metadata.cu_cores'], info['metadata.cu_runtime'], resource_label,
                  info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
                  info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
                  ), fontsize=TITLE_FONTSIZE)
    mp.pyplot.xlabel("Time (s)", fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylabel("%s Rate (Unit/s)" % plot_label, fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylim(0,)
    #mp.pyplot.xlim('0:00', '0:40')
    #mp.pyplot.xlim(380, 400)
    #mp.pyplot.xlim(675, 680)
    #ax.get_xaxis().set_ticks([])

    from matplotlib.dates import YearLocator, MonthLocator, DateFormatter, SecondLocator

    #second_fmt = DateFormatter('%S')
    # second_loc = SecondLocator(bysecond=range(0, 300, 10))
    # ax.xaxis.set_minor_formatter(second_fmt)
    # ax.xaxis.set_minor_locator(second_loc)
    # ax.xaxis.set_major_formatter(second_fmt)
    # ax.xaxis.set_major_locator(second_loc)
    # second_loc.set_axis(ax.xaxis)  # Have to manually make this call and the one below.
    # second_loc.refresh()

    # secondsFmt = DateFormatter('%s')
    s = SecondLocator()
    ax.xaxis.set_major_locator(s)
    # ax.xaxis.set_minor_locator(SecondLocator())
    # ax.xaxis.set_major_formatter(secondsFmt)

    #ax.xaxis.set_major_locator(years)
    #ax.xaxis.set_major_formatter(yearsFmt)
    #ax.xaxis.set_minor_locator(months)

    ax.autoscale_view()

    [i.set_linewidth(BORDERWIDTH) for i in ax.spines.itervalues()]
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)

    #width = 3.487
    width = 3.3
    height = width / 1.618
    #height = 2.5
    fig = mp.pyplot.gcf()
    fig.set_size_inches(width, height)
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=1)
    fig.tight_layout(pad=0.1)


    mp.pyplot.savefig('plot_rate_%s%s_%s_%dgen.pdf' % ('micro_' if micro else '', value, resource_label, info['metadata.generations']))
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

        # BW
        # "rp.session.radical.marksant.016855.0006", # 1024
        # "rp.session.radical.marksant.016855.0008", # 2048
        # "rp.session.radical.marksant.016855.0005", # 4096
        # "rp.session.radical.marksant.016855.0007", # 8192

        # Stampede
        # "rp.session.radical.marksant.016860.0037",
        # "rp.session.radical.marksant.016860.0014",
        # "rp.session.radical.marksant.016861.0008", # 4096

        # Stampede, generation barrier
        # "rp.session.radical.marksant.016861.0006", # 256
        # "rp.session.radical.marksant.016861.0007", # 4096
        #"rp.session.netbook.test1"
        # 'mw.session.netbook.mark.016863.0014',

        # Stampede ORTE
        # 'mw.session.login3.stampede.tacc.utexas.edu.marksant.016863.0005',
        # 'mw.session.login3.stampede.tacc.utexas.edu.marksant.016863.0007',

        # 'mw.session.h2ologin3.marksant.016863.0001',

        # 'mw.session.h2ologin2.marksant.016863.0006', # ORTE only 4096x3
        # 'mw.session.h2ologin2.marksant.016863.0007', # ORTE only 4096x3
        # 'mw.session.nid25431.marksant.016863.0009', # ORTE only 8192x3

        #'mw.session.netbook.mark.016865.0041',
        # 'rp.session.radical.marksant.016865.0039',
        # 'rp.session.radical.marksant.016865.0040', # 4k
        # 'mw.session.nid25429.marksant.016865.0005' # 4k

        # 'mw.session.login3.stampede.tacc.utexas.edu.marksant.016863.0010'

        # 'rp.session.radical.marksant.016861.0007', # 4
        # 'rp.session.radical.marksant.016865.0002', # 4k
        # 'rp.session.radical.marksant.016868.0011', # 4k
        # 'rp.session.radical.marksant.016869.0000', # 4k

        # "rp.session.radical.marksant.016868.0016",
        # # "rp.session.radical.marksant.016868.0013",
        # # "rp.session.radical.marksant.016868.0015",
        # "rp.session.radical.marksant.016868.0012",

        #"mw.session.nid25337.marksant.016869.0010", # 4k 1gen 64s

        # "mw.session.nid25337.marksant.016869.0007",

        # stampede 4k
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0006", # 64s
        # # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0007", # 64s 3 gens
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0008", # 32s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0009", # 16s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0010", # 8s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0011", # 4s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0012", # 2s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0013", # 1s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0014", # 0s

        # bw - 8k
        # 'mw.session.nid25254.marksant.016869.0011',
        # 'mw.session.nid25254.marksant.016869.0012',
        # 'mw.session.nid25254.marksant.016869.0013',
        # 'mw.session.nid25254.marksant.016869.0014',
        # 'mw.session.nid25254.marksant.016869.0015',
        # 'mw.session.nid25254.marksant.016869.0016',
        # 'mw.session.nid25254.marksant.016869.0017',

        #'rp.session.radical.marksant.016917.0020'
        # 'rp.session.radical.marksant.016918.0005', # 512 cores, 64s

        # Micro Benchmarks ORTELIB BW Exec
        # 'rp.session.radical.marksant.016918.0018', # 32 cores, 512s
        # 'rp.session.radical.marksant.016918.0008', # 64 cores, 512s
        # 'rp.session.radical.marksant.016918.0014', # 128 cores, 512s
        # 'rp.session.radical.marksant.016918.0010', # 256 cores, 512s
        # 'rp.session.radical.marksant.016918.0022', # 512 cores, 512s
        # # 'rp.session.radical.marksant.016918.0016', # 1024 cores, 512s
        # 'rp.session.radical.marksant.016918.0006', # 2048 cores, 512s
        # 'rp.session.radical.marksant.016918.0012', # 4096 cores, 512s
        # 'rp.session.radical.marksant.016918.0020', # 8192 cores, 512s

        # 'rp.session.radical.marksant.016918.0019', # 32 cores, 0 s
        # 'rp.session.radical.marksant.016918.0009', # 64 cores, 0s
        # 'rp.session.radical.marksant.016918.0015', # 128 cores, 0s
        # 'rp.session.radical.marksant.016918.0011', # 256 cores, 0s
        # # 'rp.session.radical.marksant.016918.0023', # 512 cores, 0s
        # 'rp.session.radical.marksant.016918.0017', # 1024 cores, 0s
        # 'rp.session.radical.marksant.016918.0007', # 2048 cores, 0s
        # 'rp.session.radical.marksant.016918.0013', # 4096 cores, 0s
        # 'rp.session.radical.marksant.016918.0021', # 8192 cores, 0s

        # Micro Benchmarks ORTELIB BW Scheduling w/ unscheduling
        # 'rp.session.radical.marksant.016923.0000', # 32
        # 'rp.session.radical.marksant.016923.0002', # 64
        # 'rp.session.radical.marksant.016923.0005', # 128
        # 'rp.session.radical.marksant.016923.0001', # 256
        # 'rp.session.radical.marksant.016923.0004', # 512
        # 'rp.session.radical.marksant.016923.0003', # 1024
        # 'rp.session.radical.marksant.016923.0006', # 2048
        # 'rp.session.radical.marksant.016918.0032', # 4096 cores
        # 'rp.session.radical.marksant.016918.0024', # 8192 cores

        # Micro Benchmarks ORTELIB BW Scheduling w/o unscheduling
        # 'rp.session.radical.marksant.016923.0034', # 32
        # 'rp.session.radical.marksant.016923.0037', # 64
        # 'rp.session.radical.marksant.016923.0035', # 128
        # 'rp.session.radical.marksant.016923.0032', # 256
        # 'rp.session.radical.marksant.016923.0038', # 512
        # 'rp.session.radical.marksant.016923.0033', # 1024
        # 'rp.session.radical.marksant.016923.0036', # 2048
        # 'rp.session.radical.marksant.016923.0031', # 4096
        # 'rp.session.radical.marksant.016923.0039', # 8192

        # Andre's Microbenchmark experiments on BW with ORTE, 1k
        # 'rp.session.cameo.merzky.016744.0082', # 1:1
        # 'rp.session.cameo.merzky.016746.0067', # 2:1
        # 'rp.session.cameo.merzky.016747.0046', # 4:1
        # 'rp.session.cameo.merzky.016747.0117', # 8:1

        # Micro Benchmark ORTE CLI Executing on BW, 4k, 0s
        # 'rp.session.radical.marksant.016927.0013', # 1 SA
        # 'rp.session.radical.marksant.016927.0012', # 2 SA  ### PAPER UNIT DURATION PLOT ###
        # 'rp.session.radical.marksant.016927.0014', # 4 SA
        # 'rp.session.radical.marksant.016927.0015', # 8 SA

        # Micro Benchmark ORTE CLI Executing on BW, 4k, 300s
        # 'rp.session.radical.marksant.016927.0017', # 2 SA  ### PAPER UNIT DURATION PLOT ###

        # Micro Benchmark ORTE LIB Executing on BW, 4k, 0s ### PAPER ###
        # 'rp.session.radical.marksant.016927.0020', # 1 SA
        # 'rp.session.radical.marksant.016927.0022', # 2 SA
        # 'rp.session.radical.marksant.016927.0021', # 4 SA
        # 'rp.session.radical.marksant.016927.0019', # 8 SA
        # 'rp.session.radical.marksant.016927.0023', # 16 SA

        # Micro Benchmark ORTE LIB Executing on BW, 4k, 300s
        # 'rp.session.radical.marksant.016927.0024', # 1 SA
        # 'rp.session.radical.marksant.016927.0025', # 2 SA
        # 'rp.session.radical.marksant.016927.0027', # 4 SA
        # 'rp.session.radical.marksant.016927.0026', # 8 SA

        # Micro Benchmark ORTE LIB Executing on BW, 4k, 0s, 1 SA NODE ### PAPER ###
        'rp.session.radical.marksant.016927.0029', # 1 EW
        'rp.session.radical.marksant.016927.0030', # 2 EW
        'rp.session.radical.marksant.016927.0031', # 4 EW
        'rp.session.radical.marksant.016927.0028', # 8 EW

        # Micro Benchmark ORTE CLI Executing on BW, 4k, 0s, 1 SA
        # 'rp.session.radical.marksant.016927.0032',
        # 'rp.session.radical.marksant.016927.0033',
        # 'rp.session.radical.marksant.016927.0034',
        # 'rp.session.radical.marksant.016927.0035',

        # Micro Benchmark ORTE LIB Executing on BW, 4k, 0s, 1 SA, mom node ### PAPER ###
        # 'rp.session.radical.marksant.016927.0037', # 1 EW
        # 'rp.session.radical.marksant.016927.0040', # 2 EW
        # 'rp.session.radical.marksant.016927.0039', # 4 EW
        # 'rp.session.radical.marksant.016927.0038', # 8 EW
        # 'rp.session.radical.marksant.016927.0036', # 16 EW

        # Exp F, finding scale
        # 'rp.session.radical.marksant.016928.0000', # 8k, 512s
        # 'rp.session.radical.marksant.016928.0001', # 32k, 512s
        # 'rp.session.radical.marksant.016928.0002', # 16k, 512s
        # 'rp.session.radical.marksant.016928.0003', # 32k, 1024s
        # 'rp.session.radical.marksant.016928.0004', # 64k, 2048s


    ]


    label = ''

    # comp = 'cu_runtime'
    # comp = 'num_sub_agents'
    # comp = 'effective_cores'
    comp = 'num_exec_instances_per_sub_agent'

    # for value in ['sched_freq']:
    for value in ['exec_freq']:
    #for value in ['fork_freq']:
    #for value in ['done_freq']:
        plot(session_ids, value, label, paper=True, window=1, plot_mean=True, compare=comp, micro=True)
