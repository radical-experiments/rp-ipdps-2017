import os
import sys
import time
import glob
import pandas as pd
#from radical.pilot import utils as rpu
from radical.pilot import states as rps

import numpy as np

from common import PICKLE_DIR, get_ppn, get_resources, BARRIER_FONTSIZE, TITLE_FONTSIZE

# Global Pandas settings
pd.set_option('display.width', 180)
pd.set_option('io.hdf.default_format','table')

import matplotlib as mp


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
def plot(sid, values, label='', paper=False, window=1.0):

    labels = []

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

        print unit_prof_df.head()

        add_frequency(unit_prof_df, value, window, spec)
        df = unit_prof_df[
            (unit_prof_df[value] >= 0) &
            #(unit_prof_df.event == 'advance') &
            (unit_prof_df.sid == sid)
            ][['time', value]]

        #df.columns = ['time', value]
        #df['time'] -= df['time'].min()
        df.time = pd.to_datetime(df.time, unit='s')
        df.set_index('time', inplace=True, drop=True, append=False)

        print ("Head of %s before resample" % value)
        print df.head()

        def _mean(array_like):
            return np.mean(array_like)/window
        df = df.resample('%dL' % int(1000.0*window), how=_mean)[value]
        df = df.fillna(0)

        print ("Head of %s after resample" % value)
        print df.head()

        if first:
            df_all = df
        else:
            print ("Head of df_all")
            print df_all.head()

            #df_all = pd.merge(df_all, df,  on='time', how='outer')
            #df_all = pd.merge(df_all, df,  on='time')
            #df_all = pd.merge(df_all, df)
            df_all = pd.concat([df_all, df], axis=1)
            #df_all.append(df)

        labels.append("%s" % value)

        first = False

        # df.plot(drawstyle='steps-pre')

    #df_all.set_index('time', inplace=True)
    #print df_all.head(500)
    #df_all.plot(colormap='Paired')
    #df_all.plot(drawstyle='steps-post')
    ax = df_all.plot(drawstyle='steps-pre')
    # df_all.plot(drawstyle='steps')
    #df_all.plot()

    # Vertial reference
    # x_ref = info['metadata.generations'] * info['metadata.cu_runtime']
    # mp.pyplot.plot((x_ref, x_ref),(0, 1000), 'k--')
    # labels.append("Optimal")

    mp.pyplot.legend(labels, loc='upper right', fontsize=BARRIER_FONTSIZE, labelspacing=0)
    if not paper:
        mp.pyplot.title("Rate of various components: %s'.\n"
                "%d generations of %d 'concurrent' CUs of %d core(s) with a %ss payload on a variable core pilot on %s.\n"
                "Constant number of %d sub-agent with %d ExecWorker(s) each.\n"
                "RP: %s - RS: %s - RU: %s"
               % (values,
                  info['metadata.generations'], cores, info['metadata.cu_cores'], info['metadata.cu_runtime'], resource_label,
                  info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
                  info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
                  ), fontsize=TITLE_FONTSIZE)
    mp.pyplot.xlabel("Time (s)", fontsize=BARRIER_FONTSIZE)
    mp.pyplot.ylabel("Rate (CU/s)", fontsize=BARRIER_FONTSIZE)
    #mp.pyplot.ylim(-1, 400)
    #mp.pyplot.xlim(-1,)
    #mp.pyplot.xlim(['1/1/2000', '1/1/2000'])
    #mp.pyplot.xlim('03:00', '04:00')
    #mp.pyplot.xlim(380, 400)
    #mp.pyplot.xlim(675, 680)
    #ax.get_xaxis().set_ticks([])

    #mp.pyplot.xlim((60000.0, 200000.0))

    print "xlim:", ax.get_xlim()
    mp.pyplot.savefig('plot_more_rates.pdf')
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
    # session_id = 'mw.session.h2ologin3.marksant.016863.0003' # 64 x 3
    # session_id = 'mw.session.h2ologin2.marksant.016863.0006' # 4096 x 3
    # session_id = 'mw.session.nid25431.marksant.016863.0009' # 8192x3
    # session_id = 'mw.session.nid25263.marksant.016864.0000' # 8192 cores 10k x
    # session_id = 'mw.session.login3.stampede.tacc.utexas.edu.marksant.016864.0001' # stampede 8k x 3
    # session_id = 'mw.session.login3.stampede.tacc.utexas.edu.marksant.016864.0002' # 100k 0s interrupted at ~75k

    # session_id = 'rp.session.radical.marksant.016864.0001' # stampede, 8k, 5gen

    # session_id = 'mw.session.netbook.mark.016865.0044'
    # session_id = 'rp.session.radical.marksant.016865.0039' # 8k
    session_id = 'rp.session.radical.marksant.016865.0040'#  4k

    # session_id = 'mw.session.nid25429.marksant.016865.0005' 4k

    label = ''

    #values = ['stagein_freq']
    #values = ['sched_freq']
    #values = ['exec_freq']
    #values = ['fork_freq']
    #values = ['exit_freq']
    #values = ['stageout_pend_freq']
    #values = ['stageout_freq']
    values = ['stagein_freq', 'sched_freq', 'exec_freq', 'fork_freq', 'exit_freq', 'stageout_pend_freq', 'stageout_freq']

    plot(session_id, values, label, paper=False, window=1)
