import os
import sys
import time
import glob
import pandas as pd
#from radical.pilot import utils as rpu
from radical.pilot import states as rps

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

    #frame[tgt] = frame[tgt].fillna(0)

    return frame

###############################################################################
#
# TODO: add concurrent CUs on right axis
def plot(sids, value, label='', paper=False):

    labels = []

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
        #cores = 32

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
            # Executing
            #
            df = unit_prof_df[
                (unit_prof_df.cc_exec >= 0) &
                (unit_prof_df.event == 'advance') &
                (unit_prof_df.sid == sid)
                ][['time', 'cc_exec']]

        elif value == 'exec_freq':

            spec = {'state' : 'Executing', 'event' : 'advance'}
            add_frequency(unit_prof_df, 'exec_freq', 1, spec)

            #
            # feq
            #
            df = unit_prof_df[
                (unit_prof_df.exec_freq >= 0) &
                (unit_prof_df.event == 'advance') &
                (unit_prof_df.sid == sid)
                ][['time', 'exec_freq']]

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


        df.columns = ['time', cores]
        #df['time'] -= df['time'].min()

        from scipy.interpolate import spline
        from scipy.interpolate import interp1d
        import numpy as np

        #xnew = np.linspace(df['time'].min(), df['time'].max(), 1000)
        #power_smooth = spline(df['time'], df[cores], xnew)
        #power_smooth = interp1d(df['time'], df[cores])

        #df = df.interpolate(method='cubic')


        if first:
            df_all = df
        else:
            df_all = pd.merge(df_all, df,  on='time', how='outer')

        #labels.append("Cores: %d" % cores)
        labels.append("%d" % cores)

        first = False


    df_all.set_index('time', inplace=True)
    print df_all.head(500)
    #df_all.plot(colormap='Paired')
    #df_all.plot(drawstyle='steps-post')
    df_all.plot(drawstyle='steps')
    #df_all.plot()
    #from matplotlib import pyplot
    #pyplot.plot(xnew, power_smooth)


    # Vertial reference
    # x_ref = info['metadata.generations'] * info['metadata.cu_runtime']
    # mp.pyplot.plot((x_ref, x_ref),(0, 1000), 'k--')
    # labels.append("Optimal")

    mp.pyplot.legend(labels, loc='upper right', fontsize=BARRIER_FONTSIZE, labelspacing=0)
    if not paper:
        mp.pyplot.title("Concurrent number of CUs in stage '%s'.\n"
                "%d generations of a variable number of 'concurrent' CUs of %d core(s) with a %ss payload on a variable core pilot on %s.\n"
                "Constant number of %d sub-agent with %d ExecWorker(s) each.\n"
                "RP: %s - RS: %s - RU: %s"
               % (value,
                  info['metadata.generations'], info['metadata.cu_cores'], info['metadata.cu_runtime'], resource_label,
                  info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
                  info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
                  ), fontsize=TITLE_FONTSIZE)
    mp.pyplot.xlabel("Time (s)", fontsize=BARRIER_FONTSIZE)
    mp.pyplot.ylabel("# Concurrent Compute Units", fontsize=BARRIER_FONTSIZE)
    # mp.pyplot.ylim(0, 200)
    #mp.pyplot.xlim(2180, 2185)
    mp.pyplot.xlim(380, 400)
    #mp.pyplot.xlim(675, 680)
    #ax.get_xaxis().set_ticks([])

    mp.pyplot.savefig('plot_launch_rate.pdf')
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

        # Stampede, generation barrier
        "rp.session.radical.marksant.016861.0006", # 256
        #"rp.session.radical.marksant.016861.0007", # 4096
    ]

    label = ''

    for value in ['exec_freq']:
    #for value in ['done_freq']:
    #for value in ['exec']:
        plot(session_ids, value, label, paper=False)
