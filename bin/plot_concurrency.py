import os
import sys
import time
import glob
import pandas as pd

from common import PICKLE_DIR, get_ppn, get_resources, BARRIER_FONTSIZE, TITLE_FONTSIZE

# Global Pandas settings
pd.set_option('display.width', 180)
pd.set_option('io.hdf.default_format','table')

import matplotlib as mp


###############################################################################
#
# TODO: add concurrent CUs on right axis
def plot(sids, value, label='', paper=False):


    labels = []

    first = True

    for sid in sids:

        if sid.startswith('rp.session'):
            rp = True
        else:
            rp = False

        session_dir = os.path.join(PICKLE_DIR, sid)

        unit_info_df = pd.read_pickle(os.path.join(session_dir, 'unit_info.pkl'))
        pilot_info_df = pd.read_pickle(os.path.join(session_dir, 'pilot_info.pkl'))
        unit_prof_df = pd.read_pickle(os.path.join(session_dir, 'unit_prof.pkl'))
        session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))

        # Legend info
        info = session_info_df.loc[sid]

        if rp:
            # For this call assume that there is only one pilot per session
            resources = get_resources(unit_info_df, pilot_info_df, sid)
            assert len(resources) == 1
            resource_label = resources.values()[0]
        else:
            resource_label = 'bogus'

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
            # Scheduling
            #
            df = unit_prof_df[
                (unit_prof_df.cc_exec >= 0) &
                (unit_prof_df.event == 'advance') &
                (unit_prof_df.sid == sid)
                ][['time', 'cc_exec']]

        else:
            raise Exception("Value %s unknown" % value)

        df.columns = ['time', cores]
        df['time'] -= df['time'].min()

        if first:
            df_all = df
        else:
            df_all = pd.merge(df_all, df,  on='time', how='outer')

        #labels.append("Cores: %d" % cores)
        #labels.append("%d" % cores)
        labels.append("%d - %s" % (cores, 'RP' if rp else 'ORTE'))

        first = False

    df_all.set_index('time', inplace=True)
    print df_all.head()
    #df_all.plot(colormap='Paired')
    #df_all.plot(drawstyle='steps-post')
    df_all.plot(drawstyle='steps')

    # Vertial reference
    x_ref = info['metadata.generations'] * info['metadata.cu_runtime']
    mp.pyplot.plot((x_ref, x_ref),(0, 1000), 'k--')
    labels.append("Optimal")

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
    # mp.pyplot.xlim(0, 300)
    #ax.get_xaxis().set_ticks([])

    mp.pyplot.savefig('plot5_%s%s.pdf' % (value, label))
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


        # Stampede ORTE LIB - with agent logging
        # "rp.session.radical.marksant.016848.0009", # 1 / 16
        # "rp.session.radical.marksant.016848.0006", # 2 / 32
        # "rp.session.radical.marksant.016848.0002", # 4 / 64
        # "rp.session.radical.marksant.016848.0007", # 8 / 128
        # "rp.session.radical.marksant.016848.0005", # 16 / 256
        # "rp.session.radical.marksant.016848.0014", # 32 / 512
        # "rp.session.radical.marksant.016848.0013", # 64 / 1024
        # "rp.session.radical.marksant.016848.0012", # 128 / 2048
        #"rp.session.radical.marksant.016848.0011", # 256 / 4096

        # Stampede ORTE LIB - without agent logging
        # "rp.session.radical.marksant.016848.0016", # 256

        # Stampede ORTE LIB - cloned
        # "rp.session.radical.marksant.016848.0024", # 16
        # "rp.session.radical.marksant.016848.0023", # 32
        # "rp.session.radical.marksant.016848.0031", # 64
        # "rp.session.radical.marksant.016848.0029", # 128
        # "rp.session.radical.marksant.016848.0026", # 256
        # "rp.session.radical.marksant.016848.0027", # 512
        # "rp.session.radical.marksant.016848.0030", # 1024
        # "rp.session.radical.marksant.016848.0025", # 2048
        # "rp.session.radical.marksant.016848.0028", # 4096

        # Stampede ORTE - cloned
        # "rp.session.radical.marksant.016848.0036", # 16
        # "rp.session.radical.marksant.016848.0035", # 32
        # "rp.session.radical.marksant.016848.0037", # 64
        # "rp.session.radical.marksant.016848.0040", # 128
        # "rp.session.radical.marksant.016848.0039", # 256
        # "rp.session.radical.marksant.016848.0046", # 512
        # "rp.session.radical.marksant.016848.0047", # 1024
        # "rp.session.radical.marksant.016848.0049", # 2048
        # 4096 => running out of file descriptors

        # Stampede SSH - cloned
        # "rp.session.radical.marksant.016849.0000", # 16
        # "rp.session.radical.marksant.016849.0003", # 32
        # "rp.session.radical.marksant.016849.0001", # 64
        # "rp.session.radical.marksant.016849.0002", # 128
        # "rp.session.radical.marksant.016849.0004", # 256
        # "rp.session.radical.marksant.016849.0005", # 512
        # "rp.session.radical.marksant.016849.0006", # 1024
        # "rp.session.radical.marksant.016849.0007", # 2048
        # "rp.session.radical.marksant.016849.0008", # 4096

        # BW ORTELIB - cloned
        # "rp.session.radical.marksant.016849.0025", # 32 - no dedicated agent node
        # "rp.session.radical.marksant.016849.0027", # 64 - no dedicated agent node
        # "rp.session.radical.marksant.016849.0023", # 128
        # "rp.session.radical.marksant.016849.0024", # 256 - no dedicated agent node
        # "rp.session.radical.marksant.016849.0028", # 512 - no dedicated agent node
        # "rp.session.radical.marksant.016849.0026", # 1024 - no dedicated agent node
        # "rp.session.radical.marksant.016849.0031", # 2048 - no dedicated agent node
        # "rp.session.radical.marksant.016849.0029", # 4096 - no dedicated agent node
        # "rp.session.radical.marksant.016849.0030", # 8192 - no dedicated agent node

        # BW ORTE - cloned - dedicated agent node
        # "rp.session.radical.marksant.016849.0033", # 256
        # "rp.session.radical.marksant.016849.0038", # 256
        # "rp.session.radical.marksant.016849.0032", # 256
        # "rp.session.radical.marksant.016849.0034", # 256
        # "rp.session.radical.marksant.016849.0035", # 256
        # "rp.session.radical.marksant.016849.0036", # 256
        # "rp.session.radical.marksant.016849.0037", # 256

        # BW ORTE LIB
        # "rp.session.radical.marksant.016855.0003",
        # "rp.session.radical.marksant.016855.0004",
        # BW ORTE
        # "rp.session.radical.marksant.016855.0006", # 1024
        # "rp.session.radical.marksant.016855.0008", # 2048
        # "rp.session.radical.marksant.016855.0005", # 4096
        # "rp.session.radical.marksant.016855.0007", # 8192

        # bw orte lib - 0 seconds
        # "rp.session.radical.marksant.016855.0009", # 64
        # "rp.session.radical.marksant.016855.0010", # 4096
        # "rp.session.radical.marksant.016855.0011", # 2048
        # "rp.session.radical.marksant.016855.0012", # 128
        # "rp.session.radical.marksant.016855.0013", # 512
        # "rp.session.radical.marksant.016855.0014", # 32
        # "rp.session.radical.marksant.016855.0015", # 256
        # "rp.session.radical.marksant.016855.0016", # 1024

        # # bw orte lib - cores = 128, varying runtime
        # "rp.session.radical.marksant.016855.0018", # 1
        # "rp.session.radical.marksant.016855.0019", # 2
        # #"rp.session.radical.marksant.016855.0025", # 4
        # "rp.session.radical.marksant.016855.0023", # 8
        # "rp.session.radical.marksant.016855.0021", # 16
        # "rp.session.radical.marksant.016855.0026", # 32
        # "rp.session.radical.marksant.016855.0017", # 64
        # "rp.session.radical.marksant.016855.0024", # 128
        # "rp.session.radical.marksant.016855.0022", # 256
        # "rp.session.radical.marksant.016855.0020", # 512
        #
        # # bw orte lib - cores = 1024, varying runtime
        # "rp.session.radical.marksant.016855.0027", #
        # "rp.session.radical.marksant.016855.0028", #
        # "rp.session.radical.marksant.016855.0029", #
        # "rp.session.radical.marksant.016855.0030", #
        # "rp.session.radical.marksant.016855.0031", #
        # "rp.session.radical.marksant.016855.0032", #
        # "rp.session.radical.marksant.016855.0033", #
        # "rp.session.radical.marksant.016855.0034", #
        # "rp.session.radical.marksant.016855.0035", #
        # "rp.session.radical.marksant.016855.0036", #
        #
        # # bw orte lib - cores = 2048, varying runtime
        # "rp.session.radical.marksant.016855.0036", #
        # "rp.session.radical.marksant.016855.0037", #
        # "rp.session.radical.marksant.016855.0038", #
        # "rp.session.radical.marksant.016855.0039", #
        # "rp.session.radical.marksant.016855.0040", #
        # "rp.session.radical.marksant.016855.0041", #
        # "rp.session.radical.marksant.016855.0042", #
        # "rp.session.radical.marksant.016855.0043", #
        # "rp.session.radical.marksant.016855.0044", #
        # "rp.session.radical.marksant.016855.0045", #
        # "rp.session.radical.marksant.016855.0046", #
        #
        # # bw orte lib - cores = 32, varying runtime
        # "rp.session.radical.marksant.016855.0047", #
        # "rp.session.radical.marksant.016855.0048", #
        # "rp.session.radical.marksant.016855.0049", #
        # "rp.session.radical.marksant.016855.0050", #
        # "rp.session.radical.marksant.016855.0051", #
        # "rp.session.radical.marksant.016855.0052", #
        # "rp.session.radical.marksant.016855.0053", #
        # "rp.session.radical.marksant.016855.0054", #
        # "rp.session.radical.marksant.016855.0055", #
        # "rp.session.radical.marksant.016855.0056", #
        #
        # # bw orte lib - cores = 64, varying runtime
        # "rp.session.radical.marksant.016855.0057", #
        # "rp.session.radical.marksant.016855.0058", #
        # "rp.session.radical.marksant.016855.0059", #
        # "rp.session.radical.marksant.016855.0060", #
        # "rp.session.radical.marksant.016855.0061", #
        # "rp.session.radical.marksant.016855.0062", #
        # "rp.session.radical.marksant.016855.0063", #
        # "rp.session.radical.marksant.016855.0064", #
        # "rp.session.radical.marksant.016855.0065", #
        # "rp.session.radical.marksant.016855.0066", #
        #
        # # bw orte lib - cores = 256, varying runtime
        # "rp.session.radical.marksant.016855.0067", #
        # "rp.session.radical.marksant.016855.0068", #
        # "rp.session.radical.marksant.016855.0069", #
        # "rp.session.radical.marksant.016855.0070", #
        # "rp.session.radical.marksant.016855.0071", #
        # "rp.session.radical.marksant.016855.0072", #
        # "rp.session.radical.marksant.016855.0073", #
        # "rp.session.radical.marksant.016855.0074", #
        # "rp.session.radical.marksant.016855.0075", #
        # "rp.session.radical.marksant.016855.0076", #
        #
        # # bw orte lib - cores = 512, varying runtime
        # "rp.session.radical.marksant.016855.0077", #
        # "rp.session.radical.marksant.016856.0000", #
        # "rp.session.radical.marksant.016856.0001", #
        # "rp.session.radical.marksant.016856.0002", #
        # "rp.session.radical.marksant.016856.0003", #
        # "rp.session.radical.marksant.016856.0004", #
        # "rp.session.radical.marksant.016856.0005", #
        # "rp.session.radical.marksant.016856.0006", #
        # "rp.session.radical.marksant.016856.0007", #
        # "rp.session.radical.marksant.016856.0008", #
        #
        # # bw orte lib - cores = 4096, varying runtime
        # "rp.session.radical.marksant.016856.0009", #
        # "rp.session.radical.marksant.016856.0010", #
        # "rp.session.radical.marksant.016856.0011", #
        # "rp.session.radical.marksant.016856.0012", #
        # "rp.session.radical.marksant.016856.0013", #
        # "rp.session.radical.marksant.016856.0014", #
        # "rp.session.radical.marksant.016856.0015", #
        # "rp.session.radical.marksant.016856.0016", #
        # "rp.session.radical.marksant.016856.0017", #
        # "rp.session.radical.marksant.016856.0018", #

        # bw orte lib - cores = 1024, varying exec workers
        # "rp.session.radical.marksant.016856.0019", #
        # "rp.session.radical.marksant.016856.0020", #
        # "rp.session.radical.marksant.016856.0021", #
        # "rp.session.radical.marksant.016856.0022", #
        # "rp.session.radical.marksant.016856.0023", #
        # "rp.session.radical.marksant.016856.0024", #
        # "rp.session.radical.marksant.016856.0025", #
        # "rp.session.radical.marksant.016856.0026", #
        # "rp.session.radical.marksant.016856.0027", #
        # "rp.session.radical.marksant.016856.0028", #

        # 'mw.session.h2ologin2.marksant.016863.0006', # 4096 x 3'
        # 'mw.session.nid25431.marksant.016863.0009' # 8192x3
        #'mw.session.login3.stampede.tacc.utexas.edu.marksant.016864.0002' # 100k 0s interrupted at ~75k
        #'rp.session.radical.marksant.016865.0031',
        # 'mw.session.netbook.mark.016865.0041'
        'rp.session.radical.marksant.016865.0040', # 4k
        'mw.session.nid25429.marksant.016865.0005' # 4k
    ]

    label = '_10sa_1ew'

    for value in ['exec']:
        plot(session_ids, value, label, paper=False)
