import os
import sys
import time
import glob
import pandas as pd

from common import PICKLE_DIR, get_ppn, get_resources, LEGEND_FONTSIZE, TITLE_FONTSIZE, TICK_FONTSIZE, LABEL_FONTSIZE, LINEWIDTH, BORDERWIDTH
from radical.pilot import states as rps
from radical.pilot import utils as rpu

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

    colors = [cmap(i) for i in np.linspace(0, 1, len(sids))]

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

        if value == 'cc_fork':
            spec = {
                'in': [
                    {'info' : 'aec_start_script'}
                ],
                'out' : [
                    {'info' : 'aec_after_exec'}
                ]
            }
            rpu.add_concurrency (unit_prof_df, 'cc_fork', spec)

        elif value == 'cc_exit':
            spec = {
                'in': [
                    {'info' : 'aec_after_exec'}
                ],
                'out' : [
                    {'state': rps.AGENT_STAGING_OUTPUT_PENDING, 'event': 'advance'},
                ]
            }
            rpu.add_concurrency (unit_prof_df, 'cc_exit', spec)

        df = unit_prof_df[
            (unit_prof_df[value] >= 0) &
            #(unit_prof_df.event == 'advance') &
            (unit_prof_df.sid == sid)
            ][['time', value]]

        df.columns = ['time', cores]
        df['time'] -= df['time'].min()

        if first:
            df_all = df
        else:
            df_all = pd.merge(df_all, df,  on='time', how='outer')

        #labels.append("Cores: %d" % cores)
        # labels.append("%d" % cores)
        #labels.append("%d - %s" % (cores, 'RP' if rp else 'ORTE'))
        #labels.append(sid[-4:])
        labels.append("%d" % info['metadata.cu_runtime'])

        first = False

    df_all.set_index('time', inplace=True)
    print df_all.head()
    #df_all.plot(colormap='Paired')
    #df_all.plot(drawstyle='steps-post')
    #ax = df_all.plot(drawstyle='steps-pre', fontsize=TICK_FONTSIZE, linewidth=LINEWIDTH, colors=colors)
    ax = df_all.plot(fontsize=TICK_FONTSIZE, linewidth=LINEWIDTH, colors=colors)

    # Vertial reference
    #x_ref = info['metadata.generations'] * info['metadata.cu_runtime']
    #mp.pyplot.plot((x_ref, x_ref),(0, 1000), 'k--')
    #labels.append("Optimal")

    location = 'upper right'
    legend = mp.pyplot.legend(labels, loc=location, fontsize=LEGEND_FONTSIZE, labelspacing=0)
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
                  ), fontsize=TITLE_FONTSIZE)
    mp.pyplot.xlabel("Time (s)", fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylabel("\# Concurrent Units", fontsize=LABEL_FONTSIZE)
    # mp.pyplot.ylim(0, 200)
    mp.pyplot.ylim(-50,)
    mp.pyplot.xlim(0, 600)
    #ax.get_xaxis().set_ticks([])
    print dir(ax)

    [i.set_linewidth(BORDERWIDTH) for i in ax.spines.itervalues()]
    plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)
    plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)

    # width = 3.487
    width = 3.3
    height = width / 1.618
    # height = 2.5
    fig = mp.pyplot.gcf()
    fig.set_size_inches(width, height)
    # fig.subplots_adjust(left=0, right=1, top=1, bottom=1)

    # fig.tight_layout(w_pad=0.0, h_pad=0.0, pad=0.1)
    fig.tight_layout(pad=0.1)

    mp.pyplot.savefig('plot_concurrency.pdf')
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

        # BW ORTELIB - cloned ### INITIAL CUG PAPER DATA ###
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
        # 'rp.session.radical.marksant.016865.0040', # 4k - 512s - 3gen

        # 'mw.session.nid25429.marksant.016865.0005', # 4k - 60s - 5gen
        # 'rp.session.radical.marksant.016848.0028', # 4k - 60s - 5gen

        #'rp.session.radical.marksant.016865.0040', # 4k - 512s - 3 gen
        # 'rp.session.radical.marksant.016865.0002', # 4k - 64s - 3 gen

        # 'rp.session.radical.marksant.016868.0011', # 4k - 64s - 3 gen
        # 'rp.session.radical.marksant.016869.0000', # 4k - 64s - 3 gen


        # "rp.session.radical.marksant.016868.0016",
        # "rp.session.radical.marksant.016868.0013",
        # "rp.session.radical.marksant.016868.0015",
        # "rp.session.radical.marksant.016868.0012",

        # 'rp.session.radical.marksant.016865.0039' # 8k

        # 'rp.session.radical.marksant.016864.0062',
        # 'rp.session.radical.marksant.016865.0000',
        # 'rp.session.radical.marksant.016865.0001',
        # 'rp.session.radical.marksant.016865.0002',
        # 'rp.session.radical.marksant.016865.0003',
        # 'rp.session.radical.marksant.016865.0004',
        # 'rp.session.radical.marksant.016865.0006',
        # 'rp.session.radical.marksant.016865.0007',
        # 'rp.session.radical.marksant.016865.0008',
        # 'rp.session.radical.marksant.016865.0040',

        # "mw.session.nid25337.marksant.016869.0007", # bw 4k - 3gen x 64s - no barrier
        # "mw.session.nid25337.marksant.016869.0008", # bw 4k - 3gen x 128s - no barrier
        # "mw.session.nid25337.marksant.016869.0010", # bw 4k - 1gen x 64s - no barrier

        # "rp.session.radical.marksant.016870.0004"

        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0006", # 64s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0007", # 64s 3 gens
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0008", # 32s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0009", # 16s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0010", # 8s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0011", # 4s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0012", # 2s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0013", # 1s
        # "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0014", # 0s

        # 'mw.session.nid25254.marksant.016869.0011',
        # 'mw.session.nid25254.marksant.016869.0012',
        # 'mw.session.nid25254.marksant.016869.0013',
        # 'mw.session.nid25254.marksant.016869.0014',
        # 'mw.session.nid25254.marksant.016869.0015',
        # 'mw.session.nid25254.marksant.016869.0016',
        # 'mw.session.nid25254.marksant.016869.0017',

        # Stampede SSH
        # 'rp.session.radical.marksant.016895.0000', # 16
        # 'rp.session.radical.marksant.016895.0005', # 32
        # 'rp.session.radical.marksant.016895.0003', # 64
        # 'rp.session.radical.marksant.016895.0004', # 256
        # 'rp.session.radical.marksant.016884.0040', # 512
        # 'rp.session.radical.marksant.016884.0058', # 128
        # 'rp.session.radical.marksant.016884.0022', # 1024
        # 'rp.session.radical.marksant.016895.0001', # 2048
        # 'rp.session.radical.marksant.016895.0006', # 4096
        # 'rp.session.radical.marksant.016895.0007', # 8192

        # BW SCALING micro
        # 'rp.session.radical.marksant.016929.0006', # 8k, 128s
        # 'rp.session.radical.marksant.016928.0003', # 32k, 1024s
        # # 'rp.session.radical.marksant.016928.0000', # 8k, 512s
        # 'rp.session.radical.marksant.016928.0002', # 16k, 512s
        # # 'rp.session.radical.marksant.016928.0001', # 32k
        # 'rp.session.radical.marksant.016928.0004', # 64k, 2048s

        'rp.session.radical.marksant.016927.0012', # 2 SA  ### PAPER UNIT DURATION PLOT ###
        'rp.session.radical.marksant.016927.0017', # 2 SA  ### PAPER UNIT DURATION PLOT ###

        # BW SCALING AGENT
        # 'rp.session.radical.marksant.016929.0001', # 2k
        # 'rp.session.radical.marksant.016929.0000', # 4k
        # # 'rp.session.radical.marksant.016929.0002', # 8k, too short
        # 'rp.session.radical.marksant.016929.0007', # 8k
        # 'rp.session.radical.marksant.016930.0000', # 16k
    ]



    label = ''

    for value in ['cc_exec']:
    #for value in ['cc_fork']:
    # for value in ['cc_exit']:
        plot(session_ids, value, label, paper=True)
