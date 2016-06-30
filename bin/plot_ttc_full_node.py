import os
import sys
import time
import glob
import pandas as pd

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

from common import PICKLE_DIR, get_resources,\
    BARRIER_AGENT_LAUNCH, BARRIER_CLIENT_SUBMIT, BARRIER_GENERATION,\
    resource_legend, resource_colors, resource_marker, LINEWIDTH, LABEL_FONTSIZE, LEGEND_FONTSIZE, TICK_FONTSIZE, TITLE_FONTSIZE

# Global Pandas settings
pd.set_option('display.width', 180)
pd.set_option('io.hdf.default_format','table')

import matplotlib as mp


###############################################################################
#
# TODO: add concurrent CUs on right axis
def plot(sids, paper=False):

    labels = []

    for key in sids:

        orte_ttc = {}

        for sid in sids[key]:

            if sid.startswith('rp.session'):
                rp = True
            else:
                rp = False

            session_dir = os.path.join(PICKLE_DIR, sid)

            unit_info_df = pd.read_pickle(os.path.join(session_dir, 'unit_info.pkl'))
            pilot_info_df = pd.read_pickle(os.path.join(session_dir, 'pilot_info.pkl'))
            tr_unit_prof_df = pd.read_pickle(os.path.join(session_dir, 'tr_unit_prof.pkl'))
            session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))

            # Legend info
            info = session_info_df.loc[sid]

            cores = info['metadata.effective_cores']
            nodes = cores / 32

            if nodes not in orte_ttc:
                orte_ttc[nodes] = pd.Series()

            if rp:
                # For this call assume that there is only one pilot per session
                resources = get_resources(unit_info_df, pilot_info_df, sid)
                assert len(resources) == 1
                resource_label = resources.values()[0]
            else:
                resource_label = 'bogus'

            # Get only the entries for this session
            tuf = tr_unit_prof_df[tr_unit_prof_df['sid'] == sid]

            # Only take completed CUs into account
            #tuf = tuf[tuf['Done'].notnull()]

            # We sort the units based on the order they arrived at the agent
            #tufs = tuf.sort('awo_get_u_pend')
            #tufs = tuf.sort('awo_adv_u')
            #tufs = tuf.sort('asic_get_u_pend')
            tufs = tuf.sort()

            orte_ttc[nodes] = orte_ttc[nodes].append(pd.Series((tufs['aec_after_exec'].max() - tufs['asic_get_u_pend'].min())))

        print 'orte_ttc raw:', orte_ttc
        #print 'orte_ttc mean:', orte_ttc.mean()
        orte_df = pd.DataFrame(orte_ttc)
        print 'orte_ttc df:', orte_df

        labels.append("%s" % resource_legend[key])
        ax = orte_df.mean().plot(kind='line', color=resource_colors[key], marker=resource_marker[key], fontsize=TICK_FONTSIZE, linewidth=LINEWIDTH)

    # ORTE only
    # Data for BW
    #mp.pyplot.plot((128, 256, 512, 1024, 2048, 4096, 8192), (305, 309, 309, 313, 326, 351, 558), 'b-+')
    # Data for Stampede
    #mp.pyplot.plot((128, 256, 512, 1024, 2048, 4096), (301, 303, 305, 311, 322, 344), 'b-+')
    #labels.append("ORTE-only (C)")

    # Horizontal reference
    y_ref = info['metadata.generations'] * info['metadata.cu_runtime']
    mp.pyplot.plot((0, 10000), (y_ref, y_ref), 'k--', linewidth=LINEWIDTH)
    labels.append("Optimal")

    print 'labels: %s' % labels
    location = 'upper left'
    mp.pyplot.legend(labels, loc=location, fontsize=LEGEND_FONTSIZE, markerscale=0)
    if not paper:
        mp.pyplot.title("TTC for a varying number of 'concurrent' Full-Node CUs.\n"
            "%d generations of a variable number of 'concurrent' CUs of %d core(s) with a %ss payload on a variable core pilot on %s.\n"
            "Constant number of %d sub-agent with %d ExecWorker(s) each.\n"
            "RP: %s - RS: %s - RU: %s"
           % (info['metadata.generations'], info['metadata.cu_cores'], info['metadata.cu_runtime'], resource_label,
              info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
              info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
              ), fontsize=TITLE_FONTSIZE)
    mp.pyplot.xlabel("\# Nodes", fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylabel("Time to Completion (s)", fontsize=LABEL_FONTSIZE)
    #mp.pyplot.ylim(0)
    #mp.pyplot.ylim(290, 500)
    #mp.pyplot.ylim(y_ref-10) #ax.get_xaxis().set_ticks([])
    # #ax.get_xaxis.set

    #width = 3.487
    width = 3.3
    height = width / 1.618
    # height = 2.7
    fig = mp.pyplot.gcf()
    fig.set_size_inches(width, height)
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=1)

    #fig.tight_layout(w_pad=0.0, h_pad=0.0, pad=0.1)
    fig.tight_layout(pad=0.1)

    mp.pyplot.savefig('plot_ttc_full_node.pdf')

    mp.pyplot.close()


###############################################################################
#
if __name__ == '__main__':

    session_ids = {
        # 'orte_lib': [
        #
        #     # Stampede ORTE LIB - cloned
        #     "rp.session.radical.marksant.016848.0024", # 16
        #     "rp.session.radical.marksant.016848.0023", # 32
        #     "rp.session.radical.marksant.016848.0031", # 64
        #     "rp.session.radical.marksant.016848.0029", # 128
        #     "rp.session.radical.marksant.016848.0026", # 256
        #     "rp.session.radical.marksant.016848.0027", # 512
        #     "rp.session.radical.marksant.016848.0030", # 1024
        #     "rp.session.radical.marksant.016848.0025", # 2048
        #     "rp.session.radical.marksant.016848.0028", # 4096
        #     'rp.session.radical.marksant.016864.0001'  # 8192
        #     ],
        #
        # 'orte': [
        #     # Stampede ORTE - cloned
        #     "rp.session.radical.marksant.016848.0036", # 16
        #     "rp.session.radical.marksant.016848.0035", # 32
        #     "rp.session.radical.marksant.016848.0037", # 64
        #     "rp.session.radical.marksant.016848.0040", # 128
        #     "rp.session.radical.marksant.016848.0039", # 256
        #     "rp.session.radical.marksant.016848.0046", # 512
        #     "rp.session.radical.marksant.016848.0047", # 1024
        #     "rp.session.radical.marksant.016848.0049", # 2048
        #     # 4096 => running out of file descriptors
        # ],
        # 'ssh': [
        #     "rp.session.radical.marksant.016849.0000", # 16
        #     "rp.session.radical.marksant.016849.0003", # 32
        #     "rp.session.radical.marksant.016849.0001", # 64
        #     "rp.session.radical.marksant.016849.0002", # 128
        #     "rp.session.radical.marksant.016849.0004", # 256
        #     "rp.session.radical.marksant.016849.0005", # 512
        #     "rp.session.radical.marksant.016849.0006", # 1024
        #     "rp.session.radical.marksant.016849.0007", # 2048
        #     "rp.session.radical.marksant.016849.0008", # 4096
        # ]


        # # BW
        # 'orte': [
        #     # BW ORTE - cloned - dedicated agent node
        #     "rp.session.radical.marksant.016849.0033", # 256
        #     "rp.session.radical.marksant.016849.0038", # 256
        #     "rp.session.radical.marksant.016849.0032", # 256
        #     "rp.session.radical.marksant.016849.0034", # 256
        #     "rp.session.radical.marksant.016849.0035", # 256
        #     "rp.session.radical.marksant.016849.0036", # 256
        #     "rp.session.radical.marksant.016849.0037", # 256
        # ],
        #
        # 'orte_lib': [
        #     # BW ORTELIB - cloned
        #     "rp.session.radical.marksant.016849.0025", # 32 - no dedicated agent node
        #     "rp.session.radical.marksant.016849.0027", # 64 - no dedicated agent node
        #     # "rp.session.radical.marksant.016849.0023", # 128
        #     "rp.session.radical.marksant.016849.0024", # 256 - no dedicated agent node
        #     "rp.session.radical.marksant.016849.0028", # 512 - no dedicated agent node
        #     "rp.session.radical.marksant.016849.0026", # 1024 - no dedicated agent node
        #     "rp.session.radical.marksant.016849.0031", # 2048 - no dedicated agent node
        #     "rp.session.radical.marksant.016849.0029", # 4096 - no dedicated agent node
        #     "rp.session.radical.marksant.016849.0030", # 8192 - no dedicated agent node
        # ]

        # MW
        # 'orte': [
        #     # 'mw.session.h2ologin2.marksant.016863.0006', # 4096 x 3'
        #     # 'mw.session.nid25431.marksant.016863.0009', # 8192x3
        #     # 'mw.session.netbook.mark.016865.0041'
        #      "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0006",
        #     "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0007",
        #     "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0008",
        #     "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0009",
        #     "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0010",
        #     "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0011",
        #     "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0012",
        #     "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0013",
        #     "mw.session.c406-003.stampede.tacc.utexas.edu.marksant.016869.0014",
        # ]

        # FUll node BW
        # 'ccm': [
        #
        # ],

        'aprun': [
            'rp.session.radical.marksant.016932.0001',
            'rp.session.radical.marksant.016926.0001',
            'rp.session.radical.marksant.016926.0002',
            'rp.session.radical.marksant.016926.0003',
            'rp.session.radical.marksant.016926.0004',
            'rp.session.radical.marksant.016926.0005',
            'rp.session.radical.marksant.016926.0006',
            # 'rp.session.radical.marksant.016926.0007', ### outlier
            'rp.session.radical.marksant.016926.0008',
            'rp.session.radical.marksant.016926.0009',
        ],

        'orte': [
            'rp.session.radical.marksant.016926.0011',
            'rp.session.radical.marksant.016926.0012',
            'rp.session.radical.marksant.016926.0013',
            'rp.session.radical.marksant.016926.0014',
            'rp.session.radical.marksant.016926.0015',
            'rp.session.radical.marksant.016926.0016',
            'rp.session.radical.marksant.016926.0017',
            'rp.session.radical.marksant.016926.0018',
            'rp.session.radical.marksant.016926.0019',
        ],

        'orte_lib': [
            'rp.session.radical.marksant.016926.0028',
            'rp.session.radical.marksant.016926.0027',
            'rp.session.radical.marksant.016926.0026',
            'rp.session.radical.marksant.016926.0025',
            'rp.session.radical.marksant.016926.0024',
            'rp.session.radical.marksant.016926.0023',
            'rp.session.radical.marksant.016926.0022',
            'rp.session.radical.marksant.016926.0021',
            'rp.session.radical.marksant.016926.0020',
            'rp.session.radical.marksant.016926.0019',
        ],

        'ccm': [
            'rp.session.radical.marksant.016926.0033',
            'rp.session.radical.marksant.016927.0001',
            'rp.session.radical.marksant.016927.0009',
            'rp.session.radical.marksant.016927.0008',
            'rp.session.radical.marksant.016927.0007',
            'rp.session.radical.marksant.016927.0006',
            'rp.session.radical.marksant.016927.0005',
            'rp.session.radical.marksant.016927.0004',
            'rp.session.radical.marksant.016927.0003',
            'rp.session.radical.marksant.016927.0002',
        ]

    }

    plot(session_ids, paper=True)
