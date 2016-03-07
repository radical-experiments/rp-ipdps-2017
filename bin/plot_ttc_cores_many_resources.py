import os
import sys
import time
import glob
import pandas as pd

from common import PICKLE_DIR, get_resources,\
    BARRIER_AGENT_LAUNCH, BARRIER_CLIENT_SUBMIT, BARRIER_GENERATION,\
    resource_legend, resource_colors, resource_marker, BARRIER_FONTSIZE, BARRIER_LINEWIDTH

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

            if cores not in orte_ttc:
                orte_ttc[cores] = pd.Series()

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

            orte_ttc[cores] = orte_ttc[cores].append(pd.Series((tufs['aec_after_exec'].max() - tufs['asic_get_u_pend'].min())))

        print 'orte_ttc raw:', orte_ttc
        #print 'orte_ttc mean:', orte_ttc.mean()
        orte_df = pd.DataFrame(orte_ttc)
        print 'orte_ttc df:', orte_df

        labels.append("%s" % resource_legend[key])
        ax = orte_df.mean().plot(kind='line', color=resource_colors[key], marker=resource_marker[key], fontsize=BARRIER_FONTSIZE, linewidth=BARRIER_LINEWIDTH)

    # ORTE only
    # Data for BW
    #mp.pyplot.plot((128, 256, 512, 1024, 2048, 4096, 8192), (305, 309, 309, 313, 326, 351, 558), 'b-+')
    # Data for Stampede
    mp.pyplot.plot((128, 256, 512, 1024, 2048, 4096), (301, 303, 305, 311, 322, 344), 'b-+')
    labels.append("ORTE-only (C)")

    # Horizontal reference
    y_ref = info['metadata.generations'] * info['metadata.cu_runtime']
    mp.pyplot.plot((0, 10000), (y_ref, y_ref), 'k--')
    labels.append("Optimal")

    print 'labels: %s' % labels
    mp.pyplot.legend(labels, loc='upper left', fontsize=BARRIER_FONTSIZE)
    if not paper:
        mp.pyplot.title("TTC for a varying number of 'concurrent' CUs.\n"
            "%d generations of a variable number of 'concurrent' CUs of %d core(s) with a %ss payload on a variable core pilot on %s.\n"
            "Constant number of %d sub-agent with %d ExecWorker(s) each.\n"
            "RP: %s - RS: %s - RU: %s"
           % (info['metadata.generations'], info['metadata.cu_cores'], info['metadata.cu_runtime'], resource_label,
              info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
              info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
              ), fontsize=8)
    mp.pyplot.xlabel("# Cores", fontsize=BARRIER_FONTSIZE)
    mp.pyplot.ylabel("Time to Completion (s)", fontsize=BARRIER_FONTSIZE)
    #mp.pyplot.ylim(0)
    #mp.pyplot.ylim(290, 500)
    mp.pyplot.ylim(y_ref-10)
    #ax.get_xaxis().set_ticks([])
    #ax.get_xaxis.set

    if paper:
        mp.pyplot.savefig('plot_ttc_cores_resources.pdf')
    else:
        mp.pyplot.savefig('plot_ttc_cores_many.pdf')

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
        'orte': [
            # 'mw.session.h2ologin2.marksant.016863.0006', # 4096 x 3'
            # 'mw.session.nid25431.marksant.016863.0009', # 8192x3
            'mw.session.netbook.mark.016865.0041'
        ]
    }

    plot(session_ids, paper=False)
