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
from matplotlib import pyplot as plt
import numpy as np

cmap = plt.get_cmap('jet')


###############################################################################
#
# TODO: add concurrent CUs on right axis
def plot(sids, paper=False):

    colors = [cmap(i) for i in np.linspace(0, 1, len(sids))]
    c = 0

    labels = []

    for key in sorted(sids.keys(), key=int, reverse=False):

        orte_ttc = {}

        for sid in sids[key]:

            session_dir = os.path.join(PICKLE_DIR, sid)

            unit_info_df = pd.read_pickle(os.path.join(session_dir, 'unit_info.pkl'))
            pilot_info_df = pd.read_pickle(os.path.join(session_dir, 'pilot_info.pkl'))
            tr_unit_prof_df = pd.read_pickle(os.path.join(session_dir, 'tr_unit_prof.pkl'))
            session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))

            # Legend info
            info = session_info_df.loc[sid]

            cu_runtime = info['metadata.cu_runtime']
            generations = info['metadata.generations']

            if cu_runtime not in orte_ttc:
                orte_ttc[cu_runtime] = pd.Series()

            # For this call assume that there is only one pilot per session
            resources = get_resources(unit_info_df, pilot_info_df, sid)
            assert len(resources) == 1
            resource_label = resources.values()[0]

            # Get only the entries for this session
            tuf = tr_unit_prof_df[tr_unit_prof_df['sid'] == sid]

            # We sort the units based on the order they arrived at the agent
            #tufs = tuf.sort('awo_get_u_pend')
            #tufs = tuf.sort('awo_adv_u')
            tufs = tuf.sort('asic_get_u_pend')

            val = orte_ttc[cu_runtime].append(pd.Series((tufs['aec_after_exec'].max() - tufs['asic_get_u_pend'].min())))
            if val[0] < (generations * cu_runtime):
                # This likely means the pilot runtime was too short and we didn't complete all cu's
                print ("Einstein was wrong!?!")
                val = val/val
            else:
                val /= (generations * cu_runtime)
                val = 1 / val
                val *= 100

            orte_ttc[cu_runtime] = val

        print 'orte_ttc raw:', orte_ttc
        #print 'orte_ttc mean:', orte_ttc.mean()
        orte_df = pd.DataFrame(orte_ttc)
        print 'orte_ttc df:', orte_df

        #labels.append("%s" % resource_legend[key])
        labels.append("%s" % key)
        #ax = orte_df.mean().plot(kind='line', color=resource_colors[key], marker=resource_marker[key], fontsize=BARRIER_FONTSIZE, linewidth=BARRIER_LINEWIDTH)
        ax = orte_df.mean().plot(kind='line', marker='+', color=colors[c], fontsize=BARRIER_FONTSIZE, linewidth=BARRIER_LINEWIDTH)
        c += 1

    # ORTE only
    # Data for BW
    #mp.pyplot.plot((128, 256, 512, 1024, 2048, 4096, 8192), (305, 309, 309, 313, 326, 351, 558), 'b-+')
    # Data for Stampede
    #mp.pyplot.plot((128, 256, 512, 1024, 2048, 4096), (301, 303, 305, 311, 322, 344), 'b-+')
    #labels.append("ORTE-only (C)")

    # Horizontal reference
    y_ref = 100
    mp.pyplot.plot((0, 10000), (y_ref, y_ref), 'k--')
    labels.append("Optimal")

    print 'labels: %s' % labels
    mp.pyplot.legend(labels, loc='lower right', fontsize=BARRIER_FONTSIZE)
    if not paper:
        mp.pyplot.title("Resource efficiency for varying CU runtime.\n"
            "%d generations of a variable number of 'concurrent' CUs with a variable payload on a variable core pilot on %s.\n"
            "Constant number of %d sub-agent with %d ExecWorker(s) each.\n"
            "RP: %s - RS: %s - RU: %s"
           % (info['metadata.generations'], resource_label,
              info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
              info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
              ), fontsize=8)
    mp.pyplot.xlabel("# CU Runtime (s)", fontsize=BARRIER_FONTSIZE)
    mp.pyplot.ylabel("Resource efficiency (%)", fontsize=BARRIER_FONTSIZE)
    #mp.pyplot.ylim(0, 105)
    #mp.pyplot.xlim(0, 4096)
    #mp.pyplot.ylim(290, 500)
    #mp.pyplot.ylim(0, 2000)
    #mp.pyplot.ylim(y_ref-10)
    #ax.get_xaxis().set_ticks([])
    #ax.get_xaxis.set
    #ax.set_yscale('log', basey=10)
    ax.set_xscale('log', basex=2)

    mp.pyplot.savefig('plot_cu_duration_effect.pdf')

    mp.pyplot.close()


###############################################################################
#
if __name__ == '__main__':

    session_ids = {

        # BW
        # '32': [
        #     # bw orte lib - cores = 32, varying runtime
        #     "rp.session.radical.marksant.016855.0047", #
        #     "rp.session.radical.marksant.016855.0048", #
        #     "rp.session.radical.marksant.016855.0049", #
        #     "rp.session.radical.marksant.016855.0050", #
        #     "rp.session.radical.marksant.016855.0051", #
        #     "rp.session.radical.marksant.016855.0052", #
        #     "rp.session.radical.marksant.016855.0053", #
        #     "rp.session.radical.marksant.016855.0054", #
        #     # "rp.session.radical.marksant.016855.0055", # no pickle
        #     "rp.session.radical.marksant.016855.0056", #
        # ],
        #
        # '64': [
        #     # bw orte lib - cores = 64, varying runtime
        #     "rp.session.radical.marksant.016855.0057", #
        #     "rp.session.radical.marksant.016855.0058", #
        #     "rp.session.radical.marksant.016855.0059", #
        #     # "rp.session.radical.marksant.016855.0060", # no pickle
        #     "rp.session.radical.marksant.016855.0061", #
        #     # "rp.session.radical.marksant.016855.0062", # no pickle
        #     "rp.session.radical.marksant.016855.0063", #
        #     "rp.session.radical.marksant.016855.0064", #
        #     "rp.session.radical.marksant.016855.0065", #
        #     # "rp.session.radical.marksant.016855.0066", # no pickle
        # ],
        #
        # '128': [
        #     # bw orte lib - cores = 128, varying runtime
        #     "rp.session.radical.marksant.016855.0018", # 1
        #     "rp.session.radical.marksant.016855.0019", # 2
        #     #"rp.session.radical.marksant.016855.0025", # 4 no pickle
        #     "rp.session.radical.marksant.016855.0023", # 8
        #     "rp.session.radical.marksant.016855.0021", # 16
        #     "rp.session.radical.marksant.016855.0026", # 32
        #     "rp.session.radical.marksant.016855.0017", # 64
        #     "rp.session.radical.marksant.016855.0024", # 128
        #     "rp.session.radical.marksant.016855.0022", # 256
        #     "rp.session.radical.marksant.016855.0020", # 512
        # ],
        #
        # '256': [
        #     # bw orte lib - cores = 256, varying runtime
        #     "rp.session.radical.marksant.016855.0067", #
        #     "rp.session.radical.marksant.016855.0068", #
        #     "rp.session.radical.marksant.016855.0069", #
        #     # "rp.session.radical.marksant.016855.0070", # no pickle
        #     "rp.session.radical.marksant.016855.0071", #
        #     "rp.session.radical.marksant.016855.0072", #
        #     "rp.session.radical.marksant.016855.0073", #
        #     "rp.session.radical.marksant.016855.0074", #
        #     # "rp.session.radical.marksant.016855.0075", # no pickle
        #     "rp.session.radical.marksant.016855.0076", #
        # ],
        #
        # '512': [
        #     # bw orte lib - cores = 512, varying runtime
        #     "rp.session.radical.marksant.016855.0077", #
        #     "rp.session.radical.marksant.016856.0000", #
        #     "rp.session.radical.marksant.016856.0001", #
        #     "rp.session.radical.marksant.016856.0002", #
        #     "rp.session.radical.marksant.016856.0003", #
        #     "rp.session.radical.marksant.016856.0004", #
        #     "rp.session.radical.marksant.016856.0005", #
        #     "rp.session.radical.marksant.016856.0006", #
        #     # "rp.session.radical.marksant.016856.0007", # no pickle
        #     # "rp.session.radical.marksant.016856.0008", # no pickle
        # ],
        #
        # '1024': [
        #     # bw orte lib - cores = 1024, varying runtime
        #     "rp.session.radical.marksant.016855.0027", # 64
        #     "rp.session.radical.marksant.016855.0028", # 1
        #     # "rp.session.radical.marksant.016855.0029", # 2, no pickle
        #     "rp.session.radical.marksant.016855.0030", # 512
        #     "rp.session.radical.marksant.016855.0031", # 16
        #     "rp.session.radical.marksant.016855.0032", # 256
        #     # "rp.session.radical.marksant.016855.0033", # 8, no pickle
        #     "rp.session.radical.marksant.016855.0034", # 128
        #     # "rp.session.radical.marksant.016855.0035", # 4, no pickle
        #     # "rp.session.radical.marksant.016855.0036", # 32, no pickle
        # ],
        #
        # '2048': [
        #     # # bw orte lib - cores = 2048, varying runtime
        #     # "rp.session.radical.marksant.016855.0036", # no pickle
        #     "rp.session.radical.marksant.016855.0037", #
        #     # "rp.session.radical.marksant.016855.0038", # no pickle
        #     # "rp.session.radical.marksant.016855.0039", # no pickle
        #     # "rp.session.radical.marksant.016855.0040", # no pickle
        #     "rp.session.radical.marksant.016855.0041", #
        #     # "rp.session.radical.marksant.016855.0042", # no pickle
        #     "rp.session.radical.marksant.016855.0043", #
        #     "rp.session.radical.marksant.016855.0044", #
        #     # "rp.session.radical.marksant.016855.0045", # no pickle
        #     "rp.session.radical.marksant.016855.0046", #
        # ],
        #
        # '4096': [
        #     # bw orte lib - cores = 4096, varying runtime
        #     "rp.session.radical.marksant.016856.0009", #
        #     "rp.session.radical.marksant.016856.0010", #
        #     "rp.session.radical.marksant.016856.0011", #
        #     "rp.session.radical.marksant.016856.0012", #
        #     "rp.session.radical.marksant.016856.0013", #
        #     "rp.session.radical.marksant.016856.0014", #
        #     "rp.session.radical.marksant.016856.0015", #
        #     "rp.session.radical.marksant.016856.0016", #
        #     "rp.session.radical.marksant.016856.0017", #
        #     # "rp.session.radical.marksant.016856.0018", # no pickle
        # ]

        # STAMPEDE
        '64': [
            'rp.session.radical.marksant.016864.0002',
            'rp.session.radical.marksant.016864.0003',
            'rp.session.radical.marksant.016864.0004',
            'rp.session.radical.marksant.016864.0005',
            'rp.session.radical.marksant.016864.0006',
            'rp.session.radical.marksant.016864.0007',
            #'rp.session.radical.marksant.016864.0008', # 512s too short pilot
            'rp.session.radical.marksant.016864.0009',
            'rp.session.radical.marksant.016864.0010',
            'rp.session.radical.marksant.016864.0011',
            'rp.session.radical.marksant.016865.0030',
        ],
        '128': [
            'rp.session.radical.marksant.016864.0012',
            'rp.session.radical.marksant.016864.0013',
            'rp.session.radical.marksant.016864.0014',
            'rp.session.radical.marksant.016864.0015',
            'rp.session.radical.marksant.016864.0016',
            'rp.session.radical.marksant.016864.0017',
            #'rp.session.radical.marksant.016864.0018', # 512s too short pilot
            'rp.session.radical.marksant.016864.0019',
            'rp.session.radical.marksant.016864.0020',
            'rp.session.radical.marksant.016864.0021',
            'rp.session.radical.marksant.016865.0034',
        ],
        '2048': [
            'rp.session.radical.marksant.016864.0022',
            'rp.session.radical.marksant.016864.0023',
            'rp.session.radical.marksant.016864.0024',
            # 'rp.session.radical.marksant.016864.0025',
            'rp.session.radical.marksant.016864.0026',
            'rp.session.radical.marksant.016864.0027',
            #'rp.session.radical.marksant.016864.0028', # 512s too short pilot
            'rp.session.radical.marksant.016864.0029',
            'rp.session.radical.marksant.016864.0030',
            'rp.session.radical.marksant.016864.0031',
            'rp.session.radical.marksant.016865.0035',
        ],
        '512': [
            'rp.session.radical.marksant.016864.0032',
            'rp.session.radical.marksant.016864.0033',
            'rp.session.radical.marksant.016864.0034',
            'rp.session.radical.marksant.016864.0035',
            'rp.session.radical.marksant.016864.0036',
            'rp.session.radical.marksant.016864.0037',
            #'rp.session.radical.marksant.016864.0038', # 512s too short pilot
            'rp.session.radical.marksant.016864.0039',
            'rp.session.radical.marksant.016864.0040',
            'rp.session.radical.marksant.016864.0041',
            'rp.session.radical.marksant.016865.0036',
        ],
        '32': [
            'rp.session.radical.marksant.016864.0042',
            'rp.session.radical.marksant.016864.0043',
            'rp.session.radical.marksant.016864.0044',
            'rp.session.radical.marksant.016864.0045',
            'rp.session.radical.marksant.016864.0046',
            'rp.session.radical.marksant.016864.0047',
            #'rp.session.radical.marksant.016864.0048', # 512s too short pilot
            'rp.session.radical.marksant.016864.0049',
            'rp.session.radical.marksant.016864.0050',
            'rp.session.radical.marksant.016864.0051',
            'rp.session.radical.marksant.016865.0029',
        ],
        '16': [
            # 'rp.session.radical.marksant.016864.0052',
            'rp.session.radical.marksant.016864.0053',
            'rp.session.radical.marksant.016864.0054',
            'rp.session.radical.marksant.016864.0055',
            'rp.session.radical.marksant.016864.0056',
            'rp.session.radical.marksant.016864.0057',
            #'rp.session.radical.marksant.016864.0058', # 512s too short pilot
            'rp.session.radical.marksant.016864.0059',
            'rp.session.radical.marksant.016864.0060',
            'rp.session.radical.marksant.016864.0061',
            'rp.session.radical.marksant.016865.0042',
            'rp.session.radical.marksant.016866.0000'
        ],
        '4096': [
            'rp.session.radical.marksant.016864.0062',
            'rp.session.radical.marksant.016865.0000',
            'rp.session.radical.marksant.016865.0001',
            'rp.session.radical.marksant.016865.0002',
            'rp.session.radical.marksant.016865.0003',
            'rp.session.radical.marksant.016865.0004',
            # 'rp.session.radical.marksant.016865.0005', # 512s too short pilot
            'rp.session.radical.marksant.016865.0006',
            'rp.session.radical.marksant.016865.0007',
            'rp.session.radical.marksant.016865.0008',
            'rp.session.radical.marksant.016865.0040',
        ],
        '1024': [
            'rp.session.radical.marksant.016865.0009',
            'rp.session.radical.marksant.016865.0010',
            'rp.session.radical.marksant.016865.0011',
            'rp.session.radical.marksant.016865.0012',
            'rp.session.radical.marksant.016865.0013',
            'rp.session.radical.marksant.016865.0014',
        #     #'rp.session.radical.marksant.016865.0015', # 512s too short pilot
            'rp.session.radical.marksant.016865.0016',
            'rp.session.radical.marksant.016865.0017',
            'rp.session.radical.marksant.016865.0018',
            'rp.session.radical.marksant.016865.0041',
        ],
        '256': [
            'rp.session.radical.marksant.016865.0019',
            'rp.session.radical.marksant.016865.0020',
            'rp.session.radical.marksant.016865.0021',
            'rp.session.radical.marksant.016865.0022',
            'rp.session.radical.marksant.016865.0023',
            'rp.session.radical.marksant.016865.0024',
        #     #'rp.session.radical.marksant.016865.0025', # 512s too short pilot
            'rp.session.radical.marksant.016865.0026',
            'rp.session.radical.marksant.016865.0027',
            'rp.session.radical.marksant.016865.0028',
            'rp.session.radical.marksant.016865.0038',
        ],
        '8192': [
            #'rp.session.radical.marksant.016865.0031',
            'rp.session.radical.marksant.016865.0039',
            'rp.session.radical.marksant.016866.0001',
            'rp.session.radical.marksant.016866.0002',
            'rp.session.radical.marksant.016866.0003',
            #'rp.session.radical.marksant.016866.0004',
            #'rp.session.radical.marksant.016866.0005', # 1/3 units failed ...
            #'rp.session.radical.marksant.016866.0006',
            #'rp.session.radical.marksant.016866.0007',
            #'rp.session.radical.marksant.016866.0008',
            #'rp.session.radical.marksant.016866.0009',
        ]

    }

    plot(session_ids, paper=False)
