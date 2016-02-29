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
from matplotlib import style
#style.use('ggplot')
#style.use('fivethirtyeight')
#cmap = plt.get_cmap('gnuplot')
#cmap = plt.get_cmap('Paired')
cmap = plt.get_cmap('jet')
#cmap = plt.get_cmap('Dark2')

###############################################################################
#
# TODO: add concurrent CUs on right axis
def plot(sids, paper=False):

    labels = []

    colors = [cmap(i) for i in np.linspace(0, 1, len(sids))]
    c = 0

    for key in sids:

        orte_ttc = {}

        for sid in sids[key]:

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

            # For this call assume that there is only one pilot per session
            resources = get_resources(unit_info_df, pilot_info_df, sid)
            assert len(resources) == 1
            resource_label = resources.values()[0]

            # Get only the entries for this session
            tuf = tr_unit_prof_df[tr_unit_prof_df['sid'] == sid]

            # Only take completed CUs into account
            #tuf = tuf[tuf['Done'].notnull()]

            # We sort the units based on the order they arrived at the agent
            #tufs = tuf.sort('awo_get_u_pend')
            #tufs = tuf.sort('awo_adv_u')
            tufs = tuf.sort('asic_get_u_pend')

            orte_ttc[cores] = orte_ttc[cores].append(pd.Series((tufs['aec_after_exec'].max() - tufs['asic_get_u_pend'].min())))

        print 'orte_ttc raw:', orte_ttc
        #print 'orte_ttc mean:', orte_ttc.mean()
        orte_df = pd.DataFrame(orte_ttc)
        print 'orte_ttc df:', orte_df

        #labels.append("%s" % resource_legend[key])
        labels.append("%s" % key)
        #ax = orte_df.mean().plot(kind='line', color=resource_colors[key], marker=resource_marker[key], fontsize=BARRIER_FONTSIZE, linewidth=BARRIER_LINEWIDTH)
        ax = orte_df.mean().plot(kind='line', color=colors[c])
        c += 1

    # ORTE only
    # Data for BW
    #mp.pyplot.plot((128, 256, 512, 1024, 2048, 4096, 8192), (305, 309, 309, 313, 326, 351, 558), 'b-+')
    # Data for Stampede
    # mp.pyplot.plot((128, 256, 512, 1024, 2048, 4096), (301, 303, 305, 311, 322, 344), 'b-+')
    # labels.append("ORTE-only (C)")


    # Horizontal reference
    y_ref = info['metadata.generations'] * info['metadata.cu_runtime']
    mp.pyplot.plot((0, 10000), (y_ref, y_ref), 'k--')
    labels.append("Optimal")

    #print 'labels: %s' % labels
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
    #mp.pyplot.ylim(y_ref-10)
    #ax.get_xaxis().set_ticks([])
    #ax.get_xaxis.set

    mp.pyplot.savefig('plot_worker_effect.pdf')

    mp.pyplot.close()


###############################################################################
#
if __name__ == '__main__':

    session_ids = {

        "1,1 (1)": [
            "rp.session.radical.marksant.016856.0027", # -  {'pilot_cores': 1056, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 1024,  'num_sub_agents': 1,
            "rp.session.radical.marksant.016856.0045", # -  {'pilot_cores': 4128, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 4096,  'num_sub_agents': 1,
            "rp.session.radical.marksant.016856.0036", # -  {'pilot_cores': 2080, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 2048,  'num_sub_agents': 1,
            "rp.session.radical.marksant.016856.0049", # -  {'pilot_cores': 8224, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 8192,  'num_sub_agents': 1,
        ],
        "1,2 (2)": [
            "rp.session.radical.marksant.016856.0025", # -  {'pilot_cores': 1056, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 1024,  'num_sub_agents': 1,
            "rp.session.radical.marksant.016856.0034", # -  {'pilot_cores': 2080, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 2048,  'num_sub_agents': 1,
            "rp.session.radical.marksant.016856.0043", # -  {'pilot_cores': 4128, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 4096,  'num_sub_agents': 1,
            "rp.session.radical.marksant.016856.0051", # -  {'pilot_cores': 8224, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 8192,  'num_sub_agents': 1,
        ],
        "1,4 (4)": [
            "rp.session.radical.marksant.016856.0026", # -  {'pilot_cores': 1056, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 1024,  'num_sub_agents': 1,
            # "rp.session.radical.marksant.016856.0035", # -  {'pilot_cores': 2080, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 2048,  'num_sub_agents': 1,
            "rp.session.radical.marksant.016856.0044", # -  {'pilot_cores': 4128, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 4096,  'num_sub_agents': 1,
            "rp.session.radical.marksant.016856.0050", # -  {'pilot_cores': 8224, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 8192,  'num_sub_agents': 1,
        ],
        "2,1 (2)": [
            # "rp.session.radical.marksant.016856.0024", # -  {'pilot_cores': 1088, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 1024,  'num_sub_agents': 2,
            "rp.session.radical.marksant.016856.0033", # -  {'pilot_cores': 2112, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 2048,  'num_sub_agents': 2,
            "rp.session.radical.marksant.016856.0046", # -  {'pilot_cores': 8256, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 8192,  'num_sub_agents': 2,
            "rp.session.radical.marksant.016856.0042", # -  {'pilot_cores': 4160, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 4096,  'num_sub_agents': 2,
        ],
        "2,2 (4)": [
            "rp.session.radical.marksant.016856.0022", # -  {'pilot_cores': 1088, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 1024,  'num_sub_agents': 2, ,
            "rp.session.radical.marksant.016856.0031", # -  {'pilot_cores': 2112, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 2048,  'num_sub_agents': 2,
            "rp.session.radical.marksant.016856.0048", # -  {'pilot_cores': 8256, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 8192,  'num_sub_agents': 2,
            "rp.session.radical.marksant.016856.0040", # -  {'pilot_cores': 4160, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 4096,  'num_sub_agents': 2,
        ],
        "2,4 (8)": [
            "rp.session.radical.marksant.016856.0023", #-  {'pilot_cores': 1088, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 1024,  'num_sub_agents': 2,
            "rp.session.radical.marksant.016856.0032", # -  {'pilot_cores': 2112, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 2048,  'num_sub_agents': 2,
            "rp.session.radical.marksant.016856.0041", # -  {'pilot_cores': 4160, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 4096,  'num_sub_agents': 2,
            "rp.session.radical.marksant.016856.0047", # -  {'pilot_cores': 8256, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 8192,  'num_sub_agents': 2,
        ],
        "4,1 (4)": [
            "rp.session.radical.marksant.016856.0021", #-  {'pilot_cores': 1152, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 1024,  'num_sub_agents': 4,
            # "rp.session.radical.marksant.016856.0030", # -  {'pilot_cores': 2176, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 2048,  'num_sub_agents': 4,
            "rp.session.radical.marksant.016856.0039", # -  {'pilot_cores': 4224, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 4096,  'num_sub_agents': 4,
            # "rp.session.radical.marksant.016856.0052", # -  {'pilot_cores': 8320, , 'num_exec_instances_per_sub_agent': 1, 'effective_cores': 8192,  'num_sub_agents': 4,
        ],
        "4,2 (8)": [
            "rp.session.radical.marksant.016856.0019", # -  {'pilot_cores': 1152, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 1024,  'num_sub_agents': 4,
            "rp.session.radical.marksant.016856.0028", # -  {'pilot_cores': 2176, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 2048,  'num_sub_agents': 4,
            "rp.session.radical.marksant.016856.0037", # -  {'pilot_cores': 4224, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 4096,  'num_sub_agents': 4,
            "rp.session.radical.marksant.016856.0054", # -  {'pilot_cores': 8320, , 'num_exec_instances_per_sub_agent': 2, 'effective_cores': 8192,  'num_sub_agents': 4,
        ],
        "4,4 (16)": [
            "rp.session.radical.marksant.016856.0020", #-  {'pilot_cores': 1152, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 1024,  'num_sub_agents': 4,
            "rp.session.radical.marksant.016856.0029", # -  {'pilot_cores': 2176, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 2048,  'num_sub_agents': 4,
            "rp.session.radical.marksant.016856.0038", # -  {'pilot_cores': 4224, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 4096,  'num_sub_agents': 4,
            # "rp.session.radical.marksant.016856.0053", #-  {'pilot_cores': 8320, , 'num_exec_instances_per_sub_agent': 4, 'effective_cores': 8192,  'num_sub_agents': 4,
        ],



    }

    plot(session_ids, paper=False)
