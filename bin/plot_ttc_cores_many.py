import os
import sys
import time
import glob
import pandas as pd

from common import PICKLE_DIR, get_resources,\
    BARRIER_AGENT_LAUNCH, BARRIER_CLIENT_SUBMIT, BARRIER_GENERATION,\
    barrier_legend, barrier_colors, barrier_marker, BARRIER_FONTSIZE, BARRIER_LINEWIDTH

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

            session_dir = os.path.join(PICKLE_DIR, sid)

            unit_info_df = pd.read_pickle(os.path.join(session_dir, 'unit_info.pkl'))
            pilot_info_df = pd.read_pickle(os.path.join(session_dir, 'pilot_info.pkl'))
            tr_unit_prof_df = pd.read_pickle(os.path.join(session_dir, 'tr_unit_prof.pkl'))
            session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))

            # Legend info
            info = session_info_df.loc[sid]

            cores = info['metadata.effective_cores']

            if cores not in orte_ttc:
                orte_ttc[cores] = []

            # For this call assume that there is only one pilot per session
            resources = get_resources(unit_info_df, pilot_info_df, sid)
            assert len(resources) == 1
            resource_label = resources.values()[0]

            # Get only the entries for this session
            tuf = tr_unit_prof_df[tr_unit_prof_df['sid'] == sid]

            # Only take completed CUs into account
            tuf = tuf[tuf['Done'].notnull()]

            # We sort the units based on the order they arrived at the agent
            tufs = tuf.sort('awo_get_u_pend')

            orte_ttc[cores].append((tufs['aec_after_exec'].max() - tufs['awo_get_u_pend'].min()))


        orte_df = pd.DataFrame(orte_ttc)

        labels.append("%s" % barrier_legend[key])
        ax = orte_df.mean().plot(kind='line', color=barrier_colors[key], marker=barrier_marker[key], fontsize=BARRIER_FONTSIZE, linewidth=BARRIER_LINEWIDTH)

    print 'labels: %s' % labels
    mp.pyplot.legend(labels, loc='lower left', fontsize=BARRIER_FONTSIZE)
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
    mp.pyplot.ylim(0)
    #ax.get_xaxis().set_ticks([])
    #ax.get_xaxis.set

    if paper:
        mp.pyplot.savefig('plot_ttc_cores_barriers.pdf')
    else:
        mp.pyplot.savefig('plot_ttc_cores_many.pdf')

    mp.pyplot.close()


###############################################################################
#
if __name__ == '__main__':

    session_ids = {
        BARRIER_AGENT_LAUNCH: [
            # Comet after scheduler fix2
            "rp.session.ip-10-184-31-85.santcroos.016747.0011", # 1 node
            "rp.session.ip-10-184-31-85.santcroos.016747.0009", # 2 node
            "rp.session.ip-10-184-31-85.santcroos.016747.0008", # 4 node
            "rp.session.ip-10-184-31-85.santcroos.016747.0010", # 8 nodes
            "rp.session.ip-10-184-31-85.santcroos.016747.0013", # 16 nodes
            "rp.session.ip-10-184-31-85.santcroos.016747.0000", # 32 nodes
            "rp.session.ip-10-184-31-85.santcroos.016747.0001", # 48 nodes
        ],
        BARRIER_GENERATION: [
            # Comet generation barrier / exp9
            # "rp.session.ip-10-184-31-85.santcroos.016758.0016", # 1
            # "rp.session.ip-10-184-31-85.santcroos.016758.0009", # 2
            # "rp.session.ip-10-184-31-85.santcroos.016758.0015", # 4
            # "rp.session.ip-10-184-31-85.santcroos.016758.0010", # 8
            # "rp.session.ip-10-184-31-85.santcroos.016758.0019", # 16
            # "rp.session.ip-10-184-31-85.santcroos.016758.0000", # 32
            # "rp.session.ip-10-184-31-85.santcroos.016758.0020", # 48

            "rp.session.ip-10-184-31-85.santcroos.016760.0010", # 24 / 1
            "rp.session.ip-10-184-31-85.santcroos.016760.0015", # 48 / 2
            "rp.session.ip-10-184-31-85.santcroos.016760.0014", # 96 / 4
            "rp.session.ip-10-184-31-85.santcroos.016760.0013", # 192 / 8
            "rp.session.ip-10-184-31-85.santcroos.016762.0001", # 384 / 16
            "rp.session.ip-10-184-31-85.santcroos.016760.0016", # 768 / 32
            "rp.session.ip-10-184-31-85.santcroos.016762.0002", # 1152 / 48
        ],
        BARRIER_CLIENT_SUBMIT: [
            # Single client barrier / exp10

            # "rp.session.ip-10-184-31-85.santcroos.016759.0016", # 1
            # "rp.session.ip-10-184-31-85.santcroos.016759.0015", # 2
            # "rp.session.ip-10-184-31-85.santcroos.016759.0014", # 4
            # "rp.session.ip-10-184-31-85.santcroos.016759.0009", # 8
            # "rp.session.ip-10-184-31-85.santcroos.016759.0001", # 16
            # "rp.session.ip-10-184-31-85.santcroos.016759.0000", # 32
            # "rp.session.ip-10-184-31-85.santcroos.016759.0010", # 48

            #"rp.session.ip-10-184-31-85.santcroos.016762.0000", # 1
            "rp.session.ip-10-184-31-85.santcroos.016760.0007", # 1
            "rp.session.ip-10-184-31-85.santcroos.016760.0005", # 2
            "rp.session.ip-10-184-31-85.santcroos.016760.0003", # 4
            #"rp.session.ip-10-184-31-85.santcroos.016760.0009", # 8
            "rp.session.ip-10-184-31-85.santcroos.016762.0003", # 8
            "rp.session.ip-10-184-31-85.santcroos.016760.0004", # 16
            "rp.session.ip-10-184-31-85.santcroos.016760.0008", # 32
            "rp.session.ip-10-184-31-85.santcroos.016760.0006", # 48
        ]
    }

    plot(session_ids, paper=True)
