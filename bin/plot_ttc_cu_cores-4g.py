import os
import sys
import time
import glob
import pandas as pd

from common import PICKLE_DIR, get_resources,\
    BARRIER_AGENT_LAUNCH, BARRIER_CLIENT_SUBMIT, BARRIER_GENERATION,\
    resource_legend, resource_colors, resource_marker, BORDERWIDTH, LEGEND_FONTSIZE, TICK_FONTSIZE, TITLE_FONTSIZE, LABEL_FONTSIZE, LINEWIDTH

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
def plot(sids, paper=False):

    labels = []

    all_dict = {}

    for sid in sids:

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

        # cu_cores = info['metadata.cu_cores']
        # cu_count = info['metadata.cu_count']
        cu_count = info['metadata.cu_cores']
        cu_cores = info['metadata.cu_count']
        cu_runtime = info['metadata.cu_runtime']

        if cu_count not in all_dict:
            all_dict[cu_count] = {}

        if cu_cores not in all_dict[cu_count]:
            all_dict[cu_count][cu_cores] = pd.Series()

        if rp:
            # For this call assume that there is only one pilot per session
            resources = get_resources(unit_info_df, pilot_info_df, sid)
            assert len(resources) == 1
            resource_label = resources.values()[0].replace('_', '\_')
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

        try:
            all_dict[cu_count][cu_cores] = all_dict[cu_count][cu_cores].append(pd.Series((tufs['aec_after_exec'].max() - tufs['asic_get_u_pend'].min() - cu_runtime)))
            #all_dict[cu_count][cu_cores] = all_dict[cu_count][cu_cores].append(pd.Series((tufs['aec_after_exec'].max() - tufs['asic_get_u_pend'].min())))
        except:
            print "Plotting failed for session: %s" % sid
            continue

    for key in all_dict:
        # print 'orte_ttc raw:', orte_ttc
        #print 'orte_ttc mean:', orte_ttc.mean()
        orte_df = pd.DataFrame(all_dict[key])
        print 'orte_ttc df:', orte_df

        #labels.append("%s" % resource_legend[key])
        labels.append("%s" % key)
        #ax = orte_df.mean().plot(kind='line', color=resource_colors[key], marker=resource_marker[key], fontsize=TICK_FONTSIZE, linewidth=LINEWIDTH)
        ax = orte_df.mean().plot(kind='line', fontsize=TICK_FONTSIZE, linewidth=LINEWIDTH)
        ax.set_xscale('log', basex=10)
        ax.set_yscale('log', basey=10)

    # Horizontal reference
    # y_ref = info['metadata.generations'] * info['metadata.cu_runtime']
    # ax = mp.pyplot.plot((0, 10000), (y_ref, y_ref), 'k--')
    # labels.append("Optimal")

    print 'labels: %s' % labels
    mp.pyplot.legend(labels, loc='bottom right', fontsize=LEGEND_FONTSIZE)
    if not paper:
        mp.pyplot.title("TTC overhead for variable size CU.\n"
            "%d generations of a variable number of 'concurrent' CUs with variable number of cores with a %ss payload on a variable core pilot on %s.\n"
            "Constant number of %d sub-agent with %d ExecWorker(s) each.\n"
            "RP: %s - RS: %s - RU: %s"
           % (info['metadata.generations'], info['metadata.cu_runtime'], resource_label,
              info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
              info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
              ), fontsize=8)
    mp.pyplot.xlabel("\# CUs", fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylabel("$TTC_{overhead}$ (s)", fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylim(0)
    #mp.pyplot.ylim(290, 500)
    #mp.pyplot.ylim(y_ref-10) #ax.get_xaxis().set_ticks([])
    # #ax.get_xaxis.set

    # [i.set_linewidth(BORDERWIDTH) for i in ax.spines.itervalues()]
    # plt.setp(ax.yaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)
    # plt.setp(ax.xaxis.get_ticklines(), 'markeredgewidth', BORDERWIDTH)

    #width = 3.487
    #width = 3.3
    #height = width / 1.618
    # height = 2.7
    #fig = mp.pyplot.gcf()
    #fig.set_size_inches(width, height)
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=1)

    #fig.tight_layout(w_pad=0.0, h_pad=0.0, pad=0.1)
    #fig.tight_layout(pad=0.1)
    #fig.tight_layout()

    mp.pyplot.savefig('plot_ttc_cu_cores.pdf')

    mp.pyplot.close()


###############################################################################
#
if __name__ == '__main__':

    session_ids = [
        # cu_count, cu_cores

        #'rp.session.radical.marksant.016989.0004',
        #'rp.session.radical.marksant.016989.0005',
        #'rp.session.radical.marksant.016989.0006',
        #'rp.session.radical.marksant.016989.0007',
        #'rp.session.radical.marksant.016989.0008',
        #'rp.session.radical.marksant.016989.0009',
        #'rp.session.radical.marksant.016989.0010',
        #'rp.session.radical.marksant.016989.0011',
        #'rp.session.radical.marksant.016989.0012',
        #'rp.session.radical.marksant.016989.0013',
        #'rp.session.radical.marksant.016989.0015',
        #'rp.session.radical.marksant.016989.0016',
        #'rp.session.radical.marksant.016989.0017',
        #'rp.session.radical.marksant.016989.0018',
        #'rp.session.radical.marksant.016989.0019',
        #'rp.session.radical.marksant.016989.0020',
        #'rp.session.radical.marksant.016989.0021',
        #'rp.session.radical.marksant.016989.0022',
        #'rp.session.radical.marksant.016989.0023',
        #'rp.session.radical.marksant.016989.0024',
        #'rp.session.radical.marksant.016989.0025',
        #'rp.session.radical.marksant.016989.0026',
        #'rp.session.radical.marksant.016989.0027',
        'rp.session.radical.marksant.016989.0029',
        'rp.session.radical.marksant.016989.0030',
        #'rp.session.radical.marksant.016989.0031',
        'rp.session.radical.marksant.016989.0034',
        'rp.session.radical.marksant.016989.0035',
        'rp.session.radical.marksant.016989.0036',
        'rp.session.radical.marksant.016989.0037',
        'rp.session.radical.marksant.016989.0038',
        'rp.session.radical.marksant.016989.0039',
        'rp.session.radical.marksant.016989.0040',
        'rp.session.radical.marksant.016989.0041',
        'rp.session.radical.marksant.016989.0042',
        'rp.session.radical.marksant.016989.0043',
        'rp.session.radical.marksant.016989.0044',
        'rp.session.radical.marksant.016989.0045',
        'rp.session.radical.marksant.016989.0046',
        'rp.session.radical.marksant.016989.0047',
        'rp.session.radical.marksant.016993.0001',
        'rp.session.radical.marksant.016993.0005',
        'rp.session.radical.marksant.016993.0006',
        'rp.session.radical.marksant.016993.0008',
        'rp.session.radical.marksant.016993.0009',
        'rp.session.radical.marksant.016993.0010',
        'rp.session.radical.marksant.016993.0011',
        'rp.session.radical.marksant.016993.0012',
        'rp.session.radical.marksant.016993.0013',
        'rp.session.radical.marksant.016993.0014',
        'rp.session.radical.marksant.016993.0015',
        'rp.session.radical.marksant.016993.0016',
        'rp.session.radical.marksant.016993.0018',
        'rp.session.radical.marksant.016993.0019',
        'rp.session.radical.marksant.016993.0020',
        'rp.session.radical.marksant.016993.0021',
        'rp.session.radical.marksant.016993.0022',
        'rp.session.radical.marksant.016993.0023',
        'rp.session.radical.marksant.016993.0024',
        'rp.session.radical.marksant.016993.0025',
        'rp.session.radical.marksant.016993.0026',
        'rp.session.radical.marksant.016993.0027',
        'rp.session.radical.marksant.016993.0028',
        'rp.session.radical.marksant.016993.0030',
        'rp.session.radical.marksant.016993.0031',
        'rp.session.radical.marksant.016993.0033',
        'rp.session.radical.marksant.016993.0035',
        'rp.session.radical.marksant.016993.0036',
        'rp.session.radical.marksant.016993.0038',
        'rp.session.radical.marksant.016993.0039',
        'rp.session.radical.marksant.016993.0040',
        'rp.session.radical.marksant.016993.0041',
        'rp.session.radical.marksant.016993.0042',
        'rp.session.radical.marksant.016993.0043',
        'rp.session.radical.marksant.016993.0044',
        'rp.session.radical.marksant.016993.0045',
        'rp.session.radical.marksant.016993.0046',
        'rp.session.radical.marksant.016993.0047',
        'rp.session.radical.marksant.016993.0048',
        'rp.session.radical.marksant.016993.0049',
        'rp.session.radical.marksant.016993.0050',
        'rp.session.radical.marksant.016993.0052',
        'rp.session.radical.marksant.016994.0008',
        'rp.session.radical.marksant.016995.0002',
        'rp.session.radical.marksant.016995.0003',
        'rp.session.radical.marksant.016995.0004',
        'rp.session.radical.marksant.016995.0006',
        'rp.session.radical.marksant.016995.0008',
        'rp.session.radical.marksant.016995.0009',
        'rp.session.radical.marksant.016995.0010',
        'rp.session.radical.marksant.016998.0000',
        #'rp.session.radical.marksant.016999.0000',




    ]

    plot(session_ids, paper=False)
