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
    resource_legend, resource_colors, resource_marker, LEGEND_FONTSIZE, LINEWIDTH, LABEL_FONTSIZE, TICK_FONTSIZE

# Global Pandas settings
pd.set_option('display.width', 180)
pd.set_option('io.hdf.default_format','table')

import matplotlib as mp
from matplotlib import pyplot as plt
import numpy as np

cmap = plt.get_cmap('jet')


###############################################################################
#
def plot(sids, key, paper=False):


    # keys = []
    # for sid in sids:
    #     print ("sid: %s") % sid
    #     session_dir = os.path.join(PICKLE_DIR, sid)
    #
    #     session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))
    #
    #     # Legend info
    #     info = session_info_df.loc[sid]
    #
    #
    #     keys.append(val)
    #     orte_dfs[val] = {}
    #


    all_kv_dict = {}

    for sid in sids:

        print "Sid: %s" % sid

        session_dir = os.path.join(PICKLE_DIR, sid)

        unit_info_df = pd.read_pickle(os.path.join(session_dir, 'unit_info.pkl'))
        pilot_info_df = pd.read_pickle(os.path.join(session_dir, 'pilot_info.pkl'))
        tr_unit_prof_df = pd.read_pickle(os.path.join(session_dir, 'tr_unit_prof.pkl'))
        session_info_df = pd.read_pickle(os.path.join(session_dir, 'session_info.pkl'))

        # Legend info
        info = session_info_df.loc[sid]

        if key == 'pilot_cores':
            keyval = info['metadata.pilot_cores']
        else:
            print 'Unknown key: %s' % key
            exit(-1)

        if keyval not in all_kv_dict:
            print "First time I see this number of cu_cores: %d" % keyval
            all_kv_dict[keyval] = {}
        else:
            print "Already saw this number of cu_cores: %d" % keyval

        cu_runtime = info['metadata.cu_runtime']
        generations = info['metadata.generations']


        if cu_runtime not in all_kv_dict[keyval]:
            print "First time I see this value of cu_runtime: %d" % cu_runtime
            all_kv_dict[keyval][cu_runtime] = pd.Series()
        else:
            print "Already saw this value of cu_runtime: %d" % cu_runtime

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

        val = all_kv_dict[keyval][cu_runtime].append(pd.Series((tufs['aec_after_exec'].max() - tufs['asic_get_u_pend'].min())))
        startup = all_kv_dict[keyval][cu_runtime].append(pd.Series((tufs['aec_start_script'].min() - tufs['asic_get_u_pend'].min())))

        ttc_o = generations * cu_runtime * 1.0
        ttc_a = val
        cores = info['metadata.pilot_cores']
        freq = 60.0

        # $$ \frac {generations * unit runtime}{cores / freq + (generations * unit % runtime)}$$
        proj_eff = ttc_o / ( (cores / freq) + ttc_o) * 100


        if val[0] < ttc_o:
            # This likely means the pilot runtime was too short and we didn't complete all cu's
            print ("Einstein was wrong!?!")
            val = val/val
        else:
            val /= ttc_o
            val = 1 / val
            val *= 100

        real_eff = val
        delta = real_eff - proj_eff
        print "startup: %f, ttc_a: %f, ttc_o: %f, freq: %f, proj_eff: %f, real_eff: %f, delta: %f" % (
            startup, val, ttc_o, freq, proj_eff, real_eff, delta)


        all_kv_dict[keyval][cu_runtime] = delta

    colors = [cmap(i) for i in np.linspace(0, 1, len(all_kv_dict))]
    c = 0

    labels = []
    for key in sorted(all_kv_dict, key=int, reverse=False):

        # Skip small experiments
        if key <= 128:
            continue

        print 'orte_ttc raw:', all_kv_dict[key]
        #print 'orte_ttc mean:', orte_ttc.mean()
        orte_df = pd.DataFrame(all_kv_dict[key])
        print 'orte_ttc df:', orte_df

        #labels.append("%s" % resource_legend[key])
        labels.append("%s" % key)
        #ax = orte_df.mean().plot(kind='line', color=resource_colors[key], marker=resource_marker[key], fontsize=BARRIER_FONTSIZE, linewidth=BARRIER_LINEWIDTH)
        ax = orte_df.mean().plot(kind='line', marker='+', color=colors[c], fontsize=TICK_FONTSIZE, linewidth=LINEWIDTH)
        c += 1

    # ORTE only
    # Data for BW
    #mp.pyplot.plot((128, 256, 512, 1024, 2048, 4096, 8192), (305, 309, 309, 313, 326, 351, 558), 'b-+')
    # Data for Stampede
    #mp.pyplot.plot((128, 256, 512, 1024, 2048, 4096), (301, 303, 305, 311, 322, 344), 'b-+')
    #labels.append("ORTE-only (C)")

    # Horizontal reference
    # y_ref = 100
    # mp.pyplot.plot((0, 10000), (y_ref, y_ref), 'k--', linewidth=LINEWIDTH)
    # labels.append("Optimal")

    print 'labels: %s' % labels
    position = 'lower right'
    mp.pyplot.legend(labels, loc=position, fontsize=LEGEND_FONTSIZE, markerscale=0, labelspacing=0.2)
    if not paper:
        mp.pyplot.title("Resource efficiency for varying CU runtime.\n"
            "%d generations of a variable number of 'concurrent' CUs with a variable payload on a variable core pilot on %s.\n"
            "Constant number of %d sub-agent with %d ExecWorker(s) each.\n"
            "RP: %s - RS: %s - RU: %s"
           % (info['metadata.generations'], resource_label,
              info['metadata.num_sub_agents'], info['metadata.num_exec_instances_per_sub_agent'],
              info['metadata.radical_stack.rp'], info['metadata.radical_stack.rs'], info['metadata.radical_stack.ru']
              ), fontsize=8)
    mp.pyplot.xlabel("Unit Duration (s)", fontsize=LABEL_FONTSIZE)
    mp.pyplot.ylabel("Core Utilisation (\%)", fontsize=LABEL_FONTSIZE)
    #mp.pyplot.ylim(0, 105)
    #mp.pyplot.xlim(0, 4096)
    #mp.pyplot.ylim(290, 500)
    #mp.pyplot.ylim(0, 2000)
    #mp.pyplot.ylim(y_ref-10)
    #ax.get_xaxis().set_ticks([])
    #ax.get_xaxis.set
    #ax.set_yscale('log', basey=10)
    ax.set_xscale('log', basex=2)


    #width = 3.487
    width = 3.3
    #height = width / 1.618
    height = 2.7
    fig = mp.pyplot.gcf()
    fig.set_size_inches(width, height)
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=1)

    #fig.tight_layout(w_pad=0.0, h_pad=0.0, pad=0.1)
    fig.tight_layout(pad=0.1)

    mp.pyplot.savefig('plot_formula.pdf')

    mp.pyplot.close()


###############################################################################
#
if __name__ == '__main__':

    session_ids = [

        # Stampede SSH 3 gen
        # 'rp.session.radical.marksant.016884.0063',
        # 'rp.session.radical.marksant.016884.0064',
        # 'rp.session.radical.marksant.016884.0010',
        # 'rp.session.radical.marksant.016884.0058',
        # 'rp.session.radical.marksant.016884.0062',
        # 'rp.session.radical.marksant.016884.0057',
        # 'rp.session.radical.marksant.016884.0059',
        # 'rp.session.radical.marksant.016884.0040',
        # 'rp.session.radical.marksant.016884.0037',
        # 'rp.session.radical.marksant.016884.0060',
        # 'rp.session.radical.marksant.016884.0022',
        # 'rp.session.radical.marksant.016884.0021',
        # 'rp.session.radical.marksant.016884.0017',
        # 'rp.session.radical.marksant.016884.0018',
        # 'rp.session.radical.marksant.016884.0007',
        # 'rp.session.radical.marksant.016884.0006',
        # 'rp.session.radical.marksant.016884.0005',
        #
        # # 'rp.session.radical.marksant.016885.0001',
        # 'rp.session.radical.marksant.016885.0003',
        # 'rp.session.radical.marksant.016885.0004',
        # 'rp.session.radical.marksant.016885.0005',
        # 'rp.session.radical.marksant.016885.0006',
        # 'rp.session.radical.marksant.016885.0007',
        # 'rp.session.radical.marksant.016885.0008',
        # 'rp.session.radical.marksant.016885.0009',
        # 'rp.session.radical.marksant.016885.0010',
        # # 'rp.session.radical.marksant.016885.0011', # Duplicate
        # # 'rp.session.radical.marksant.016885.0012',
        # 'rp.session.radical.marksant.016885.0013',
        # # 'rp.session.radical.marksant.016885.0014', # Broken
        # # 'rp.session.radical.marksant.016885.0015',
        # 'rp.session.radical.marksant.016885.0016',
        # 'rp.session.radical.marksant.016885.0017',
        # 'rp.session.radical.marksant.016885.0018',
        # 'rp.session.radical.marksant.016885.0019',
        # 'rp.session.radical.marksant.016885.0020',
        # 'rp.session.radical.marksant.016885.0021',
        # 'rp.session.radical.marksant.016885.0022',
        # 'rp.session.radical.marksant.016885.0023',
        # 'rp.session.radical.marksant.016885.0024',
        # 'rp.session.radical.marksant.016885.0025',
        # 'rp.session.radical.marksant.016885.0026',
        # 'rp.session.radical.marksant.016885.0027',
        # 'rp.session.radical.marksant.016885.0028',
        # 'rp.session.radical.marksant.016885.0029',
        # 'rp.session.radical.marksant.016885.0030',
        # 'rp.session.radical.marksant.016885.0031',
        # 'rp.session.radical.marksant.016885.0032',
        # 'rp.session.radical.marksant.016885.0033',
        # 'rp.session.radical.marksant.016885.0034',
        # # 'rp.session.radical.marksant.016885.0035',
        # 'rp.session.radical.marksant.016885.0037',
        # 'rp.session.radical.marksant.016885.0038',
        # 'rp.session.radical.marksant.016885.0039',
        # 'rp.session.radical.marksant.016885.0040',
        #
        # 'rp.session.radical.marksant.016895.0000', # 16
        # # 'rp.session.radical.marksant.016895.0005', # 32
        # 'rp.session.radical.marksant.016895.0003', # 64
        # # 'rp.session.radical.marksant.016884.0058', # 128
        # # 'rp.session.radical.marksant.016895.0004', # 256
        # # 'rp.session.radical.marksant.016884.0040', # 512
        # # 'rp.session.radical.marksant.016884.0022', # 1024
        # 'rp.session.radical.marksant.016895.0001', # 2048
        # # 'rp.session.radical.marksant.016895.0006', # 4096
        # 'rp.session.radical.marksant.016895.0007', # 8192
        # 'rp.session.radical.marksant.016895.0017',
        # 'rp.session.radical.marksant.016895.0016',
        # 'rp.session.radical.marksant.016895.0015',
        # 'rp.session.radical.marksant.016895.0014',
        # 'rp.session.radical.marksant.016895.0013',
        # 'rp.session.radical.marksant.016895.0012',
        # 'rp.session.radical.marksant.016895.0011',
        # 'rp.session.radical.marksant.016895.0018',
        # 'rp.session.radical.marksant.016895.0019',
        # 'rp.session.radical.marksant.016895.0020',
        # 'rp.session.radical.marksant.016895.0021',
        # 'rp.session.radical.marksant.016895.0022',
        # 'rp.session.radical.marksant.016895.0023',
        # 'rp.session.radical.marksant.016895.0024',
        # 'rp.session.radical.marksant.016895.0025',
        # 'rp.session.radical.marksant.016895.0026',
        # 'rp.session.radical.marksant.016896.0000',
        # 'rp.session.radical.marksant.016896.0001',
        # 'rp.session.radical.marksant.016896.0002',
        # 'rp.session.radical.marksant.016896.0003',
        # 'rp.session.radical.marksant.016896.0004',
        # 'rp.session.radical.marksant.016896.0005',
        # 'rp.session.radical.marksant.016896.0006',
        # 'rp.session.radical.marksant.016896.0007',
        # 'rp.session.radical.marksant.016896.0008',
        # 'rp.session.radical.marksant.016896.0010',
        # 'rp.session.radical.marksant.016896.0011',
        # 'rp.session.radical.marksant.016896.0012',

        # Stampede SSH 1 gen
        'rp.session.radical.marksant.016882.0001',
        'rp.session.radical.marksant.016882.0002',
        'rp.session.radical.marksant.016882.0003',
        'rp.session.radical.marksant.016882.0004',
        'rp.session.radical.marksant.016882.0005',
        'rp.session.radical.marksant.016882.0006',
        'rp.session.radical.marksant.016882.0007',
        'rp.session.radical.marksant.016882.0008',
        'rp.session.radical.marksant.016882.0009',
        'rp.session.radical.marksant.016882.0010',
        'rp.session.radical.marksant.016882.0011',
        'rp.session.radical.marksant.016882.0012',
        'rp.session.radical.marksant.016882.0013',
        'rp.session.radical.marksant.016882.0014',
        'rp.session.radical.marksant.016882.0015',
        'rp.session.radical.marksant.016882.0016',
        'rp.session.radical.marksant.016882.0017',
        'rp.session.radical.marksant.016883.0018',
        'rp.session.radical.marksant.016883.0019',
        'rp.session.radical.marksant.016883.0020',
        'rp.session.radical.marksant.016883.0021',
        'rp.session.radical.marksant.016883.0026',
        'rp.session.radical.marksant.016883.0027',
        'rp.session.radical.marksant.016883.0037',
        'rp.session.radical.marksant.016883.0038',
        'rp.session.radical.marksant.016883.0039',
        'rp.session.radical.marksant.016883.0040',
        # 'rp.session.radical.marksant.016883.0041', # failing
        'rp.session.radical.marksant.016883.0048',
        'rp.session.radical.marksant.016883.0049',
        'rp.session.radical.marksant.016883.0050',
        'rp.session.radical.marksant.016883.0051',
        'rp.session.radical.marksant.016883.0052',
        'rp.session.radical.marksant.016883.0053',
        'rp.session.radical.marksant.016883.0056',
        'rp.session.radical.marksant.016883.0057',
        'rp.session.radical.marksant.016883.0058',
        'rp.session.radical.marksant.016883.0059',
        'rp.session.radical.marksant.016883.0060',
        'rp.session.radical.marksant.016883.0061',
        'rp.session.radical.marksant.016883.0062',
        'rp.session.radical.marksant.016883.0063',
        'rp.session.radical.marksant.016883.0066',
        'rp.session.radical.marksant.016883.0067',
        'rp.session.radical.marksant.016883.0068',
        # 'rp.session.radical.marksant.016883.0070', #  empty profiles
        'rp.session.radical.marksant.016883.0071',
        'rp.session.radical.marksant.016883.0072',
        'rp.session.radical.marksant.016883.0073',
        'rp.session.radical.marksant.016883.0074',
        'rp.session.radical.marksant.016883.0075',
        'rp.session.radical.marksant.016883.0076',
        'rp.session.radical.marksant.016883.0077',
        'rp.session.radical.marksant.016883.0078',
        'rp.session.radical.marksant.016883.0079',
        'rp.session.radical.marksant.016883.0080',
        # 'rp.session.radical.marksant.016883.0081',
        # 'rp.session.radical.marksant.016883.0082',
        'rp.session.radical.marksant.016883.0083',
        # 'rp.session.radical.marksant.016883.0084',
        # 'rp.session.radical.marksant.016883.0085',
        # 'rp.session.radical.marksant.016883.0086',
        # 'rp.session.radical.marksant.016883.0087',
        # 'rp.session.radical.marksant.016883.0088',
        # 'rp.session.radical.marksant.016883.0089',
    ]

    plot(session_ids, key='pilot_cores', paper=True)
