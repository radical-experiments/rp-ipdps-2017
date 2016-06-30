import os
import sys
import radical.pilot.utils as rpu
import radical.utils as ru
import pandas as pd

from common import TARGET_DIR, CLIENT_DIR, JSON_DIR

###############################################################################
#
def collect(sid):

    # Default
    access = None
    client_dir = CLIENT_DIR
    target_dir = TARGET_DIR

    # If we run from the titan headnode, collect over GO
    if 'rp.session.titan' in sid:
        access = 'go://olcf#dtn'
        #access = 'go://olcf#dtn_atlas'
        client_dir = 'go://olcf#dtn/ccs/home/marksant1/ccgrid16/client'
        #client_dir = 'go://olcf#dtn_atlas/ccs/home/marksant1/ccgrid16/client'
        target_dir = 'go://localhost%s' % TARGET_DIR

    # If we ran on another client, only change the client
    elif 'rp.session.ip-10-184-31-85.santcroos' in sid:
        client_dir = 'sftp://ec2-107-21-218-167.compute-1.amazonaws.com/home/santcroos/experiments/ccgrid16/client'

    # elif 'rp.session.radical.marksant' in sid:
    #     client_dir = 'sftp://radserv/home/marksant/sc16/client'

    elif 'mw.session.netbook.mark' in sid:
        client_dir = '/Users/mark/proj/openmpi/mysubmit/client'

    elif 'mw.session' in sid and 'stampede.tacc.utexas.edu.marksant' in sid:
        client_dir = 'sftp://stampede/work/01740/marksant/client'

    elif 'mw.session.h2ologin' in sid or 'mw.session.nid' in sid:
        client_dir = 'gsisftp://bw/u/sciteam/marksant/mysubmit/client'

    report.info("Collecting profiles for session: %s.\n" % sid)
    rpu.fetch_profiles(sid=sid, client=client_dir, tgt=target_dir,
                       access=access, skip_existing=True)

    report.info("Collecting json for session: %s.\n" % sid)
    rpu.fetch_json(sid, tgt=JSON_DIR, skip_existing=True)


###############################################################################
#
def collect_all(sessions_to_fetch):

    for sid in sessions_to_fetch:
        try:
            collect(sid)
        except Exception as e:
            report.error("Collection of info for %s failed" % sid)


###############################################################################
#
if __name__ == '__main__':

    report = ru.Reporter("Collect profiling and json data to local disk.")

    session_ids = []

    # Read from file if specified, otherwise read from stdin
    f = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    for line in f:
        session = line.strip()
        if session:
            session_ids.append(session)

    report.info("Session ids found on input:\n")
    report.plain("%s\n" % session_ids)

    collect_all(session_ids)
