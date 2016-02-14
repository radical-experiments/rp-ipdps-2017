import os
import sys
import glob
import radical.pilot.states as rps
import radical.pilot.utils as rpu
import radical.utils as ru
import pandas as pd
from multiprocessing import Pool

from common import JSON_DIR, TARGET_DIR, PICKLE_DIR, HDF5_DIR

# Global Pandas settings
pd.set_option('display.width', 200)
pd.set_option('io.hdf.default_format','table')

###############################################################################
# Convert from unicode to strings
def convert(input):
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

###############################################################################
#
def json2frame(db, sid):

    json_docs = convert(rpu.get_session_docs(db, sid, cachedir=JSON_DIR))

    session_info = pd.io.json.json_normalize(json_docs['session'])
    session_info.set_index('_id', inplace=True)
    session_info.index.name = None

    unit_info = pd.io.json.json_normalize(json_docs['unit'])
    unit_info.set_index('_id', inplace=True)
    unit_info.index.name = None
    unit_info.insert(0, 'sid', sid)

    pilot_info = pd.io.json.json_normalize(json_docs['pilot'])
    pilot_info.set_index('_id', inplace=True)
    pilot_info.index.name = None
    pilot_info.insert(0, 'sid', sid)

    return session_info, pilot_info, unit_info

###############################################################################
#
def find_profiles(sid):

    session_dir = os.path.join(TARGET_DIR, sid)
    profile_paths = glob.glob('%s/*.prof' % session_dir)
    profile_paths.extend(glob.glob('%s/pilot.*/*.prof' % session_dir))
    if not profile_paths:
        raise Exception("No session files found in directory %s" % session_dir)

    print "Found profiling files in %s: %s" % (session_dir, profile_paths)

    return profile_paths

###############################################################################
#
def preprocess(sid):

    session_dir = os.path.join(PICKLE_DIR, sid)

    if os.path.isdir(session_dir):
        report.warn("Session dir '%s' already exists, skipping session." % session_dir)
        return

    try:
        sid_profiles = find_profiles(sid)
        print sid_profiles
        report.info("Combining profiles for session: %s.\n" % sid)
        combined_profiles = rpu.combine_profiles(sid_profiles)
        uids = set()
        for p in combined_profiles:
            uids.add(p['uid'])

        report.info("Converting profiles to frames for session: %s.\n" % sid)
        frames = rpu.prof2frame(combined_profiles)

        report.info("Head of Combined DF for session %s:\n" % sid)
        print frames.entity.unique()

        ses_prof_fr, pilot_prof_fr, cu_prof_fr = rpu.split_frame(frames)

        report.info("Head of Session DF for session %s:\n" % sid)
        ses_prof_fr.insert(0, 'sid', sid)
        print ses_prof_fr.head()

        report.info("Head of Pilot DF for session %s:\n" % sid)
        pilot_prof_fr.insert(0, 'sid', sid)
        print pilot_prof_fr.head()

        report.info("Head of CU DF for session %s:\n" % sid)
        rpu.add_states(cu_prof_fr)
        print cu_prof_fr.head()
        report.info("Head of CU DF for session %s (after states added):\n" % sid)
        rpu.add_info(cu_prof_fr)
        print cu_prof_fr.head()
        report.info("Head of CU DF for session %s (after info added):\n" % sid)
        print cu_prof_fr.head()

        report.info("Head of CU DF for session %s (after concurrency added):\n" % sid)

        # Add a column with the number of concurrent populating the database
        spec = {
            'in': [
                {'state': rps.STAGING_INPUT, 'event': 'advance'}
            ],
            'out' : [
                {'state':rps.AGENT_STAGING_INPUT_PENDING, 'event': 'advance'},
                {'state':rps.FAILED, 'event': 'advance'},
                {'state':rps.CANCELED, 'event': 'advance'}
            ]
        }
        rpu.add_concurrency (cu_prof_fr, 'cc_populating', spec)

        # Add a column with the number of concurrent staging in units
        spec = {
            'in': [
                {'state': rps.AGENT_STAGING_INPUT, 'event': 'advance'}
            ],
            'out' : [
                {'state':rps.ALLOCATING_PENDING, 'event': 'advance'},
                {'state':rps.FAILED, 'event': 'advance'},
                {'state':rps.CANCELED, 'event': 'advance'}
            ]
        }
        rpu.add_concurrency (cu_prof_fr, 'cc_stage_in', spec)

        # Add a column with the number of concurrent scheduling units
        spec = {
            'in': [
                {'state': rps.ALLOCATING, 'event': 'advance'}
            ],
            'out' : [
                {'state':rps.EXECUTING_PENDING, 'event': 'advance'},
                {'state':rps.FAILED, 'event': 'advance'},
                {'state':rps.CANCELED, 'event': 'advance'}
            ]
        }
        rpu.add_concurrency (cu_prof_fr, 'cc_sched', spec)

        # Add a column with the number of concurrent Executing units
        spec = {
            'in': [
                {'state': rps.EXECUTING, 'event': 'advance'}
            ],
            'out' : [
                {'state':rps.AGENT_STAGING_OUTPUT_PENDING, 'event': 'advance'},
                {'state':rps.FAILED, 'event': 'advance'},
                {'state':rps.CANCELED, 'event': 'advance'}
            ]
        }
        rpu.add_concurrency (cu_prof_fr, 'cc_exec', spec)

        # Add a column with the number of concurrent Executing units
        spec = {
            'in': [
                {'state': rps.AGENT_STAGING_OUTPUT, 'event': 'advance'}
            ],
            'out' : [
                {'state':rps.PENDING_OUTPUT_STAGING, 'event': 'advance'},
                {'state':rps.FAILED, 'event': 'advance'},
                {'state':rps.CANCELED, 'event': 'advance'}
            ]
        }
        rpu.add_concurrency (cu_prof_fr, 'cc_stage_out', spec)

        print cu_prof_fr.head()

        report.info("Head of CU DF for session %s (after sid added):\n" % sid)
        cu_prof_fr.insert(0, 'sid', sid)
        print cu_prof_fr.head()

        report.info("CU DF columns for session %s:\n" % sid)
        print cu_prof_fr['info'].unique()

        # transpose
        tr_cu_prof_fr = rpu.get_info_df(cu_prof_fr)
        tr_cu_prof_fr.insert(0, 'sid', sid)
        report.info("Head of Transposed CU DF for session %s:\n" % sid)
        print tr_cu_prof_fr.head()

        ses_info_fr, pilot_info_fr, unit_info_fr = json2frame(db=None, sid=sid)
        report.info("Head of json Docs for session %s:\n" % sid)
        print ses_info_fr.head()

    except:
        report.error("Failed to pre-process data for session %s" % sid)
        return

    report.header("Writing dataframes to disk.\n")
    try:

        os.mkdir(session_dir)

        ses_info_fr.to_pickle(os.path.join(session_dir, 'session_info.pkl'))
        pilot_info_fr.to_pickle(os.path.join(session_dir, 'pilot_info.pkl'))
        unit_info_fr.to_pickle(os.path.join(session_dir, 'unit_info.pkl'))
        ses_prof_fr.to_pickle(os.path.join(session_dir, 'session_prof.pkl'))
        pilot_prof_fr.to_pickle(os.path.join(session_dir, 'pilot_prof.pkl'))
        cu_prof_fr.to_pickle(os.path.join(session_dir, 'unit_prof.pkl'))
        tr_cu_prof_fr.to_pickle(os.path.join(session_dir, 'tr_unit_prof.pkl'))
    except:
        report.error("Failed to write data")
        return


###############################################################################
#
def preprocess_all(session_ids):

    pool = Pool()

    pool.map(preprocess, session_ids)



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

    report = ru.Reporter("Inject profiling and json data into database.")

    session_ids = []

    # Read from file if specified, otherwise read from stdin
    f = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    for line in f:
        session_ids.append(line.strip())

    if not session_ids:
        session_ids = find_sessions(JSON_DIR)

    preprocess_all(session_ids)
