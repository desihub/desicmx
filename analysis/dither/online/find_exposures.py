#! /usr/bin/env python
"""
Find exposures using the DOS database. This is a 'module-ized' version of the
script in /software/products/Scripts-trunk/bin/find_exposures. It is updated
to allow users to specify lists of observation days, exposure IDs, and 
sequence types.
"""
from DOSlib.util import simple_log, obs_day
import re, time, os, psycopg2, sys
import numpy as np


def find_exposures(exp_id=None, seq=None, obsdate=None):
    """Query the exposure database at KPNO for info related to exposures.

    Parameters
    ----------
    exp_id : int or list
        List of exposures to query.
    seq : str or list
        Exposure sequence type (e.g., GFA, FVC, ...).
    obsdate : str or list
        Date string(s) in format YYYYMMDD.

    Returns
    -------
    records : list
        List of returned records (id, night, seq, program).
    """
    db_query = 'SELECT id, night, sequence, program FROM exposure.exposure WHERE '
    search_str = []

    # Add exposure ID(s) to query.
    if exp_id is not None:
        if isinstance(exp_id, (list, np.ndarray)):
            ids_ = ' OR '.join(['id = {}'.format(ex) for ex in exp_id])
        else:
            ids_ = 'id = {}'.format(exp_id)
        search_str.append('({})'.format(ids_))

    # Add sequence(s) to query.
    if seq is not None:
        if isinstance(seq, (list, np.ndarray)):
            seq_ = ' OR '.join(['sequence = \'{}\''.format(s) for s in seq])
        else:
            seq_ = 'sequence = \'{}\''.format(seq)
        
        if search_str:
            search_str.append('AND')
        search_str.append('({})'.format(seq_))

    # Add obsdate(s) to query.
    if obsdate is not None:
        if isinstance(obsdate, (list, np.ndarray)):
            obs_ = ' OR '.join(['night = {}'.format(o) for o in obsdate])
        else:
            obs_ = 'night = {}'.format(obsdate)
        
        if search_str:
            search_str.append('AND')
        search_str.append('({})'.format(obs_))

    # Complete and send the query.
    db_query = '{} {} order by id asc limit 2000'.format(
        db_query, ' '.join(search_str))

    if 'DOS_DB_NAME' in os.environ:
        # Connect to DB.
        conn = psycopg2.connect(dbname=os.environ['DOS_DB_NAME'],
                                host=os.environ['DOS_DB_HOST'],
                                port=os.environ['DOS_DB_PORT'],
                                user=os.environ['DOS_DB_READER'],
                                password=os.environ['DOS_DB_READER_PASSWORD'])
    else:
        # Very bad... but hack needed to access DB via jupyter notebook at KPNO.
        conn = psycopg2.connect(dbname='desi_dev',
                                host='desi-db',
                                port=5442,
                                user='desi_reader',
                                password='reader')
    cur = conn.cursor()

    cur.execute(db_query)
    records = cur.fetchall()

    if not records:
        raise SystemExit('No records found.\nQuery: {}'.format(db_query))

    return records


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='DOS exposure finder')
    parser.add_argument('-e', '--expid', dest='expid', nargs='*',
                        help='Exposure ID(s)')
    parser.add_argument('-s', '--sequence', dest='sequence', nargs='*',
                        help='Sequence type [GFA, FVC, Spectrograph...]')
    parser.add_argument('-o', '--obsdate', dest='obsdate', nargs='*',
                        help='Observation date(s) [YYYYMMDD]')
    args = parser.parse_args()

    # Get exposure records, spin through them, and pretty-print results.
    records = find_exposures(exp_id=args.expid, seq=args.sequence, obsdate=args.obsdate)
    for rec in records:
        id_, date_, seq_, prog_ = rec
        record = '{:8d}'.format(id_)
        if date_ is not None:
            record = '{} {:12d}'.format(record, date_)
        else:
            record = '{} {:^12s}'.format(record, '--')
        record = '{} {:^15s}'.format(record, seq_)
        record = '{}  {:s}'.format(record, prog_)

        print(record)
