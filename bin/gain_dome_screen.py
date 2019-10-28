import argparse
import json
import numpy as np

exptimes = np.arange(2, 34, 2).astype(float)

def all_observations(n_per_exptime):
    # can I specify track=False in this request, or is that only valid for slews?

    requests = []
    for exptime in exptimes:
        request = {"sequence": "GFA",
                   "flavor": "science",
                   "exptime": exptime,
                   "program": "gain dome screen",
                   "track": False}
        requests = requests + [request]*n_per_exptime

    return requests

def total_time_estimate(requests):
    t = 0.0 # seconds

    t_readout = 3.0 # seconds

    for request in requests:
        t += (request['exptime'] + t_readout)

    t_minutes = t/60.0

    return t_minutes

if __name__ == "__main__":
    descr = 'create script to gather closed-dome data that can be used to measure gain'
    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument('--n_per_exptime', default=4, type=int,
                        help='number of exposures per EXPTIME value, default 4')

    parser.add_argument('--outname', default='gain_dome_screen.json',
                        help='output file name for observing script')

    args = parser.parse_args()

    requests = all_observations(args.n_per_exptime)

    outname = args.outname

    print('GFA exposure sequence will take ' + "{:.1f}".format(total_time_estimate(requests)) + ' minutes')

    with open(outname, 'w') as outfile:
        json.dump(requests, outfile, indent=2)

