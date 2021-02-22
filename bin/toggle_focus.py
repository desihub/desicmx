import json
import copy

def _script(chunksize=1, repeats=15, dz=100.0, reverse=False):

    with open('exp.json') as f:
        data = json.load(f)

    plus = data[0]
    plus['focus'] = float(dz)

    if reverse:
        plus['program'] = plus['program'].replace('plus', 'minus')
    if chunksize > 1:
        plus['program'] = plus['program'] + ' chunksize 5'
    
    minus = copy.deepcopy(plus)
    minus['focus'] = -1*plus['focus']
    if chunksize == 1:
        unit = [plus] + [minus]
    else:
        neutral = copy.deepcopy(plus)
        neutral['focus'] = 0.0
        unit = [plus] + [neutral]*(chunksize-1) + [minus] + [neutral]*(chunksize-1)

    if reverse:
        #unit.reverse()
        for item in unit:
            item['focus'] = item['focus']*-1.0
    
    result = unit*repeats
    
    return result

def _write_sample(repeats=15, reverse=False, chunksize=1):

    result = _script(repeats=repeats, reverse=reverse,
                     chunksize=chunksize)

    outname = 'toggle_focus.json'

    if reverse:
        outname = outname.replace('.json', '-reverse.json')

    if chunksize != 1:
        outname = outname.replace('.json', '-' + str(chunksize) + '.json')
        
    with open(outname, 'w') as outfile:
        json.dump(result, outfile, indent=4)

    
