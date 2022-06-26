import numpy as np
import numpy.random as rnd
from pathlib import Path
from solve_and_serialize_cqm import *
#from solve_and_serialize_bqm import *
import time
from tqdm import tqdm


def find_folder(name):
    working_dir = Path.cwd()
    i = 0
    results_path = working_dir / (name + str(i))
    while(results_path.is_dir()):
        i += 1
        results_path = working_dir / (name + str(i))
    results_path.mkdir()
    print("You can find the results of this script in the folder", results_path)
    return results_path
    

def listlength(start, stop, step):
    if start < 1:
        print("the lenght of a list must be at least 1")
        return -1

    results_path = find_folder("Listlenght_results")
    start_time = time.time()
    for i in tqdm(range(start, stop+1, step)):
        x = rnd.randint(0, 100, i)
        v = x[0]
        result = solve_single_problem(x, v)
        data_path = results_path / ("Lenght" + str(i) + ".dir")
        serialize_result(result, data_path)
        if (time.time() - start_time) > (5 * 60):
            print("Time limit of 5 min. exceeded.")
            break


def listrange(start, stop, step):
    if start == stop:
        print("lowest and highest allowed should be different")
        return -1

    results_path = find_folder("Listrange_results")
    start_time = time.time()
    for i in tqdm(range(start, stop+1, step)):
        x = rnd.randint(-i, i, 50)
        v = x[0]
        result = solve_single_problem(x, v)
        data_path = results_path / ("Range" + str(i) + ".dir")
        serialize_result(result, data_path)
        if (time.time() - start_time) > (5 * 60):
            print("Time limit of 5 min. exceeded.")
            break
        
def hits(start, stop, step):
    if start < 0:
        print("number of hits cannot be negative")
    if start == stop:
        print("lowest and highest allowed should be different")
        return -1

    results_path = find_folder("Hits_results")
    start_time = time.time()
    for i in tqdm(range(start, stop+1, step)):
        x = rnd.randint(2, 100, stop-i)
        x = np.append(x, np.repeat(1,i))
        np.random.shuffle(x)
        v = 1
        result = solve_single_problem(x, v)
        data_path = results_path / ("Hits" + str(i) + ".dir")
        serialize_result(result, data_path)
        if (time.time() - start_time) > (5 * 60):
            print("Time limit of 5 min. exceeded.")
            break


if __name__ == '__main__':
    hits(41, 60, 1)
