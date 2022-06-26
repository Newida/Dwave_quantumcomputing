#Imports
import shutil
import json
from pathlib import Path
import time
import numpy as np
import itertools
from dwave.system import LeapHybridSampler 
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel
from dimod import *

def solve_single_problem(x, v, lagrange_list, direct = False):
    if lagrange_list is None:
        print("BQM solver needs lagrange multiplier. If you do not know what to put here you might want to use CQM instead")

    if not isinstance(lagrange_list, list):
        lagrange_list = list(itertools.repeat(lagrange_list, len(x)))

    #measure runtime of scriptcalculations
    start_time = time.time()
    result = {}
    #Build BQM
    #create variables
    z = np.arange(len(x))

    bqm = BinaryQuadraticModel('BINARY')

    #--------------Objective function---------
    #create binary variables and objective function
    for i in range(len(x)):
        bqm.add_variable(z[i], -1)

    #--------------Constraints--------
    #constraints
    for i in range(len(x)):
        c = [(z[i], x[i]-v)]
        bqm.add_linear_equality_constraint(c, constant = 0, lagrange_multiplier = lagrange_list[i])

    #--------------Solve by Sampling--------
    if direct:
        print("Solving with direct QPU")
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample(bqm)
    else:
        print("Solving with Hybrid Solver")
        sampler = LeapHybridSampler(solver={'category':'hybrid'}) 
        sampleset = sampler.sample(bqm, label="bqm_hybrid")

    #--------------save output--------
    result['problem'] = {"x": x, "v": v, "bqm": bqm, "lagrange:" : lagrange_list}
    
    #get lowest enery solution
    best_solution = sampleset.first
    #convert the solution to numpy array
    solution_array = np.array(list(sampleset.first.sample.values()))[:len(x)]
    #calculate classical solution
    classical_solution = np.zeros_like(solution_array)
    classical_solution[x == v] = 1

    result['solution'] = {"all": sampleset, "best": solution_array, "classical": classical_solution}
    result['solution_correctness'] = (True if np.linalg.norm(classical_solution - solution_array) < 1e-16 else False)
    result['time'] = (time.time() - start_time)
    result['sampler_properties'] = sampler.properties
    result['sampleset_info'] = sampleset.info

    return result

def testserialize():
    result = {}
    bqm = BinaryQuadraticModel('BINARY')
    z = np.arange(len(x))
    for i in range(len(x)):
        bqm.add_variable(z[i], -1)

    lagrange_list = [1,1,1,1]
    for i in range(len(x)):
        c = [(z[i], x[i])]
        bqm.add_linear_inequality_constraint(c, constant = -v, lagrange_multiplier = lagrange_list[i], label=str(i))

    result['problem'] = {"x": np.array([1,2,2,3]), "v": x[1], "bqm": bqm, "lagrange": lagrange_list} 
    sampleset = dimod.SampleSet.from_samples(np.ones(5, dtype='int8'), 'BINARY', 0)
    result['solution'] = {"all": sampleset, "best": np.array([0,1,1,0]), "classical": np.array([0,1,1,0])}
    result['solution_correctness'] = True
    result['time'] = 4.0 
    result['sampler_properties'] = {}
    result['sampleset_info'] = {}
    return result

def serialize_result(result, path):
    if path.is_dir():
        print("File already exists")
        return False

    path.mkdir()
    #serialize problem
    #create folders for dict
    problem_path = path / "problem"
    problem_path.mkdir()
    #serialize x and bqm
    with open(problem_path / "x.npy", 'wb') as file:
        np.save(file, result['problem']['x'])
    with open(problem_path / "bqm.dwave", 'wb') as file:
        shutil.copyfileobj(result['problem']['bqm'].to_file(), file)
    
    #serialize solution
    solution_path = path / "solution"
    solution_path.mkdir()
    #serialize all, feasible, best, classical
    with open(solution_path / "all.js", 'w') as file:
        json.dump(result['solution']['all'].to_serializable(), file)
    with open(solution_path / "best.npy", 'wb') as file:
        np.save(file, result['solution']['best'])
    with open(solution_path / "classical_solution.npy", 'wb') as file:
        np.save(file, result['solution']['classical'])

    #serialize everything else
    result.pop('solution')
    result['problem'].pop('x')
    result['problem'].pop('bqm')
    
    #v is a numpytype and needs to be converted first
    v = result['problem']['v']
    if isinstance(v, np.integer):
        result['problem']['v'] = int(v)
    if isinstance(v, np.floating):
        result['problem']['v'] = float(v)

    with open(path / "sub_dict.js", 'w') as file:
        json.dump(result, file)

    return True

def deserialize_result(path):
    if not path.is_dir():
        print("file could not be found")
        return None
    
    result = {}
    #load result_subdict
    with open(path / "sub_dict.js", 'r') as file:
        result = json.load(file)
    
    #load problem
    problem_path = path / "problem"
    with open(problem_path / "x.npy", 'rb') as file:
        result['problem']['x'] = np.load(file)
    with open(problem_path / "bqm.dwave", 'rb') as file:
        result['problem']['bqm'] = BinaryQuadraticModel.from_file(file)

    #load solution
    solution_path = path / "solution"
    result['solution'] = {}
    with open(solution_path / "all.js", 'r') as file:
        result['solution']['all'] = SampleSet.from_serializable(json.load(file))
    with open(solution_path / "best.npy", 'rb') as file:
        result['solution']['best'] = np.load(file)
    with open(solution_path / "classical_solution.npy", 'rb') as file:
        result['solution']['classical'] = np.load(file)
    
    return result

def measure_lagrange(x, v):
    lagrange_tests = [0.01, 1.01, len(x)]
    for l in lagrange_tests:
        result = solve_single_problem(x, v,l)
        if result['solution_correctness']:
            return (l, result)
    return (-1, result) #since l should not be negative the -1 encodes as an error
    

def testserialize_and_deserialize():
    workdir = Path.cwd() 
    x = np.array([1,2,2,3])
    v = x[1]
    if serialize_result(testserialize(), workdir / "test.dir"):
        print("Saved to ", workdir / "test.dir")
    else:
        print("Something went wrong")

    result = deserialize_result(workdir / "test.dir")
    print(result)


if __name__ == '__main__':
    workdir = Path.cwd() 
    x = np.array([1,2,2,3])
    v = x[1]
    if serialize_result(solve_single_problem(x, v, len(x), True), workdir / "test.dir"):
        print("Saved to ", workdir / "test.dir")
    else:
        print("Something went wrong")
    
    result = deserialize_result(workdir / "test.dir")
    
    print("-----------------serialized------------------")
    print(result["solution_correctness"])

    print(result["solution"])
