#Imports
import shutil
import json
from pathlib import Path
import time
import numpy as np
from dimod import Binary, ConstrainedQuadraticModel
from dimod import *
from dwave.system import LeapHybridCQMSampler

def solve_single_problem(x, v):
    #measure runtime of scriptcalculations
    start_time = time.time()
    result = {}
    #Build CQM
    cqm = ConstrainedQuadraticModel()

    #--------------Objective function---------
    #create binary variables
    z = [Binary(i) for i in range(len(x))]

    #objective function
    cqm.set_objective(-sum(z))

    #--------------Constraints--------
    #single constraint version
    cqm.add_constraint(sum((x[i]-v)**2 * z[i] for i in range(len(x))) == 0)

    #--------------Solve by Sampling--------
    sampler = LeapHybridCQMSampler()
    #make sure we do not need more time as necessary
    #sampler.min_time_limit(cqm) calculates the minimal time necessary. time_limit = min_time_limit(cqm) as default
    sampleset = sampler.sample_cqm(cqm, label="Select Statement")

    #--------------save output--------
    #result['problem'] = {"x": x, "v": v, "objective": cqm.objective, "constraints": cqm.constraints}
    result['problem'] = {"x": x, "v": v, "cqm": cqm} 
    feasible_solutions = sampleset.filter(lambda d: d.is_feasible)
    
    #get lowest enery feasable solution
    best_solution = feasible_solutions.first
    #convert the solution to numpy array
    solution_array = np.array(list(best_solution.sample.items()))[:, 1]
    #calculate classical solution
    classical_solution = np.zeros_like(solution_array)
    classical_solution[x == v] = 1
    
    result['solution'] = {"all": sampleset, "feasible": feasible_solutions, "best": solution_array, "classical": classical_solution}
    result['solution_correctness'] = (True if np.linalg.norm(classical_solution - solution_array) < 1e-16 else False)
    result['time'] = (time.time() - start_time)
    result['sampler_properties'] = sampler.properties
    result['sampleset_info'] = sampleset.info

    return result

def testserialize():
    result = {}
    cqm = ConstrainedQuadraticModel()
    z = [Binary(i) for i in range(len(x))]
    cqm.set_objective(-sum(z))
    cqm.add_constraint(sum((x[i]-v)**2 * z[i] for i in range(len(x))) == 0)
    result['problem'] = {"x": np.array([1,2,2,3]), "v": x[1], "cqm": cqm} 
    sampleset = None #dimod.SampleSet(np.recarray((3,), dtype=[('sample', float), ('energy', float), ('num_occurrences', float)]), np.arange(3), {}, 'BINARY')
    result['solution'] = {"all": sampleset, "feasible": sampleset, "best": np.array([0,1,1,0]), "classical": np.array([0,1,1,0])}
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
    #serialize x and cqm
    with open(problem_path / "x.npy", 'wb') as file:
        np.save(file, result['problem']['x'])
    with open(problem_path / "cqm.dwave", 'wb') as file:
        shutil.copyfileobj(result['problem']['cqm'].to_file(), file)
    
    #serialize solution
    solution_path = path / "solution"
    solution_path.mkdir()
    #serialize all, feasible, best, classical
    with open(solution_path / "all.js", 'w') as file:
        json.dump(result['solution']['all'].to_serializable(), file)
    with open(solution_path / "feasible.js", 'w') as file:
        json.dump(result['solution']['feasible'].to_serializable(), file)
    with open(solution_path / "best.npy", 'wb') as file:
        np.save(file, result['solution']['best'])
    with open(solution_path / "classical_solution.npy", 'wb') as file:
        np.save(file, result['solution']['classical'])

    #serialize everything else
    result.pop('solution')
    result['problem'].pop('x')
    result['problem'].pop('cqm')
    
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
    with open(problem_path / "cqm.dwave", 'rb') as file:
        result['problem']['cqm'] = ConstrainedQuadraticModel.from_file(file)

    #load solution
    solution_path = path / "solution"
    result['solution'] = {}
    with open(solution_path / "all.js", 'r') as file:
        result['solution']['all'] = SampleSet.from_serializable(json.load(file))
    with open(solution_path / "feasible.js", 'w') as file:
        result['solution']['feasible'] = SampleSet.from_serializable(json.load(file))
    with open(solution_path / "best.npy", 'rb') as file:
        result['solution']['best'] = np.load(file)
    with open(solution_path / "classical_solution.npy", 'rb') as file:
        result['solution']['classical'] = np.load(file)
    
    return result

if __name__ == '__main__':
    workdir = Path.cwd() 
    x = np.array([1,2,2,3])
    v = x[1]
    if serialize_result(testserialize(), workdir / "test.dir"):
        print("Saved to ", workdir / "test.dir")
    else:
        print("Something went wrong")

    result = deserialize_result(workdir / "test.dir")
    print(result)
    print(result['problem']['cqm'].objective)
    
