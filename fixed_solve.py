""" Solve a layout problem with CMA-ES with the option of setting one or more
coordinates to fixed values. 

To fix a coordinate, set the corresponding index in fixed_indices to the index of
those coordinates in the flattened array, and set the corresponding value in a
list in the same order.

The function 'expand_genome' will take the flattened array and expand it into a
the full thing, replacing the fixed values with the ones provided.

"""
    
import json
import cma
from FacilityPlacementTask import FacilityPlacementTask
import numpy as np
import time
import unittest

def expand_genome(flat_genome, fixed_indices, fixed_values):
    """
    Expand the flattened genome into the full genome.

    1) Create a blank genome of the correct size
    2) insert the fixed values into the correct indices
    3) fill in the rest of the values in order
    """
    if len(fixed_indices) != len(fixed_values):
        raise ValueError("fixed_indices and fixed_values must be the same length")
    if len(fixed_indices) == 0:
        return flat_genome    

    genome = [None] * (len(flat_genome)+len(fixed_indices))
    for index, value in zip(fixed_indices, fixed_values):
        genome[index] = value
    genome_index = 0
    for i in range(len(genome)):
        if genome[i] is None:
            genome[i] = flat_genome[genome_index]
            genome_index += 1
    return np.array(genome)  # Convert to numpy array for consistency

class TestExpandGenome(unittest.TestCase):
    def test_expand_genome(self):
        flat_genome = [1, 2, 3]
        fixed_indices = [0, 3]
        fixed_values = [0, 4]
        expected = np.array([0, 1, 2, 4, 3])
        result = expand_genome(flat_genome, fixed_indices, fixed_values)
        np.testing.assert_array_equal(result, expected)

# -- CMA-ES with Restarts -- #
def cmaes_with_restarts(task, max_fevals, fixed_indices=[], fixed_values=[]):
    bestever = cma.optimization_tools.BestSolution()
    num_facilities = len(task.Facillities)
    initial_guess = np.zeros( (2 * num_facilities)-len(fixed_indices) )
    sigma = 50.0
    options = {'bounds': [0, 100], 'maxfevals': max_fevals}
    popsize = None
    cumulative_fevals = 0

    # Additional logging information
    best_objective_per_generation = []
    wall_time_per_generation = []
    evaluations_per_generation = []

    start_time = time.time()

    while cumulative_fevals < max_fevals:
        current_options = options.copy()
        if popsize is not None:
            current_options['popsize'] = popsize
        
        es = cma.CMAEvolutionStrategy(initial_guess, sigma, current_options)
        
        while not es.stop() and cumulative_fevals < max_fevals:
            X = es.ask()
            X_eval = [expand_genome(x, fixed_indices, fixed_values) for x in X]
            fit = [-task.evaluate_fitness(x[:num_facilities], x[num_facilities:]) for x in X_eval]
            es.tell(X, fit)
            es.disp()
            cumulative_fevals += len(X)

            # Log the best objective and wall time at each generation
            if not best_objective_per_generation or -es.result.fbest > best_objective_per_generation[-1]:
                best_objective_per_generation.append(-es.result.fbest)
            else:
                best_objective_per_generation.append(best_objective_per_generation[-1])

            wall_time_per_generation.append(time.time() - start_time)
            evaluations_per_generation.append(cumulative_fevals)

        es.best.geno = es.best.x
        es.best.x = expand_genome(es.best.x, fixed_indices, fixed_values)
        bestever.update(es.best)

        if bestever.f < -0.99999999:  # if global optimum reached
            print("\n[*] Global optimum reached after %d fevals" % cumulative_fevals)
            break

        if popsize is None:  # Initialize for the first time
            popsize = es.sp.popsize
        popsize *= 2  # Double for the next iteration

        print("\n\t[*] Restarting with popsize = %d" % popsize)
            
    return (bestever.x, cumulative_fevals, best_objective_per_generation,
            wall_time_per_generation, evaluations_per_generation)


def fixed_run(task_file='tasksets/task_1.json', task_name='test_task', output_file='results.json',
              max_fevals=500, fixed_indices=[], fixed_values=[]):
    task = FacilityPlacementTask.load_from_json(json.load(open(task_file, 'r')), task_name)
    results = cmaes_with_restarts(task, max_fevals, fixed_indices, fixed_values)

    # Save the results to a file
    with open(output_file, 'w') as f:
        json.dump({'best_x': results[0].tolist(),
                   'cumulative_fevals': results[1],
                   'best_objective_per_generation': results[2],
                   'wall_time_per_generation': results[3],
                   'evaluations_per_generation': results[4],
                   'fixed_indices': fixed_indices,
                   'fixed_values': fixed_values}, f, indent=4)
        
    print(f"\n[*] Results saved to '{output_file}'")


if __name__ == '__main__':
    #unittest.main(argv=[''], exit=False)
    #fixed_run(fixed_indices=[0,1,2,3], fixed_values=[35,95,6,15])

    # To run on command line
    import fire # pip install fire
    fire.Fire(fixed_run)
    # example command:
    # python fixed_run.py --task_file='tasksets/task_1.json' --task_name='test_task' --output_file='results.json' --max_fevals=500 --fixed_indices=[0,1,2,3] --fixed_values=[35,95,6,15]

    
