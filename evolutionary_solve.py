""" This is the code the fitness function used by the ES. It is a just a wrapped
version of the code in FacilityPlacementTask.py.
"""

import json
import cma
from FacilityPlacementTask import FacilityPlacementTask
import numpy as np
import time

def cmaes_with_restarts(task, max_fevals):
    bestever = cma.optimization_tools.BestSolution()
    num_facilities = len(task.Facillities)
    initial_guess = np.zeros(2 * num_facilities)
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
            fit = [-task.evaluate_fitness(x[:num_facilities], x[num_facilities:]) for x in X]
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


def run_experiments(task_start=1, task_stop=2, task_set='generated_tasks_10_terrain_10_constraints', max_fevals=5000):
    """
    Run CMA-ES on multiple tasks and record the results in a JSON file.
    """
    results = {
        'task_name': [],
        'coord': [],
        'objective': [],
        'fevals': [],
        'best_objective_per_generation': [],
        'wall_time_per_generation': [],
        'evaluations_per_generation': []
    }

    for i in range(task_start, task_stop+1):
        # Load Task
        task_name = f"task_{i}"
        task = FacilityPlacementTask.load_from_json(json.load(open(f'tasksets/{task_set}/{task_name}.json', 'r')), task_name)

        # Run Optimizer
        (optimized_coords, fevals, best_objective_gen,
         wall_time_gen, evals_gen) = cmaes_with_restarts(task, max_fevals=max_fevals)


        # Save Results
        num_facilities = len(task.Facillities)
        optimized_x = optimized_coords[:num_facilities]
        optimized_y = optimized_coords[num_facilities:]

        optimized_fitness = task.evaluate_fitness(optimized_x, optimized_y)
        print(f'Optimized fitness for {task_name}:', optimized_fitness)
        results['task_name'].append(task_name)
        results['coord'].append(optimized_coords.tolist())
        results['objective'].append(optimized_fitness)
        results['fevals'].append(fevals)
        results['best_objective_per_generation'].append(best_objective_gen)
        results['wall_time_per_generation'].append(wall_time_gen)
        results['evaluations_per_generation'].append(evals_gen)        
    
    # Write results to JSON file
    #
    fname = f'results_{task_set}_task{task_start}-{task_stop}.json'
    print("Writing results to", fname)
    with open(fname, 'w') as f:
        json.dump(results, f, indent=4)

import json
import matplotlib.pyplot as plt

def plot_run_results(fname, x_axis='evals'):
    """
    Load results from a JSON file and plot each run's optimization history by evaluations or wall time,
    saving the plot as a PNG with the same base file name, with lines in shades of blue.

    Args:
    - fname: The file name of the JSON containing the results.
    - x_axis: The x-axis type for the plot. 'evals' for evaluations, 'wall_time' for wall time.
    """
    with open(fname, 'r') as f:
        results = json.load(f)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    num_runs = len(results['task_name'])
    blue_shades = plt.cm.Blues(np.linspace(0.3, 1, num_runs))

    # Determine which data to use for the x-axis
    x_axis_data = results['wall_time_per_generation'] if x_axis == 'wall_time' else results['evaluations_per_generation']

    for i, (x_data, objectives) in enumerate(zip(x_axis_data, results['best_objective_per_generation'])):
        # Create a list to hold the true best ever values
        true_best_ever = [objectives[0]]
        for obj in objectives[1:]:
            true_best_ever.append(max(obj, true_best_ever[-1]))

        ax.plot(x_data, true_best_ever, color=blue_shades[i], alpha=0.8)

    ax.set_title('Optimization History Across Runs')
    ax.set_xlabel('Wall Time (seconds)' if x_axis == 'wall_time' else 'Number of Evaluations')
    ax.set_ylabel('Best Ever Fitness')

    plot_fname = f"{fname.rsplit('.', 1)[0]}.png"
    plt.savefig(plot_fname)
    plt.close(fig)

def summarize_experiments(fname='experiment_results.json'):
    # Open experimental results
    with open(fname, 'r') as f:
        results = json.load(f)

    # Filter only solved solutions
    solved_solutions = [x for x in results['objective'] if x == 1.0]
    solved_fevals = [feval for obj, feval in zip(results['objective'], results['fevals']) if obj == 1.0]

    # Print summary statistics for solved solutions
    print("\n--- Summary Statistics ---")
    print(f"Percent Perfectly Solved:  {len(solved_solutions) / len(results['objective']) * 100:.1f}%")
    print(f"Median Evals till Solved:  {int(round(np.median(solved_fevals))) if solved_fevals else 0}")
    print(f"Median Fitness:            {np.median(results['objective']):.3f}")

def cmaes(task):
    """
    Optimize the task using the CMA-ES algorithm.
    """
    num_facilities = len(task.Facillities)

    # Flatten x and y coordinates into a single array for CMA-ES
    def fitness_function(coords):
        x = coords[:num_facilities]
        y = coords[num_facilities:]
        return -task.evaluate_fitness(x, y)  # Negate because CMA-ES minimizes
    
    # Initial guess and standard deviation
    initial_guess = 50+np.zeros(2 * num_facilities)
    sigma = 20.0

    # Run CMA-ES
    options = {'bounds': [0, 100], 'maxfevals': 20000}
    es = cma.CMAEvolutionStrategy(initial_guess, sigma, options)
    es.optimize(fitness_function)
    return es.result[0], es.result[3]  # Return the best solution and fevals

def single_run():
    # Existing code for loading task
    task = FacilityPlacementTask.load_from_json(json.load(open('tasksets/generated_tasks_10_terrain_10_constraints/task_1.json', 'r')), "task_1")
    print('Initial sat value: ', task.evaluate())
    
    # Run CMA-ES optimization
    optimized_coords, fevals = cmaes_with_restarts(task)

    # Extract optimized x and y coordinates
    num_facilities = len(task.Facillities)
    optimized_x = optimized_coords[:num_facilities]
    optimized_y = optimized_coords[num_facilities:]

    # Evaluate fitness for optimized coordinates
    optimized_fitness = task.evaluate_fitness(optimized_x, optimized_y)
    print(f'Optimized fitness: {optimized_fitness}, Fevals: {fevals}')

    # # Rendering
    # from FacilityPlacementTaskRenderer import FacilityPlacementTaskRenderer
    # renderer = FacilityPlacementTaskRenderer(task)
    # renderer.render_task(0)    

if __name__ == '__main__':
    import fire
    fire.Fire(run_experiments)
    #run_experiments(task_start=1, task_stop=1, task_set='30_constraints_1000_polygons_10_facilities')
    #single_run()
    #plot_run_results(fname='results/results_90_constraints_1000_polygons_10_facilities.json', x_axis='wall_time')
    #plot_run_results(fname='results/results_90_constraints_1000_polygons_60_facilities.json', x_axis='wall_time')
    #run_experiments(task_set='90_constraints_1000_polygons_60_facilities')
    #plot_run_results(fname='results/results_90_constraints_1000_polygons_60_facilities.json')
    #summarize_experiments(fname='experiment_results_0.json')
