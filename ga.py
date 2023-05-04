
"""
TODO:
1. (DONE) initialize population with positive & negative values 
2. callback to keep track of each generation
3. (DONE) parallel_processing=None 
4. (DONE) plot the fitness of each generation 
5. (DONE) how does next generation created by crossover and mutation
6. mutation rate not working
7. (DONE) create a random pool of numbers and use index for position representation
8. (DONE) trajectory: sphere interpolation
    a. concern: how to balance dist & time
"""

"""
1. parent_selection_type="sss": The parent selection type. Supported types are sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection). 

2. crossover_type="single_point": The crossover type. Supported types are single_point, two_points, uniform, and shuffle.

3. mutation_type="random": The mutation type. Supported types are random, swap, scramble, inversion, and shuffle.

"""

############################################################################################################

#! DEBUG 
DEBUG = False

import pygad
import numpy as np
import trajectory_utils

if not DEBUG:
    from render import rendering
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    from scipy.spatial.transform import Rotation as R
    from PIL import Image
    import trimesh
    import math
    import json
    from pathlib import Path
    from torchngp import torchngp 

def instant_nerf(file):
    (psnr, lpips, loss) = torchngp(data_path="GA_random/055/",workspace="nerf_test",O_mode=True)
    # return np.random.randint(0,30)
    return psnr

def nerf_fitness(ga_instance, solution, solution_idx):
    output_dic = {}
    output_dic['camera_angle_y'] = renderer.camera.camera.yfov
    output_dic['h'] = 128
    output_dic['w'] = 128
    output_dic['frames'] = []

    # sols = solution.reshape(10,3)
    ####trajectory start####
    sols = []
    num_waypoints = len(solution)
    for i in range(num_waypoints-1):
        sols.extend(trajectory_utils.interpolate_arc(candidates[solution[i]], candidates[solution[i+1]], 5)[:-1])
    sols.append(candidates[solution[-1]])
    ####trajectory end####
    for i, pos in enumerate(sols):
        orientation = renderer.cam_from_positions(pos)
        renderer.update_camera_pose(pos, orientation)
        orientation_rad = orientation * (math.pi / 180)
        
        blender_pose = renderer.nerf_matrix(pos, orientation_rad, invert=False, to_opengl=False)
        R = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 7.549790126404332e-08, -1.0, 0.0],
                    [0.0, 1.0, 7.549790126404332e-08, 0.0],
                    [0.0, 0.0, 0.0, 1.0]]) #rotate 90 degrees in X to compensate; derived from mathutils.Matrix.Rotation()
        blender_pose = R @ blender_pose
        image = Image.fromarray(renderer.render())
        
        imgdir = os.path.join(output_path, name, "train")
        Path(imgdir).mkdir(parents=True, exist_ok=True)
        imgpath = os.path.join(imgdir, str(i)) + '.png'
        image.save(imgpath)
        tempd = {}
        tempd['file_path'] = imgpath 
        tempd['transform_matrix'] = blender_pose.tolist()
        output_dic['frames'].append(tempd)

    jsonpath = os.path.join(output_path, name, 'transforms_train.json')
    with open(jsonpath, 'w') as f:
        json.dump(output_dic, f)

    return instant_nerf(jsonpath)


# Define the fitness function
def test_fitness(solution, solution_idx):
    viewpoints = []
    num_waypoints = len(solution)
    # generate an arc interpolation
    for i in range(num_waypoints-1):
        viewpoints.extend(trajectory_utils.interpolate_arc(candidates[solution[i]], candidates[solution[i+1]], 5)[:-1])
    viewpoints.append(candidates[solution[-1]])
    return np.random.rand()

# Define a custom function to create the initial gene
def create_gene():
    # create a random 3D vector
    vector = np.random.rand(3) - 0.5
    return vector / np.linalg.norm(vector)

# Create a pool for the gene candidates
def create_gene_pool(size):
    pool = []
    for _ in range(size):
        pool.append(create_gene())
    return np.array(pool)


# Define the on_generation function
def on_generation(ga_instance):
    generation_num = ga_instance.generations_completed
    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    best_solution = best_solution.reshape(-1, 3)
    print(f"Generation: {generation_num}, Best solution: {best_solution}, Fitness: {best_solution_fitness}, Index: {best_solution_idx}")


if __name__ == "__main__":
    ############################# Parameters #############################
    num_generations = 100
    num_parents_mating = 10
    mutation_percent_genes = 10
    num_solutions = 20 
    num_genes = 10 # number of 3D vectors
    num_candidates = num_generations*num_solutions*num_genes
    candidates = create_gene_pool(num_candidates)
    print(candidates)
    ############################# Parameters #############################

    if DEBUG: 
        # Define the genetic algorithm parameters
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=test_fitness,
            sol_per_pop=num_solutions,
            num_genes=num_genes,
            crossover_type="single_point",
            mutation_percent_genes=mutation_percent_genes,
            mutation_type="random",
            parent_selection_type="sss",
            gene_type=int,
            gene_space=range(num_candidates),
            on_generation=None,
            save_best_solutions=True,
            parallel_processing=["thread", 1],
            random_seed=2
        )

        # Run the genetic algorithm
        ga_instance.run()

        # Get the best solution
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        answer = [candidates[i] for i in solution]
        print(f"Best solution index: {solution}, Best answer: {answer}, fitness: {solution_fitness}")

        # Plot the fitness
        ga_instance.plot_fitness()



    if not DEBUG:
        #! initializations
        renderer = rendering.Renderer([128, 128])

        #! paths
        object_path = 'data/mesh/55.obj'
        output_path = "GA_random"
        name = str(55).zfill(3)
        if not os.path.exists(os.path.join(output_path, name)):
            os.makedirs(os.path.join(output_path, name))

        #! load the object
        renderer.remove_objects()
        mesh = trimesh.load(object_path)
        renderer.add_object(mesh,  add_faces=True)

        # Define the genetic algorithm parameters
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=nerf_fitness,
            sol_per_pop=num_solutions,
            num_genes=num_genes,
            crossover_type="single_point",
            mutation_percent_genes=mutation_percent_genes,
            mutation_type="random",
            parent_selection_type="sss",
            gene_type=int,
            gene_space=range(num_candidates),
            on_generation=None,
            save_best_solutions=True,
            #parallel_processing=["thread", 1],
            random_seed=2
        )

        # Run the genetic algorithm
        ga_instance.run()

        ga_instance.plot_fitness()

        # Get the best solution
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f"Best solution: {solution}, fitness: {solution_fitness}, index: {solution_idx}")