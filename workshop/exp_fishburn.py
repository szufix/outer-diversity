
from src.vizualization.microscope import *

exp_name = 'fishburn'

COLORING_FUNCTIONS = {
    # 'first_candidate':  microscope_colored_by_first_candidate,
    # 'last_candidate':  microscope_colored_by_last_candidate,
    'position_of_candidate': microscope_colored_by_position_of_candidate,
}

sampler_names = ['largest_fishburn']
candidate_range = range(2, 10+1)

# compute_microscope(sampler_names, candidate_range)

for function_name, microscope_function in COLORING_FUNCTIONS.items():

    plot_microscopes(
        sampler_names=sampler_names,
        candidate_range=candidate_range,
        microscope_function=microscope_function,
        microscope_function_params={},
    )

    paths = []
    for num_candidates in candidate_range:
        for sampler_name in sampler_names:
            paths.append(f'images/online/{sampler_name}_m{num_candidates}.png')

    output_path = f'images/microscope/{exp_name}_microscope_{function_name}.png'
    create_image_grid(paths, len(candidate_range),1, output_path)


