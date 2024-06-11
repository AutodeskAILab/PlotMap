#!/bin/bash

# Define the file name patterns
file_names=(
    "30_constraints_1000_polygons_10_facilities"
    "30_constraints_1000_polygons_30_facilities"
    "30_constraints_1000_polygons_60_facilities"
    "60_constraints_1000_polygons_10_facilities"
    "60_constraints_1000_polygons_30_facilities"
    "60_constraints_1000_polygons_60_facilities"
    "90_constraints_1000_polygons_10_facilities"
    "90_constraints_1000_polygons_30_facilities"
    "90_constraints_1000_polygons_60_facilities"
)

# Define the task ranges
task_ranges=(
    "1,250"
    "250,500"
    "501,750"
    "751,1000"
)

# task_ranges=(
#     "1,2"
#     "3,4"
#     "5,6"
# )

# Loop through file names
for file_name in "${file_names[@]}"; do
    # Loop through task ranges
    for range in "${task_ranges[@]}"; do
        # Split range into start and stop
        IFS=',' read -r start stop <<< "$range"
        # Construct and execute the Python command in the background
        python_command="python evolutionary_solve.py --task_start=$start --task_stop=$stop --task_set='$file_name'"
        # print the command
        echo $python_command
        eval $python_command &

    done
done

# Wait for all background processes to finish
wait