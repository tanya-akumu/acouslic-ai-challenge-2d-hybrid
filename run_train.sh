#!/bin/bash

# USAGE ./run_docker_containers.sh /home/sauron/Projects/2d_classif radimagenet_opt_sub
# stop: docker ps -a --filter "name=acoustic" --format "{{.ID}}" | xargs -r docker stop


# Check if the correct number of parameters is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <project_directory> <argument>"
    exit 1
fi

# Get the parameters
project_directory=$1
argument=$2

# Number of CPUs per set
cpu_set_size=2

# Array of CUDA devices to use for each Docker container
cuda_devices=("1" "2" "3" "4")

# Total number of CPUs
total_cpus=40

# Total number of runs
total_runs=5

# Loop to start Docker containers with different consecutive CPU sets and run the Python script
for i in $(seq 0 $((total_runs - 1))); do
    start_cpu=$(( (i * cpu_set_size) % total_cpus ))
    end_cpu=$((start_cpu + cpu_set_size - 1))
    cpu_set="${start_cpu}-${end_cpu}"
    container_name="tanyacoustic$((i))"
    
    # Set the CUDA device for this container
    cuda_device="${cuda_devices[$((i % ${#cuda_devices[@]}))]}"
    echo "Run $i: CPU set $cpu_set, CUDA device $cuda_device"

    # Run the Docker container with the specified CPU set and CUDA device
    docker run --gpus all -dit --rm --cpuset-cpus="${cpu_set}" \
        -v "${project_directory}:/workspace/2d" \
        -v /home/tanya-akumu/Challenge/fetal_abdominal_challenge/raw_data:/workspace/raw_data \
        --name "${container_name}" --env CUDA_VISIBLE_DEVICES="${cuda_device}" nnunet_acoustic
done

for i in $(seq 0 $((total_runs - 1))); do
    container_name="acoustic$((i))"
    # Execute the Python script inside the running Docker container
    docker exec -d "${container_name}" python3 /workspace/2d/main.py $((i)) "${argument}"
done



# if ! main "$@"; then
#     echo "An error occured. Cleaning up..."
#     cleanup
#     exit 1
# fi
