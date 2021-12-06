#!/bin/bash
source .dockerenv

it_params=''
gpu_params=''
print_usage() {
  echo "The purpose of this script is to run docker container with the proper parameters. The script has the following arguments:"
  echo "The script has the following arguments: "
  echo "     -c   the command to run inside the container"
  echo "          Example usage: ./docker_run.sh -c /bin/bash -i"
  echo "                         ./docker_run.sh -c python script.py"
  echo "     -g   Which GPU(s) to use, it will set the CUDA_VISIBLE_DEVICES env variable inside the container"
  echo "          Example usage: ./docker_run.sh -c /bin/bash -g 2 -i"
  echo "                         ./docker_run.sh -c python script.py -g 2"
  echo "     -i   Docker will run with -it parameter"
  echo "          Example usage: ./docker_run.sh -c /bin/bash -i"
}

while getopts ihg:c: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        c) command=${OPTARG};;
        g) gpu_params="-e CUDA_VISIBLE_DEVICES=${OPTARG}";;
        i) it_params="-it";;
        h) print_usage
           exit 0 ;;
    esac
done

IMAGE="${IMAGE:-$IMAGE_NAME}"
CONTAINER_ID=$(docker inspect --format="{{.Id}}" ${IMAGE} 2> /dev/null)

if [[ "${CONTAINER_ID}" ]]; then
    docker run --shm-size=10g --rm \
        $it_params $gpu_params \
        --gpus all \
        --user $(id -u):$(id -g) \
        -e PYTHONPATH=/workspace \
        -v `pwd`/../:/workspace \
        -v /home/${USER}/cache:/cache \
        -v /home/${USER}/.pycharm_helpers:/home/${USER}/.pycharm_helpers \
        -v /home/${USER}/.config/matplotlib:/home/${USER}/.config/matplotlib \
        -v /home/${USER}/.cache:/home/${USER}/.cache \
        -v /etc/sudoers:/etc/sudoers:ro \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        -v /etc/shadow:/etc/shadow:ro \
        --workdir=/workspace \
        $IMAGE $command
else
    echo "Unknown container image: ${IMAGE}, pull it first with docker pull ${IMAGE}"
    exit 1
fi
