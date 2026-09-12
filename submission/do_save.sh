#!/usr/bin/env bash
#
# Saves the algorithm's Docker image and model as tarballs for upload to
# Grand Challenge. Run this after do_test_run.sh confirms everything works.
#
# The image tarball is uploaded as the Algorithm's container image; model.tar.gz
# is uploaded SEPARATELY as a Model (Grand Challenge extracts it to /opt/ml/model
# at runtime). Expect ~4 GB for the image and ~7.6 GB for the model.

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="example_algorithm_preliminary-docker-evaluation-sanity-check"

# Where the nnU-Net checkpoints live. They are NOT in this repository (~7.6 GB);
# see README.md for the directory layout you need to supply.
MODEL_DIR="${ISLES_MODEL_DIR:-${SCRIPT_DIR}/model}"

# Disable promotional logs from Docker
export DOCKER_CLI_HINTS=false

# ---------------------------------------------------------------------------

# Log a message to stdout with color based on level (when running in a terminal).
#   info    = blue (default)
#   warning = yellow
#   error   = red
log() {
    local message="$1"
    local level="${2:-info}"
    if [[ -t 1 ]]; then
        case "$level" in
            info)    printf "\e[38;2;36;150;237m> %s\e[0m\n" "$message" ;;
            warning) printf "\e[38;2;255;200;0m> %s\e[0m\n" "$message" ;;
            error)   printf "\e[38;2;255;50;50m> %s\e[0m\n" "$message" ;;
            *)       printf "\e[38;2;36;150;237m> %s\e[0m\n" "$message" ;;
        esac
    else
        printf "%s\n" "$message"
    fi
}

# ---------------------------------------------------------------------------

if [ ! -d "$MODEL_DIR" ]; then
    log "Error: no model directory at ${MODEL_DIR}" error
    log "Set ISLES_MODEL_DIR, or populate ./model as described in README.md" error
    exit 1
fi

log "(Re)build the image"
export DOCKER_QUIET_BUILD=1
source "${SCRIPT_DIR}/do_build.sh"

# Get the build information from the Docker image tag
build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")

if [ -z "$build_timestamp" ]; then
    log "Error: Failed to retrieve build information for container $DOCKER_IMAGE_TAG"
    exit 1
fi

# Format the build information to remove special characters
formatted_build_info=$(echo "$build_timestamp" | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')

# Set the output filename with timestamp and build information
output_filename="${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"
output_path="${SCRIPT_DIR}/$output_filename"

# Save the Docker-container image and gzip it
log "Saving the image (this can take a while)"
docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$output_path"
log "Saved as: ${output_filename}"

# Create the model tarball
log "Packing the model (this can take a while)"
output_tarball_name="${SCRIPT_DIR}/model.tar.gz"
tar -czf "$output_tarball_name" -C "$MODEL_DIR" .
log "Saved as: model.tar.gz"

log "IMPORTANT: Please upload the model.tar.gz as a separate Model to your Algorithm!" warning
