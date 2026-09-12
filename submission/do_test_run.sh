#!/usr/bin/env bash
#
# Builds the algorithm's Docker image, boots it as an HTTP server that
# implements Grand Challenge's "invoke" API, then exercises it against each
# configured interface. For each interface this script:
#   1. stages the interface's input files into the container's /input mount
#   2. calls POST /invoke and checks for a success response (HTTP 201 Created)
#   3. copies whatever the container wrote to /output back to the host
#
# Run this after changing the algorithm to confirm the container still
# behaves correctly before uploading it (see ./do_save.sh).
#
# Before the first run you need two things this repository does not ship:
#   * the nnU-Net checkpoints, under ./model (or set ISLES_MODEL_DIR) --
#     see README.md for the layout
#   * one T1w volume (.nii.gz or .mha) in
#     test/input/interf0/images/t1-brain-mri/
#
# Environment overrides:
#   ISLES_MODEL_DIR   where the checkpoints live      (default: ./model)
#   ISLES_TEST_GPUS   value for docker --gpus         (default: all; e.g. device=2)
#   ISLES_TEST_NET    published | internal            (default: published)

# Exit immediately on: an error in any command, use of an unset variable,
# or a failure in any stage of a pipeline (not just the last stage).
set -euo pipefail

# @(mha|nii|nii.gz) in the input check below needs extended globbing.
shopt -s extglob

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="example_algorithm_preliminary-docker-evaluation-sanity-check"
CONTAINER_NAME="example_algorithm_preliminary-docker-evaluation-sanity-check_container"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

# The nnU-Net checkpoints, bind-mounted read-only at /opt/ml/model exactly as
# Grand Challenge mounts an uploaded model tarball. Not in this repository.
MODEL_DIR="${ISLES_MODEL_DIR:-${SCRIPT_DIR}/model}"

# Staging directories are bind-mounted into the container as /input and /output.
# They start empty and are (re)provisioned with hard-linked input files right
# before each invocation — mimicking how Grand Challenge provisions inputs.
STAGING_INPUT_DIR="${SCRIPT_DIR}/test/.staging_input"
STAGING_OUTPUT_DIR="${SCRIPT_DIR}/test/.staging_output"

# How the tester sidecar reaches the algorithm container.
#
#   internal   the upstream template's path: both containers sit on an
#              --internal Docker network with no route to the internet, and the
#              sidecar resolves the algorithm by container name. This is the
#              closest match to Grand Challenge, where the algorithm container
#              has no network access at all. Prefer it when it works for you.
#   published  (default) the container's port is published to the host and the
#              sidecar reaches it through host.docker.internal. This is what the
#              submission was actually tested with, because container-name
#              resolution on the internal network did not work on that machine.
#              NOTE: this network is NOT --internal, so this mode does not
#              reproduce Grand Challenge's no-internet restriction — it only
#              checks health, invoke and the written outputs.
TEST_NET="${ISLES_TEST_NET:-published}"

# How long to wait for the container's /health endpoint to come up.
# These match the timeouts used on Grand Challenge.
HEALTH_CHECK_MAX_ATTEMPTS=60
HEALTH_CHECK_DELAY_SECONDS=10
HEALTH_CHECK_TIMEOUT_SECONDS=10

# How long a single /invoke call is allowed to run.
# Note that this is NOT equivalent to the maximum runtime set for jobs on Grand Challenge
# The maximum runtime on Grand Challenge includes I/O as well as model and auxiliary data loading.
# Locally (when running this script) it only includes inference (I/O, model and auxiliary data loading happen before).
INVOKE_TIMEOUT_SECONDS=300

# --- Globals set by setup() -------------------------------------------------
LOG_LINES_SHOWN=0
DOCKER_VOLUME_TAG=""
DOCKER_NETWORK_TAG=""
TESTER_NAME=""
CONTAINER_PORT=4743
BASE_URL=""
GPU_ARGS=""
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    # Prepare staging dirs, Docker network, and detect GPU support
    setup
    trap cleanup EXIT

    build_container
    start_container

    # Poll /health until the server signals it's ready
    check_health


    # Copy this interface's input files into the container
    provision "interf0"
    # Call POST /invoke and wait for inference completion
    invoke
    # Copy the results back to the host
    collect_output "interf0"


    log "Save this image for uploading via ./do_save.sh"
}

# ---------------------------------------------------------------------------
# Helper Functions
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

setup() {
    log "Setup ..."

    if [ ! -d "$MODEL_DIR" ]; then
        log "No model directory at ${MODEL_DIR}" error
        log "The checkpoints are not in this repository (~7.6 GB). Set ISLES_MODEL_DIR" error
        log "or populate ./model with the layout described in README.md" error
        exit 1
    fi

    # Match the extensions find_input_image() in inference.py actually globs for,
    # so the placeholder README.md in that directory does not count as an image.
    if ! compgen -G "${INPUT_DIR}/interf0/images/t1-brain-mri/*.@(mha|nii|nii.gz)" > /dev/null; then
        log "No input image in ${INPUT_DIR}/interf0/images/t1-brain-mri/" error
        log "Place one T1w volume (.nii.gz, .nii or .mha) there and re-run" error
        exit 1
    fi

    # Allow the Docker user to read these on the host
    chmod -R -f o+rX "$INPUT_DIR" "$MODEL_DIR"

    # Disable promotional logs from Docker
    export DOCKER_CLI_HINTS=false

    # Detect whether the NVIDIA GPU runtime is available.
    # On macOS or machines without nvidia-container-runtime, --gpus all would
    # prevent the container from starting. In that case we skip the flag and
    # the algorithm will run on CPU (torch.cuda.is_available() returns False).
    # This does not affect the exported container — on Grand Challenge your
    # algorithm will always have GPU access.
    #
    # ISLES_TEST_GPUS selects which devices to expose, e.g. ISLES_TEST_GPUS=device=2
    # on a shared multi-GPU box.
    if docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
        GPU_ARGS="--gpus ${ISLES_TEST_GPUS:-all}"
        log "NVIDIA runtime detected — enabling GPU access (${GPU_ARGS})"
    else
        GPU_ARGS=""
        log "No NVIDIA runtime detected — running on CPU (this is fine for testing)" warning
        log "On CPU inference.py drops to ONE ensemble member and disables mirroring TTA," warning
        log "so local CPU outputs are NOT the shipped two-member configuration." warning
    fi

    # Create empty staging directories for the /input and /output bind mounts
    rm -rf "$STAGING_INPUT_DIR" "$STAGING_OUTPUT_DIR"
    mkdir -m o+rwX "$STAGING_INPUT_DIR"
    mkdir -m o+rwX "$STAGING_OUTPUT_DIR"

    # A scratch volume that mimics the ephemeral /tmp on Grand Challenge
    DOCKER_VOLUME_TAG="${DOCKER_IMAGE_TAG}-scratch"
    docker volume create "$DOCKER_VOLUME_TAG" > /dev/null

    # The network the containers share, and the URL the tester sidecar uses to
    # reach the algorithm. See the TEST_NET comment at the top of this script.
    DOCKER_NETWORK_TAG="${DOCKER_IMAGE_TAG}-isolated"
    if [[ "$TEST_NET" == "internal" ]]; then
        # An isolated network that mimics the network restrictions on Grand
        # Challenge. The algorithm container has no internet access, just like
        # in production, and is reached by container name via Docker's
        # embedded DNS.
        BASE_URL="http://${CONTAINER_NAME}:${CONTAINER_PORT}"
        docker network create --internal "$DOCKER_NETWORK_TAG" > /dev/null
        log "Network mode: internal (no internet, resolved by container name)"
    else
        BASE_URL="http://host.docker.internal:${CONTAINER_PORT}"
        docker network create "$DOCKER_NETWORK_TAG" > /dev/null
        log "Network mode: published (port ${CONTAINER_PORT} exposed on the host)" warning
        log "This does NOT reproduce Grand Challenge's no-internet restriction;" warning
        log "set ISLES_TEST_NET=internal for that." warning
    fi

    # The tester sidecar: lives on the shared network so it can reach the
    # algorithm container, and is how we issue health/invoke checks without
    # needing the host to route into that network.
    TESTER_NAME="${DOCKER_IMAGE_TAG}-tester"
    docker run --detach --name "$TESTER_NAME" \
        --network "$DOCKER_NETWORK_TAG" \
        --add-host=host.docker.internal:host-gateway \
        curlimages/curl:latest sleep infinity > /dev/null 2>&1
}

cleanup() {
    log "Cleanup ..."

    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm -f "$TESTER_NAME" >/dev/null 2>&1 || true
    log "Containers stopped"

    # Remove staging directories
    rm -rf "$STAGING_INPUT_DIR"
    # Staging output may contain files owned by the container's UID on Linux
    docker run --rm --platform=linux/amd64 --quiet --user root \
        --volume "$STAGING_OUTPUT_DIR":/output \
        --entrypoint /bin/sh \
        $DOCKER_IMAGE_TAG \
        -c "rm -rf /output/* || true" 2>/dev/null || true
    rm -rf "$STAGING_OUTPUT_DIR"

    # Remove volumes and network
    docker volume rm "$DOCKER_VOLUME_TAG" > /dev/null 2>&1 || true
    docker network rm "$DOCKER_NETWORK_TAG" > /dev/null 2>&1 || true
}

build_container() {
    log "(Re)build the container"
    source "${SCRIPT_DIR}/do_build.sh"

    log "Verifying container labels"
    local api_method
    api_method=$(docker inspect \
        --format='{{index .Config.Labels "org.grand-challenge.api-method"}}' \
        "$DOCKER_IMAGE_TAG" 2>/dev/null || echo "")

    if [ "$api_method" != "invoke" ]; then
        log "ERROR: The container image is missing the required label:" error
        log "  LABEL org.grand-challenge.api-method=\"invoke\"" error
        log "" error
        log "Without this label, Grand Challenge will not recognize that your" error
        log "container implements the invoke API and will default to exec mode." error
        log "Please add this label to your Dockerfile." error
        exit 1
    fi
}

start_container() {
    log "Starting container"

    # Extra arguments worth calling out:
    #   --network <shared>                 the tester sidecar issues HTTP requests to
    #                                      the container over this network
    #   --volume <vol>:/tmp                scratch space (Grand Challenge disallows writes
    #                                      elsewhere outside the mounted directories)
    #   --volume model:/opt/ml/model:ro    the (optional) model tarball
    local docker_run_args=(
        --detach
        --name "$CONTAINER_NAME"
        ${GPU_ARGS:+$GPU_ARGS}
        --platform=linux/amd64
        --volume "$MODEL_DIR":/opt/ml/model:ro
        --volume "$STAGING_INPUT_DIR":/input:ro
        --volume "$STAGING_OUTPUT_DIR":/output
        --volume "$DOCKER_VOLUME_TAG":/tmp
        --network "$DOCKER_NETWORK_TAG"
    )

    # Publishing the port is only needed when the sidecar reaches the algorithm
    # through the host rather than by container name.
    if [[ "$TEST_NET" != "internal" ]]; then
        docker_run_args+=(-p "${CONTAINER_PORT}:${CONTAINER_PORT}")
    fi

    docker run "${docker_run_args[@]}" "$DOCKER_IMAGE_TAG" >/dev/null

    log "Container started; reachable from tester sidecar at ${BASE_URL}"
}

flush_docker_log() {
    # Prints any container log lines that haven't been shown yet.
    local total_lines new_lines
    total_lines=$(docker logs "$CONTAINER_NAME" 2>&1 | wc -l)
    new_lines=$((total_lines - LOG_LINES_SHOWN))

    if (( new_lines > 0 )); then
        docker logs --timestamps --tail "$new_lines" "$CONTAINER_NAME"
    fi

    LOG_LINES_SHOWN=$total_lines
}

http_status() {
    # Issues a request from inside the tester sidecar (not the host).
    local method="$1"
    local timeout_seconds="$2"
    local url="$3"

    docker exec "$TESTER_NAME" \
        curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout_seconds" \
        -X "$method" "$url" \
      || echo "000"
}

check_health() {
    log "Waiting for health endpoint..."

    local status
    for ((i = 1; i <= HEALTH_CHECK_MAX_ATTEMPTS; i++)); do
        status=$(http_status "GET" "$HEALTH_CHECK_TIMEOUT_SECONDS" "${BASE_URL}/health")
        log "Health check attempt $i/${HEALTH_CHECK_MAX_ATTEMPTS} returned $status"

        if [[ "$status" == "200" ]]; then
            log "API healthy"
            flush_docker_log
            return 0
        fi

        if [[ "$status" == "302" ]]; then
            log "Health endpoint returned HTTP 302 — failing" error
            flush_docker_log
            return 1
        fi

        log "Retrying in ${HEALTH_CHECK_DELAY_SECONDS}s"
        sleep "$HEALTH_CHECK_DELAY_SECONDS"
    done

    log "Health endpoint never returned HTTP 200" error
    flush_docker_log
    return 1
}

provision() {
    local interface_dir="$1"
    log "Provisioning input for ${interface_dir}"

    # Clear /output inside the container (host can't due to UID mismatch on Linux)
    docker exec --user root "$CONTAINER_NAME" /bin/sh -c "rm -rf /output/*"

    # Clear the input staging dir, then hard-link this interface's input files in
    rm -rf "${STAGING_INPUT_DIR:?}"/*
    cp -rl "${INPUT_DIR}/${interface_dir}/." "$STAGING_INPUT_DIR/"
}

invoke() {
    log "Calling invoke endpoint..."

    local status
    status=$(http_status "POST" "$INVOKE_TIMEOUT_SECONDS" "${BASE_URL}/invoke")
    flush_docker_log

    if [ "$status" != "201" ]; then
        log "Invoke failed (expected HTTP 201 Created, got $status)" error
        exit 1
    fi

    log "Invoke completed"
}

collect_output() {
    local interface_dir="$1"
    local destination="${OUTPUT_DIR}/${interface_dir}"
    log "Collecting output for ${interface_dir}"

    if [ -d "$destination" ]; then
        # Clean up any earlier collected output using a container (handles
        # files owned by the container's UID on Linux)
        docker run --rm --platform=linux/amd64 --quiet --user root \
            --volume "$destination":/output \
            --entrypoint /bin/sh \
            $DOCKER_IMAGE_TAG \
            -c "rm -rf /output/* || true"
    else
        mkdir -p -m o+rwX "$destination"
    fi

    # Fix permissions so the host user can read the output files.
    # The container may have written them as a different UID on Linux.
    docker exec --user root "$CONTAINER_NAME" \
        /bin/sh -c "chmod -R -f o+rX /output/*"

    # Copy from the staging directory to the host output directory
    cp -r "$STAGING_OUTPUT_DIR/." "${destination}/"
    log "Wrote results to ${OUTPUT_DIR}/"
}

# ---------------------------------------------------------------------------

main
