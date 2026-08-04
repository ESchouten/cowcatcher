#!/usr/bin/env bash
set -Eeuo pipefail

REPOSITORY="ESchouten/ai-detector"
RAW_ROOT="https://raw.githubusercontent.com/${REPOSITORY}"
API_ROOT="https://api.github.com/repos/${REPOSITORY}"

ASSUME_YES=0
DRY_RUN=0
START_AFTER_INSTALL=1
INSTALL_DIR=""
PLATFORM="${AI_DETECTOR_INSTALL_PLATFORM:-}"
ARCHITECTURE="${AI_DETECTOR_INSTALL_ARCH:-}"
GPU_MODE="${AI_DETECTOR_INSTALL_GPU:-auto}"

usage() {
  cat <<'EOF'
Install AI Detector on Linux or macOS.

Usage:
  curl -fsSL https://raw.githubusercontent.com/ESchouten/ai-detector/main/install.sh | bash -s -- --yes
  ./install.sh [options]

Options:
  --yes                 Apply the displayed plan without prompting.
  --dry-run             Detect the platform and print the plan only.
  --no-start            Install, but do not start AI Detector.
  --install-dir PATH    Override the installation directory.
  --gpu auto|nvidia|none
                        Override Linux GPU detection.
  --help                Show this help.
EOF
}

log() {
  printf '==> %s\n' "$*"
}

fail() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

confirm() {
  if [[ "$ASSUME_YES" -eq 1 ]]; then
    return
  fi
  if [[ -r /dev/tty ]]; then
    read -r -p "Continue? [y/N] " reply </dev/tty
  elif [[ -t 0 ]]; then
    read -r -p "Continue? [y/N] " reply
  else
    fail "Non-interactive installation requires --yes"
  fi
  [[ "$reply" =~ ^[Yy]$ ]] || exit 0
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    require_command sudo
    sudo "$@"
  fi
}

download() {
  local url="$1"
  local target="$2"
  curl --proto '=https' --tlsv1.2 --fail --location --retry 3 \
    --output "$target" "$url"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

verify_checksum() {
  local archive="$1"
  local expected="${2#sha256:}"
  local actual
  expected="$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || fail "Release checksum is invalid"
  actual="$(sha256_file "$archive" | tr '[:upper:]' '[:lower:]')"
  [[ "$actual" == "$expected" ]] || fail "Release checksum verification failed"
}

resolve_macos_release_asset() {
  local releases_file="$1"
  local tag_prefix="$2"
  local archive_pattern="$3"

  /usr/bin/osascript -l JavaScript -e '
ObjC.import("Foundation");

function run(argv) {
  const releasesPath = argv[0];
  const tagPrefix = argv[1];
  const archivePattern = new RegExp(argv[2]);
  const contents = $.NSString
    .stringWithContentsOfFileEncodingError(
      releasesPath,
      $.NSUTF8StringEncoding,
      null
    )
    .js;
  const releases = JSON.parse(contents);

  for (const release of releases) {
    if (
      release.draft ||
      release.prerelease ||
      !release.tag_name.startsWith(tagPrefix)
    ) {
      continue;
    }
    const archive = release.assets.find((asset) =>
      archivePattern.test(asset.name)
    );
    if (!archive) {
      continue;
    }
    const digest = archive.digest || "";
    if (!/^sha256:[0-9a-fA-F]{64}$/.test(digest)) {
      continue;
    }
    return [
      archive.browser_download_url,
      digest,
      release.tag_name,
      archive.name
    ].join("|");
  }

  throw new Error(
    "No verified release asset matched " + tagPrefix + " / " + argv[2]
  );
}
' "$releases_file" "$tag_prefix" "$archive_pattern"
}

detect_platform() {
  if [[ -z "$PLATFORM" ]]; then
    PLATFORM="$(uname -s)"
  fi
  if [[ -z "$ARCHITECTURE" ]]; then
    ARCHITECTURE="$(uname -m)"
  fi
  PLATFORM="$(printf '%s' "$PLATFORM" | tr '[:upper:]' '[:lower:]')"
  ARCHITECTURE="$(printf '%s' "$ARCHITECTURE" | tr '[:upper:]' '[:lower:]')"
  case "$PLATFORM" in
    linux) PLATFORM="linux" ;;
    darwin | macos) PLATFORM="darwin" ;;
    *) fail "Unsupported platform: $PLATFORM" ;;
  esac
  case "$ARCHITECTURE" in
    x86_64 | amd64) ARCHITECTURE="amd64" ;;
    arm64 | aarch64) ARCHITECTURE="arm64" ;;
    *) fail "Unsupported architecture: $ARCHITECTURE" ;;
  esac
}

detect_nvidia() {
  case "$GPU_MODE" in
    nvidia) return 0 ;;
    none) return 1 ;;
    auto) ;;
    *) fail "--gpu must be auto, nvidia, or none" ;;
  esac
  command -v nvidia-smi >/dev/null 2>&1 && return 0
  [[ -r /proc/driver/nvidia/version ]] && return 0
  if command -v lspci >/dev/null 2>&1 && lspci | grep -qi nvidia; then
    return 0
  fi
  return 1
}

install_docker_debian() {
  local distribution="$1"
  local codename="$2"
  local temporary_directory="$3"
  local architecture
  architecture="$(dpkg --print-architecture)"

  as_root apt-get update
  as_root apt-get install -y ca-certificates curl
  as_root install -m 0755 -d /etc/apt/keyrings
  download "https://download.docker.com/linux/${distribution}/gpg" \
    "${temporary_directory}/docker.asc"
  as_root install -m 0644 "${temporary_directory}/docker.asc" /etc/apt/keyrings/docker.asc
  printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/%s %s stable\n' \
    "$architecture" "$distribution" "$codename" >"${temporary_directory}/docker.list"
  as_root install -m 0644 "${temporary_directory}/docker.list" \
    /etc/apt/sources.list.d/docker.list
  as_root apt-get update
  as_root apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
}

install_docker_rpm() {
  local distribution="$1"
  as_root dnf -y install dnf-plugins-core
  as_root dnf config-manager --add-repo \
    "https://download.docker.com/linux/${distribution}/docker-ce.repo"
  as_root dnf -y install docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
}

install_docker() {
  local temporary_directory="$1"
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    log "Docker with Compose is already installed"
    return
  fi

  [[ -r /etc/os-release ]] || fail "Cannot identify this Linux distribution"
  # shellcheck disable=SC1091
  source /etc/os-release
  local distribution
  distribution="$(printf '%s' "$ID" | tr '[:upper:]' '[:lower:]')"
  local like="${ID_LIKE:-}"
  case "$distribution" in
    ubuntu | debian)
      install_docker_debian "$distribution" "${VERSION_CODENAME:?Missing VERSION_CODENAME}" \
        "$temporary_directory"
      ;;
    fedora)
      install_docker_rpm fedora
      ;;
    rhel | centos | rocky | almalinux)
      install_docker_rpm centos
      ;;
    *)
      if [[ "$like" == *ubuntu* ]]; then
        install_docker_debian ubuntu "${UBUNTU_CODENAME:-${VERSION_CODENAME:?}}" \
          "$temporary_directory"
      elif [[ "$like" == *debian* ]]; then
        install_docker_debian debian "${DEBIAN_CODENAME:-${VERSION_CODENAME:?}}" \
          "$temporary_directory"
      else
        fail "Automatic Docker installation is not supported for ${PRETTY_NAME:-$distribution}"
      fi
      ;;
  esac
  as_root systemctl enable --now docker
}

install_nvidia_toolkit() {
  local temporary_directory="$1"
  if command -v nvidia-ctk >/dev/null 2>&1; then
    log "NVIDIA Container Toolkit is already installed"
  else
    [[ -r /etc/os-release ]] || fail "Cannot identify this Linux distribution"
    # shellcheck disable=SC1091
    source /etc/os-release
    local distribution
    distribution="$(printf '%s' "$ID" | tr '[:upper:]' '[:lower:]')"
    case "$distribution" in
      ubuntu | debian)
        as_root apt-get update
        as_root apt-get install -y curl gpg
        download https://nvidia.github.io/libnvidia-container/gpgkey \
          "${temporary_directory}/nvidia-container-toolkit.key"
        gpg --dearmor --yes --output "${temporary_directory}/nvidia-container-toolkit.gpg" \
          "${temporary_directory}/nvidia-container-toolkit.key"
        as_root install -m 0644 "${temporary_directory}/nvidia-container-toolkit.gpg" \
          /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        download \
          https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
          "${temporary_directory}/nvidia-container-toolkit.list"
        sed \
          's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
          "${temporary_directory}/nvidia-container-toolkit.list" \
          >"${temporary_directory}/nvidia-container-toolkit.signed.list"
        as_root install -m 0644 "${temporary_directory}/nvidia-container-toolkit.signed.list" \
          /etc/apt/sources.list.d/nvidia-container-toolkit.list
        as_root apt-get update
        as_root apt-get install -y nvidia-container-toolkit
        ;;
      fedora | rhel | centos | rocky | almalinux)
        download \
          https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
          "${temporary_directory}/nvidia-container-toolkit.repo"
        as_root install -m 0644 "${temporary_directory}/nvidia-container-toolkit.repo" \
          /etc/yum.repos.d/nvidia-container-toolkit.repo
        as_root dnf -y install nvidia-container-toolkit
        ;;
      *)
        fail "Automatic NVIDIA Container Toolkit installation is not supported for ${PRETTY_NAME:-$ID}"
        ;;
    esac
  fi
  as_root nvidia-ctk runtime configure --runtime=docker
  as_root systemctl restart docker
}

linux_install() {
  local has_nvidia=0
  if detect_nvidia; then
    has_nvidia=1
  fi
  if [[ "$ARCHITECTURE" == "arm64" && "$has_nvidia" -eq 0 ]]; then
    fail "Linux arm64 currently requires an NVIDIA Jetson/JetPack host"
  fi

  INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/share/ai-detector}"
  log "Platform: Linux ${ARCHITECTURE}"
  log "Acceleration: $([[ "$has_nvidia" -eq 1 ]] && printf NVIDIA || printf CPU)"
  log "Install directory: $INSTALL_DIR"
  log "Plan: install Docker, pull the latest detector + app, and expose the UI on http://localhost"
  if [[ "$has_nvidia" -eq 1 ]]; then
    log "Plan: install/configure NVIDIA Container Toolkit and enable GPU access"
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return
  fi
  confirm

  require_command curl
  local temporary_directory
  temporary_directory="$(mktemp -d)"
  trap 'rm -rf "$temporary_directory"' EXIT
  install_docker "$temporary_directory"
  if [[ "$has_nvidia" -eq 1 ]]; then
    if [[ "$ARCHITECTURE" == "amd64" ]]; then
      command -v nvidia-smi >/dev/null 2>&1 ||
        fail "An NVIDIA GPU was detected, but the host driver/nvidia-smi is unavailable"
    fi
    install_nvidia_toolkit "$temporary_directory"
  fi

  mkdir -p "$INSTALL_DIR/data"
  download "${RAW_ROOT}/main/deploy/compose.yml" "$INSTALL_DIR/compose.yml"
  download "${RAW_ROOT}/main/deploy/compose.nvidia.yml" \
    "$INSTALL_DIR/compose.nvidia.yml"
  {
    printf 'AI_DETECTOR_UID=%s\n' "$(id -u)"
    printf 'AI_DETECTOR_GID=%s\n' "$(id -g)"
    if [[ "$has_nvidia" -eq 1 ]]; then
      if [[ "$ARCHITECTURE" == "arm64" ]]; then
        printf 'AI_DETECTOR_IMAGE=ghcr.io/eschouten/ai-detector:latest-jetpack6\n'
      else
        printf 'AI_DETECTOR_IMAGE=ghcr.io/eschouten/ai-detector:latest\n'
      fi
    else
      printf 'AI_DETECTOR_IMAGE=ghcr.io/eschouten/ai-detector:latest-cpu\n'
    fi
  } >"$INSTALL_DIR/.env"

  local compose_arguments=(-f "$INSTALL_DIR/compose.yml")
  if [[ "$has_nvidia" -eq 1 && "$ARCHITECTURE" == "amd64" ]]; then
    compose_arguments+=(-f "$INSTALL_DIR/compose.nvidia.yml")
    as_root docker run --rm --gpus all \
      nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi >/dev/null
  elif [[ "$has_nvidia" -eq 1 ]]; then
    compose_arguments+=(-f "$INSTALL_DIR/compose.nvidia.yml")
  fi
  as_root docker compose --project-directory "$INSTALL_DIR" "${compose_arguments[@]}" pull
  if [[ "$START_AFTER_INSTALL" -eq 1 ]]; then
    as_root docker compose --project-directory "$INSTALL_DIR" "${compose_arguments[@]}" up -d
    log "AI Detector is starting at http://localhost"
  else
    log "Installed. Start with: docker compose --project-directory '$INSTALL_DIR' ${compose_arguments[*]} up -d"
  fi
  rm -rf "$temporary_directory"
  trap - EXIT
}

macos_install() {
  [[ "$ARCHITECTURE" == "arm64" ]] ||
    fail "The native macOS application currently requires Apple silicon"
  INSTALL_DIR="${INSTALL_DIR:-$HOME/Applications/AI Detector}"
  log "Platform: macOS Apple silicon"
  log "Acceleration: native MPS"
  log "Install directory: $INSTALL_DIR"
  log "Plan: download the latest detector + app releases, verify their checksums, and preserve local data"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return
  fi
  confirm

  require_command curl
  require_command unzip
  require_command osascript
  local temporary_directory
  temporary_directory="$(mktemp -d)"
  trap 'rm -rf "$temporary_directory"' EXIT

  local releases_file="$temporary_directory/releases.json"
  download "${API_ROOT}/releases?per_page=100" "$releases_file"

  local detector_record
  local web_record
  detector_record="$(resolve_macos_release_asset \
    "$releases_file" "detector/v" '^aidetector-osx-.*[.]zip$')" ||
    fail "No checksum-verified macOS detector release was found"
  web_record="$(resolve_macos_release_asset \
    "$releases_file" "web/v" '^aidetector-web-osx-.*[.]zip$')" ||
    fail "No checksum-verified macOS web release was found"

  local detector_url detector_digest detector_tag detector_name
  local web_url web_digest web_tag web_name
  IFS='|' read -r detector_url detector_digest detector_tag detector_name \
    <<<"$detector_record"
  IFS='|' read -r web_url web_digest web_tag web_name <<<"$web_record"
  log "Detector release: $detector_tag ($detector_name)"
  log "App release: $web_tag ($web_name)"

  local detector_archive="$temporary_directory/detector.zip"
  local web_archive="$temporary_directory/web.zip"
  download "$detector_url" "$detector_archive"
  download "$web_url" "$web_archive"
  verify_checksum "$detector_archive" "$detector_digest"
  verify_checksum "$web_archive" "$web_digest"

  mkdir -p "$temporary_directory/detector" "$temporary_directory/web"
  unzip -q "$detector_archive" -d "$temporary_directory/detector"
  unzip -q "$web_archive" -d "$temporary_directory/web"
  local detector_executable
  local web_executable
  detector_executable="$(find "$temporary_directory/detector" -type f \
    -name 'aidetector-osx-*.command' -print -quit)"
  web_executable="$(find "$temporary_directory/web" -type f \
    -name 'aidetector-web-osx-*.command' -print -quit)"
  [[ -n "$detector_executable" ]] || fail "Detector release contains no macOS executable"
  [[ -n "$web_executable" ]] || fail "Web release contains no macOS executable"

  mkdir -p "$INSTALL_DIR"
  cp "$detector_executable" "$INSTALL_DIR/ai-detector.command"
  cp "$web_executable" "$INSTALL_DIR/ai-detector-web.command"
  download "${RAW_ROOT}/main/packaging/native/AI%20Detector.command" \
    "$INSTALL_DIR/AI Detector.command"
  download "${RAW_ROOT}/main/packaging/native/README.txt" \
    "$INSTALL_DIR/README.txt"
  chmod +x "$INSTALL_DIR/AI Detector.command" \
    "$INSTALL_DIR/ai-detector.command" \
    "$INSTALL_DIR/ai-detector-web.command"
  if [[ "$START_AFTER_INSTALL" -eq 1 ]]; then
    open "$INSTALL_DIR/AI Detector.command"
    log "AI Detector is starting and will open in your browser"
  else
    log "Installed. Start '$INSTALL_DIR/AI Detector.command'"
  fi
  rm -rf "$temporary_directory"
  trap - EXIT
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --yes) ASSUME_YES=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --no-start) START_AFTER_INSTALL=0 ;;
    --install-dir)
      shift
      [[ "$#" -gt 0 ]] || fail "--install-dir requires a path"
      INSTALL_DIR="$1"
      ;;
    --gpu)
      shift
      [[ "$#" -gt 0 ]] || fail "--gpu requires a value"
      GPU_MODE="$1"
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    *) fail "Unknown option: $1" ;;
  esac
  shift
done

detect_platform
case "$PLATFORM" in
  linux) linux_install ;;
  darwin) macos_install ;;
esac
