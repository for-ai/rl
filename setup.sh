#!/bin/sh

use_gpu=false
mac_package_manager="homebrew"

print_usage() {
  echo 'Usage: sh setup.sh --use_gpu --package_manager'
  echo '---gpu,-g                     install tensorflow-gpu'
  echo '--package-manager,-p          homebrew or macports'
}

check_requirements() {
  case "$(uname -s)" in
    Darwin)
      echo 'Installing on macOS.'
      if [ "$mac_package_manager" = "homebrew" ]; then
        if [ "$(command -v brew)" = "" ]; then
          echo 'Homebrew is not installed'
          exit 1
        fi
      elif [ "$mac_package_manager" = "macports" ]; then
        if [ "$(command -v port)" = "" ]; then
          echo 'MacPorts is not installed'
          exit 1
        fi
      else
        echo 'Please specify --package-manager to be homebrew or macports'
        exit 1
      fi
      ;;
    Linux)
      echo 'Install on Linux.'
      ;;
    *)
      echo 'Only Linux and macOS systems are currently supported.'
      exit 1
      ;;
  esac
}

install_system_packages() {
  echo 'Install required system packages'
  case "$(uname -s)" in
    Darwin)
      if [ $mac_package_manager = "homebrew" ]; then
        echo 'brew install qt open-mpi pkg-config ffmpeg'
        brew install qt open-mpi pkg-config ffmpeg
      elif [ $mac_package_manager = "macports" ]; then
        echo 'port install qt open-mpi pkg-config ffmpeg'
        port install qt openmpi pkgconfig ffmpeg
      fi
      ;;
    Linux)
      echo 'apt install mpich build-essential qt5-default pkg-config libtcmalloc-minimal4 ffmpeg'
      sudo apt install mpich build-essential qt5-default pkg-config libtcmalloc-minimal4 ffmpeg
      ;;
    *)
      echo 'Only Linux and macOS systems are currently supported.'
      exit 1
      ;;
  esac
}

install_tensorflow() {
  if [ "$use_gpu" = "true" ]; then
    echo '\nInstall tensorflow-gpu'
    python3 -m pip install tensorflow-gpu
  else
    echo '\nInstall tensorflow'
    python3 -m pip install tensorflow
  fi
}

install_coinrun() {
  echo '\nInstall OpenAI CoinRun';
  cd "$(dirname "$0")"
  git clone https://github.com/openai/coinrun.git coinrun
  cd coinrun
  python3 -m pip install -r requirements.txt
  python3 -m pip install -e .
}

install_python_packages() {
  echo '\nInstall Python packages'
  cd "$(dirname "$0")"
  python3 -m pip install -r requirements.txt
}

# Read flags and arguments
while [ ! $# -eq 0 ]; do
  case "$1" in
    --help | -h)
      print_usage
      exit 1
      ;;
    --gpu | -g)
      use_gpu=true
      ;;
    --package-manager | -p)
      shift
      mac_package_manager=$1
      ;;
    *)
      echo "Unkown flag $1, please check available flags with --help"
      exit 1
      ;;
  esac
  shift
done

check_requirements
install_system_packages
install_tensorflow
install_coinrun
install_python_packages

echo '\nSetup completed.'
