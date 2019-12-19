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
        echo 'brew install ffmpeg'
        brew install ffmpeg
      elif [ $mac_package_manager = "macports" ]; then
        echo 'port selfupdate'
        port selfupdate
        echo 'port install ffmpeg'
        port install ffmpeg
      fi
      ;;
    Linux)
      echo 'apt install libtcmalloc-minimal4 ffmpeg'
      sudo apt update
      sudo apt install libtcmalloc-minimal4 ffmpeg -y
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
    python3 -m pip install tensorflow-gpu==1.15.0
  else
    echo '\nInstall tensorflow'
    python3 -m pip install tensorflow==1.15.0
  fi
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
      echo "Unknown flag $1, please check available flags with --help"
      exit 1
      ;;
  esac
  shift
done

check_requirements
install_system_packages
install_tensorflow
install_python_packages

echo '\nSetup completed.'
