#!/bin/sh

cd "$(dirname "$0")"

echo 'Install required system packages'
case "$(uname -s)" in
   Darwin)
     echo 'brew install qt open-mpi pkg-config ffmpeg'
     brew install qt open-mpi pkg-config ffmpeg;;

   Linux)
     echo 'apt install mpich build-essential qt5-default pkg-config libtcmalloc-minimal4 ffmpeg'
     sudo apt install mpich build-essential qt5-default pkg-config libtcmalloc-minimal4 ffmpeg;;

   *)
     echo 'Only Linux and macOS are currently supported.'
     exit 1
esac

echo '\nInstall TensorFlow (Please change to tensorflow-gpu for GPU support)';
(
  python3 -m pip install tensorflow
)

echo '\nInstall OpenAI CoinRun';
(
  git clone https://github.com/openai/coinrun.git
  cd coinrun || exit
  python3 -m pip install -r requirements.txt
  python3 -m pip install -e .
  cd ..
)

echo '\nInstall Python packages'
(
  python3 -m pip install -r requirements.txt
)

echo '\nSetup completed.'
