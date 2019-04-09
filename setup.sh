#!/bin/zsh
cd ~
source .venv/bin/activate
cd -
cd src/CNN-Cert
export PYTHONPATH=`pwd`:$PYTHONPATH
cd -
