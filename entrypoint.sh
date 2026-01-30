#!/bin/bash
set -e

# Create privilege separation directory
mkdir -p /run/sshd

# Generate host keys if missing
ssh-keygen -A

export SHELL=/bin/bash

/usr/sbin/sshd -D &
echo "SSH daemon Started"

jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/private --ServerApp.token='' --ServerApp.password='' --ServerApp.terminals_enabled=True &
echo "JupyterLab Started"
tail -f /dev/null 