#!/bin/bash

for value in {31..45}
do
    python3 network_creator.py "network$value.txt"
done
