#!/bin/bash

for value in {1..15}
do
    python3 network_creator.py "network$value.txt"
done
