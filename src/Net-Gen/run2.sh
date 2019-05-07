#!/bin/bash

for value in {16..30}
do
    python3 network_creator.py "network$value.txt"
done
