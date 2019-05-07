#!/bin/bash

for value in {46..60}
do
    python3 network_creator.py "network$value.txt"
done
