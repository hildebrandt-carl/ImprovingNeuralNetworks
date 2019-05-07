#!/bin/bash

for value in {76..100}
do
    python3 network_creator.py "network$value.txt"
done
