#!/bin/bash

for value in {61..75}
do
    python3 network_creator.py "network$value.txt"
done
