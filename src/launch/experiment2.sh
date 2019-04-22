#!/bin/zsh

for value in {1..100} 
do
    python3 network_creator.py "network$value.txt"
done