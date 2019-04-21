#!/bin/zsh

for value in {1..9} 
do
    python3 network_creator.py "network{$value}.txt"
done