#!/bin/bash

NetworkNames=""

for value in {1..100}
do
    # If this is the first network name just use the name
    if [ "$NetworkNames" = "" ]; then
        NetworkNames="network$value"
    # Otherwise append the name after a comma
    else
        NetworkNames="$NetworkNames, network$value"
    fi;

done

python3 adversarial-combine.py "$NetworkNames"
