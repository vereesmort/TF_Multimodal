#!/bin/bash

wget http://snap.stanford.edu/decagon/bio-decagon-ppi.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-targets.tar.gz

# wget http://snap.stanford.edu/decagon/bio-decagon-targets-all.tar.gz
#Note that including this file in analysis introduces ~1400 drug nodes and ~112k drug-target edges MORE than are reported in the paper. 

wget http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-mono.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-effectcategories.tar.gz

for f in *.gz;
do
    tar -xzvf $f
    rm -f $f
done