#!/bin/bash

echo 'Running analysis of time vs test length. Chunk size for nlogm = 936.' > ../results/genes_data/performance_by_text_length
echo 'Run with: python analysis.py CAG ../Genes/Genes\ by\ Size/pow_[GENE NUM]/*' >> ../results/genes_data/performance_by_text_length

for i in `seq 6 16`;
    do
        echo 'Getting analysis of time VS text length on genes of size' $i
        python analysis.py CAG ../Genes/Genes\ by\ Size/pow_$i/* >> ../results/genes_data/performance_by_text_length
    done

    python graph.py ../results/genes_data/performance_by_text_length

echo 'Running analysis of time vs chunk size.' > ../results/genes_data/performance_by_chunk_size
echo 'Run with: python analysis.py -c 1000 CAG ../Genes/Genes\ by\ Size/pow_[GENE NUM]/*' >> ../results/genes_data/performance_by_chunk_size

echo 'Getting analysis of time VS chunk_size on genes of size 2^10'
python analysis.py -c 1000 CAG ../Genes/Genes\ by\ Size/pow_10/* >> ../results/genes_data/performance_by_chunk_size

python graph.py -c ../results/genes_data/performance_by_chunk_size
