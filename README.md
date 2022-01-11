# py-multi-seq
### Python implementation of MULTIseq barcode alignment using fuzzy string matching and GMM barcode assignment.


The scripts in this repository are roughly analogous to the provided MULTI-seq R package deMULTIplex at https://github.com/chris-mcginnis-ucsf/MULTI-seq. This script loads read data from paired end reads, performs fuzzy string matching from paired end reads to the provided MULTIseq barcode file, then counts the reads mapping to each barcode. Next, Expectation Maximization is used to fit Gaussian Mixture Models for each barcode, which assigns each cell a most likely barcode, no barcode or doublet barcodes.


## Installation


Clone this repository. The scripts within also depend on `python >= 3.7` and the following packages which can be installed with:
`pip install pandas numpy scipy fuzzywuzzy tqdm sparse_dot_topn scanpy natsort`


You will need the cellranger cell barcodes file before running. You can in theory modify the MultiseqIndices.txt along with the read length parameters for custom barcodes in the reads.


## Usage example for 10X scRNAseq or Multiome + MULTIseq: 


`python BarcodeFuzzyMatching.py /path/to/this/repo/MultiseqSamplesExample.txt /path/to/this/repo/MultiseqIndices.txt /path/to/sampleMULTIseq_R1.fastq  /path/to/cellranger/outs/filtered_feature_bc_matrix/barcodes.tsv.gz /path/to/output/dir/ 16 8 0`


`python RunDemuxEM.py /path/to/output/dir/ /path/to/cellranger/outs/filtered_feature_bc_matrix/`


Running this pipeline will output a matrix of barcodes by reads_counts, as well as a csv listing cell barcodes and their assigned barcode(s).
