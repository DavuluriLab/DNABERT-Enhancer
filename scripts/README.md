This folder contains additional programs to process data

## Negative sequences

## For creating input file
The script Create_input_data.py generates the input file required for prediction with DNABERT-Enhancer. It extracts genomic sequences
from a reference genome (organized per chromosome) using coordinates from a **BED** file and converts them into overlapping k-mers. The output file contains two columns
namely Sequence	(Overlapping k-mer representation of the DNA sequence) and Label	(Random label : 0 or 1, used as a placeholder for prediction
input), saved  as **dev.tsv**.

### Requirements
Python packages required:
```
pandas
numpy
pybedtools
```

### Usage
```
python Create_input_data.py \
    --input_bed <BED_FILE> \
    --output_dir <OUTPUT_DIRECTORY> \
    --Reference_Genome_FilePath <REFERENCE_GENOME_FOLDER> \
    --kmer_length <KMER_SIZE> \
    --length_limit <MIN_PEAK_LENGTH>
```
