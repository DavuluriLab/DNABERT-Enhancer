This folder contains additional programs to process data

## Generates negative instances and prepares sequences for fine-tuning

The Python script **bed_to_fasta_with_negatives.py** is designed to prepare datasets for enhancer fine-tuning. It generates negative instances from your positive/peak coordinates, combines them with positive instances, and extracts sequences from a reference genome. The output includes:

A combined **BED** file (positive + negative regions)
A **CSV** file with sequences
A **FASTA** file suitable for downstream machine learning or bioinformatics analyses

This FASTA file can serve as the input for:
Redundancy removal tools such as MMseqs2, CD-HIT, or Linclust (recommended if you want to remove duplicate or highly similar sequences)
Direct input to your next script for fine-tuning models

### Requirements
Python packages required:
```
pandas
pybedtools
```

### Usage
```
python Create_Negative_set.py \
    --pred <BED_FILE> \
    --ref_genome_filepath <REFERENCE_GENOME_FOLDER> \
    --genome_size_file <GENOME_SIZE_FILE> \
    --blacklist <BLACKLISTED_REGION_FILE> \
    --out <OUTPUT_DIRECTORY_PATH> \
    --tmpdir <TEMPORARY_DIRECTORY_PATH> \
    --length_limit <MIN_PEAK_LENGTH>
```
