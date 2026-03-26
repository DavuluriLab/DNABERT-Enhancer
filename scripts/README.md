This folder contains additional programs to process data

## Negative sequences

This Python script **bed_to_fasta_with_negatives.py** is designed to prepare datasets for enhancer finetune dataset by generating negative instances using your positive/peak coordinates, combining them with positive instances, and extracting sequences from a reference genome. The final output includes a combined BED file, a CSV with sequences, and a FASTA file suitable for downstream machine learning or bioinformatics analyses.

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

The resultant fasta file can be used as an input to the data redundancy tools such as mmseq, CD-hit or Linclust, if you prefer removing the sequence redundancy from your data (recommended) or you can directly use as an input for the next script.
