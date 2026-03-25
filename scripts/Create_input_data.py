import pandas as pd
import numpy as np
import pybedtools
import random
import argparse
import os

# ----------------------------
# Argument parsing
# ----------------------------
parser = argparse.ArgumentParser(
    description="Create input data for DNABERT-Enhancer"
)

parser.add_argument("--input_bed", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--Reference_Genome_FilePath", type=str, required=True)
parser.add_argument("--kmer_length", type=int, required=True)
parser.add_argument("--length_limit", type=int, required=True)

args = parser.parse_args()

input_bed = os.path.join(args.input_bed)
output_folder_path = args.output_dir
Reference_Genome_FilePath = args.Reference_Genome_FilePath
kmer_length = args.kmer_length
length_limit = args.length_limit


# ----------------------------------
# Fetch sequence for the coordinate
# ----------------------------------
sequence_df = pd.DataFrame(columns=['chr','start','end','sequence'])
RowIndex = 0

with open(input_bed, "r") as f:
    for i in f:
        bed_data = i.strip().split("\t")

        if len(bed_data) < 3:
            continue

        if int(bed_data[2]) - int(bed_data[1]) >= length_limit:
            start = int(bed_data[1]) + 1
            MyCoord = bed_data[0] + ":" + str(start) + "-" + bed_data[2]
            GenomeFile = Reference_Genome_FilePath + bed_data[0] + ".fa"

            sequence_df.loc[RowIndex, 'chr'] = bed_data[0]
            sequence_df.loc[RowIndex, 'start'] = bed_data[1]
            sequence_df.loc[RowIndex, 'end'] = bed_data[2]
            sequence_df.loc[RowIndex, 'sequence'] = pybedtools.BedTool.seq(MyCoord, GenomeFile)

            RowIndex += 1

sequence_df['length']=sequence_df['sequence'].str.len()
sequence_df['sequence']=sequence_df['sequence'].str.upper()
sequence_sub_df=sequence_df.loc[sequence_df['sequence'].str.match("[^N]")]

# ----------------------------------
# Create the overlapping kmers
# ----------------------------------
def seq2kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    allkmer = " ".join(kmer)
    
    return allkmer

chunk_l = len(sequence_sub_df['sequence'])
kmers=[]

for i in range(chunk_l):
    kmers.append(seq2kmer(sequence_sub_df['sequence'][i],kmer_length))

kmerdf=pd.DataFrame(kmers)
merged_df=pd.concat([sequence_sub_df,kmerdf], axis=1)
merged_df.rename(columns={0: "Sequence"}, inplace=True)
merged_df['Label'] = np.random.randint(0, 2, size=len(merged_df))

final_kmer=merged_df[['Sequence','Label']]
final_kmer.to_csv(output_folder_path+"dev.tsv", sep="\t", index=False)
