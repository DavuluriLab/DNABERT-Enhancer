import argparse
import os
import subprocess
import tempfile
import pandas as pd
import pybedtools


# generating negative instances
# --------------------------------
def define_negative_instance(pred_file, genome_file, blacklist_file, tmpdir):

    os.makedirs(tmpdir, exist_ok=True)

    neg_file = os.path.join(tmpdir, "Negative_instance.bed")
    shuffle_cmd = [
        "bedtools", "shuffle", 
        "-i", pred_file,
        "-g", genome_file,
        "-excl", blacklist_file,
        "-chrom"]

    print(f"Generating negative instances: {neg_file}")
    with open(neg_file, "w") as f:
        subprocess.run(shuffle_cmd, stdout=f, check=True)

    return neg_file


# combinig positive and negative instances
# ------------------------------------------
def combine_bed_file(pred_file, neg_file, tmpdir):

    combined_file = os.path.join(tmpdir, "combined.bed")
    
    print(f"Combined data: {combined_file}")
    df_pos = pd.read_csv(pred_file, sep="\t", header=None)
    df_pos["label"] = 1
    df_neg = pd.read_csv(neg_file, sep="\t", header=None)
    df_neg["label"] = 0
    combined = pd.concat([df_pos, df_neg])
    combined.to_csv(combined_file, sep="\t", header=False, index=False)

    return combined_file


# fetching sequences
# ----------------------
def fetch_sequence(comb_file, ref, length_limit, tmpdir):
    
    comb_data_df = pd.read_csv(comb_file, sep="\t", header=None)
    comb_data_df.rename(columns={0:'chr',1:'start',2:'end',3:'label'}, inplace=True)
    
    sequence_file = os.path.join(tmpdir, "sequence_with_labels.csv")
    
    sequence_df = pd.DataFrame(columns=['chr','start','end','sequence'])
    RowIndex = 0
    with open(comb_file, "r") as f:
        for i in f:
            bed_data = i.strip().split("\t")
            start_int = int(bed_data[1])
            end_int = int(bed_data[2])

            if end_int - start_int >= length_limit:
                GenomeFile = os.path.join(ref, bed_data[0] + ".fa")
                bed_line = f"{bed_data[0]}\t{start_int}\t{end_int}"
                region = pybedtools.BedTool(bed_line, from_string=True)
                fasta_path = region.sequence(fi=GenomeFile).seqfn

                with open(fasta_path, "r") as seq_f:
                    fasta_seq = "".join([l.strip() for l in seq_f if not l.startswith(">")])

                sequence_df.loc[RowIndex, 'chr'] = bed_data[0]
                sequence_df.loc[RowIndex, 'start'] = start_int
                sequence_df.loc[RowIndex, 'end'] = end_int
                sequence_df.loc[RowIndex, 'sequence'] = fasta_seq
                RowIndex += 1
    sequence_df['length']=sequence_df['sequence'].str.len()
    sequence_df['sequence']=sequence_df['sequence'].str.upper()
    sequence_sub_df=sequence_df.loc[~sequence_df['sequence'].str.contains("N")]
    merged_seq_df=pd.merge(comb_data_df,sequence_sub_df, on=['chr','start','end'])
    merged_seq_df.to_csv(sequence_file, index=False)
    
    return sequence_file


# generating fasta file
# ----------------------
def generate_fasta(seq, out):

    seq_df=pd.read_csv(seq)
    seq_df['id']=seq_df['chr']+":"+seq_df['start'].astype(str)+"-"+seq_df['end'].astype(str)+"; label="+seq_df['label'].astype(str)
    fasta_file = os.path.join(out+"Positive_and_negative_instances_with_sequences.fasta')

    with open(fasta_file, 'w') as f:
        for index, row in seq_df.iterrows():
            f.write(f">{row['id']}\n")
            f.write(f"{row['sequence']}\n")

    return fasta_file


    
def main():
    parser = argparse.ArgumentParser(
    description="Create negative set from reference genome"
    )
    
    parser.add_argument("--pred", required=True)
    parser.add_argument("--ref_genome_filepath", required=True)
    parser.add_argument("--genome_size_file", required=True)
    parser.add_argument("--blacklist", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tmpdir", required=True)
    parser.add_argument("--length_limit", type=int, required=True)
    
    args = parser.parse_args()

    neg_file = define_negative_instance(args.pred, args.genome_size_file, args.blacklist, args.tmpdir)
    combined_file = combine_bed_file(args.pred, neg_file, args.tmpdir)
    sequence_file = fetch_sequence(combined_file, args.ref_genome_filepath, args.length_limit, args.tmpdir)
    fasta_file = generate_fasta(sequence_file, args.out)
    
    print(f"Negative instances created at: {neg_file}")
    print(f"Combined BED file created at: {combined_file}")
    print(f"Sequence file created at: {sequence_file}")
    print(f"Fasta file created at: {fasta_file}")

if __name__ == "__main__":
    main()