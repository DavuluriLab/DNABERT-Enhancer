import argparse
import os
import subprocess
import tempfile
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split


# generating kmers
# ----------------------
def generate_kmer_file(fasta, kmer_length, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)

    kmers_list = []
    labels_list = []

    for record in SeqIO.parse(fasta, "fasta"):
        seq = str(record.seq)
       
        try:
            label = int(record.id.split("label=")[-1].strip())
        except:
            raise ValueError(f"FASTA ID does not contain label after 'label=': {record.id}")

        # Generate kmers
        kmers = " ".join([seq[i:i+kmer_length] for i in range(len(seq) - kmer_length + 1)])
        kmers_list.append(kmers)
        labels_list.append(label)

    kmer_df = pd.DataFrame({
        "Sequence": kmers_list,
        "Label": labels_list
    })

    # Save to temporary file
    kmer_file = os.path.join(tmpdir, "kmer_temp.tsv")
    kmer_df.to_csv(kmer_file, sep="\t", index=False)
    print(f"K-mer file saved to temporary folder: {kmer_file}")

    return kmer_file


# splitting file
# ----------------------
def split_and_save(kmer_file, split_ratio, out_dir):
    
    os.makedirs(out_dir, exist_ok=True)

    kmer_df=pd.read_csv(kmer_file, sep="\t")
    # Parse split string
    parts = [int(x) for x in split_ratio.strip().split(":")]
    if len(parts) not in [2, 3]:
        raise ValueError("Split must be two-part (train:test) or three-part (train:dev:test)")

    total = sum(parts)
    fractions = [p / total for p in parts]

    if len(fractions) == 2:
        train_df, test_df = train_test_split(
            kmer_df, train_size=fractions[0], random_state=42, stratify=kmer_df['Label']
        )
        train_df.to_csv(os.path.join(out_dir, f"train.tsv"), sep="\t", index=False)
        test_df.to_csv(os.path.join(out_dir, f"dev.tsv"), sep="\t", index=False)
        print(f"Train/Test saved: {len(train_df)} / {len(test_df)} sequences")
    else:
        train_frac, dev_frac, test_frac = fractions
        train_df, temp_df = train_test_split(
            kmer_df, train_size=train_frac, random_state=42, stratify=kmer_df['Label']
        )
        dev_size = dev_frac / (dev_frac + test_frac)
        dev_df, test_df = train_test_split(
            temp_df, train_size=dev_size, random_state=42, stratify=temp_df['Label']
        )
        train_df.to_csv(os.path.join(out_dir, f"train.tsv"), sep="\t", index=False)
        dev_df.to_csv(os.path.join(out_dir, f"dev.tsv"), sep="\t", index=False)
        test_df.to_csv(os.path.join(out_dir, f"eval.tsv"), sep="\t", index=False)
        print(f"Train/Dev/Test saved: {len(train_df)} / {len(dev_df)} / {len(test_df)} sequences")


def main():
    parser = argparse.ArgumentParser(
    description="Create fine-tune dataset"
    )

    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tmpdir", required=True)
    parser.add_argument("--kmer_length", required=True)
    parser.add_argument("--split", required =True)

    args = parser.parse_args()

    kmer_file = generate_kmer_file(args.fasta, int(args.kmer_length), args.tmpdir)
    split_and_save(kmer_file, args.split, args.out)


if __name__ == "__main__":
    main()