# NOTICE you should be using the voc environment because this uses nextclade

import re
from pathlib import Path
from argparse import ArgumentParser
import os
from typing import List, Union
import subprocess

from pydantic import BaseModel

node_rank = int(os.environ.get("NODE_RANK", 0))  # zero indexed
pmi_rank = int(os.environ.get("PMI_RANK", 0))

PathLike = Union[Path, str]


class Sequence(BaseModel):
    sequence: str
    """Biological sequence (Nucleotide sequence)."""
    tag: str
    """Sequence description tag."""


def read_fasta(fasta_file: PathLike) -> List[Sequence]:
    """Caches the last 8 weeks worth of data in memory."""
    text = Path(fasta_file).read_text()
    text = re.sub(">$", "", text, flags=re.M)
    lines = [
        line.replace("\n", "")
        for seq in text.split(">")
        for line in seq.strip().split("\n", 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    return [Sequence(sequence=seq, tag=tag) for seq, tag in zip(seqs, tags)]


def write_fasta(
    sequences: Union[Sequence, List[Sequence]], fasta_file: PathLike, mode: str = "w"
) -> None:
    seqs = [sequences] if isinstance(sequences, Sequence) else sequences
    with open(fasta_file, mode) as f:
        for seq in seqs:
            f.write(f">{seq.tag}\n{seq.sequence}\n")


def get_rbd_region(genome_sequence: Sequence, workdir: Path) -> Sequence:
    """Given a SARS-CoV-2 genome, run nextclade, get the aligned peptides, and
    return the RBD region as string"""

    # Save to a temp fasta file in order to run on nextclade
    genome_fasta = workdir / "genome.fasta"
    nextclade_outdir = workdir / "nextclade_output"
    write_fasta(genome_sequence, genome_fasta)

    command = (
        f"nextclade run --input-dataset /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/nextclade/data/sars-cov-2 "
        f"--output-all {nextclade_outdir} {genome_fasta}"
    )
    # Run nextclade
    subprocess.run(command.split())

    # Extract the peptide string from nextclade
    seq = read_fasta(nextclade_outdir / "nextclade_gene_S.translation.fasta")[0]
    peptide = seq.sequence
    # Isolate the rbd region with pattern
    rbd = peptide.split("FGE")[-1].split("LSF")[0]
    # Add back in the leading and trailing pattern
    rbd = "FGE" + rbd + "LSF"
    # Replace any ambiguous or gap chars for openfold
    rbd = rbd.replace("-", "").replace("X", "")
    seq.sequence = rbd
    return seq


def find_workseqs(in_files: List[Sequence]) -> List[Sequence]:

    num_nodes = int(os.environ.get("NRANKS", 1))
    num_gpus = num_nodes * 4
    gpu_rank = pmi_rank
    if num_gpus > 1:
        chunk_size = max(len(in_files) // num_gpus, 1)
        start_idx = gpu_rank * chunk_size
        end_idx = start_idx + chunk_size
        if gpu_rank + 1 == num_gpus:
            end_idx = len(in_files)

        print(
            f"GPU {gpu_rank} / {num_gpus} starting at {start_idx}, ending at {end_idx} ({len(in_files)=})"
        )
        node_data = in_files[start_idx:end_idx]
    else:
        node_data = in_files[:]

    return node_data


def run_openfold(in_fasta_dir: Path, out_dir: Path, test: bool = False) -> int:
    command = (
        "python /lus/eagle/projects/CVD-Mol-AI/hippekp/github/openfold/run_pretrained_openfold.py "
        + f"{in_fasta_dir} "
        + "/lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/multinode_data/pdb_mmcif/mmcif_files/ "
        + "--uniref90_database_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/multinode_data/uniref90/uniref90.fasta "
        + "--mgnify_database_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/multinode_data/mgnify/mgy_clusters_2018_12.fa "
        + "--pdb70_database_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/multinode_data/pdb70/pdb70 "
        + "--uniclust30_database_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/multinode_data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 "
        + f"--output_dir {out_dir} "
        + "--bfd_database_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/multinode_data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt "
        + f"--model_device cuda:{pmi_rank%4} "
        + "--jackhmmer_binary_path /lus/eagle/projects/CVD-Mol-AI/hippekp/conda/envs/voc/bin/jackhmmer "
        + "--hhblits_binary_path /lus/eagle/projects/CVD-Mol-AI/hippekp/conda/envs/voc/bin/hhblits "
        + "--hhsearch_binary_path /lus/eagle/projects/CVD-Mol-AI/hippekp/conda/envs/voc/bin/hhsearch "
        + "--kalign_binary_path /lus/eagle/projects/CVD-Mol-AI/hippekp/conda/envs/voc/bin/kalign "
        + "--config_preset model_1_ptm "
        + "--openfold_checkpoint_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/resources/openfold_params/finetuning_ptm_2.pt"
    )

    if test:
        out_dir.mkdir(exist_ok=True, parents=True)
        with open(out_dir / "test_out.out", "w") as f:
            f.write("Out data")
        print("*" * 50, f"Testing, command: \n{command}")
        return 0

    res = subprocess.run(command.split())
    return res.returncode


def main(fasta: Path, out_dir: Path, glob_pattern: str, test: bool, nextclade: bool):
    out_dir.mkdir(exist_ok=True, parents=True)
    fasta_temp_dir = out_dir / "tmp_fasta"
    fasta_temp_dir.mkdir(exist_ok=True, parents=True)
    seqs = []
    if fasta.is_file():
        # Write each seq to a temp fasta file inside a dir
        for seq in read_fasta(fasta):
            fasta_temp_file = fasta_temp_dir / f"{seq.tag}.fasta"
            if not (out_dir / fasta_temp_file.stem).is_dir():
                seqs.append(seq)

    else:  # Is a directory of fasta files
        # Assuming just one seq per fasta file
        for file in fasta.glob(glob_pattern):
            seq = read_fasta(file)[0]  # Here is the one seq assumption
            fasta_temp_file = fasta_temp_dir / f"{seq.tag}.fasta"
            if not (out_dir / fasta_temp_file.stem).is_dir():
                seqs.append(seq)

    node_seqs = find_workseqs(seqs)

    for seq in node_seqs:
        seq_temp_dir = fasta_temp_dir / seq.tag
        seq_temp_dir.mkdir(exist_ok=True, parents=True)
        seq_temp_file = seq_temp_dir / f"{seq.tag}.fasta"
        if not seq_temp_file.is_file():
            write_fasta(seq, seq_temp_file)

        if nextclade:
            try:
                seq = get_rbd_region(seq, seq_temp_dir)
                rbd_spike_path = fasta_temp_dir / f"{seqs.tag}_rbd"
                rbd_spike_path.mkdir(exist_ok=True)
                seq_temp_dir = rbd_spike_path

                rbd_spike_path = rbd_spike_path / "rbd.fasta"
                write_fasta(seq, rbd_spike_path)
                print(f"Spike rbd: {seq} saved to path: {rbd_spike_path}")
            except IndexError:
                print(f"Nextclade failed on '{seq.tag}'")

        file_out_dir = out_dir / seq_temp_file.stem

        status_code = run_openfold(seq_temp_dir, file_out_dir, test)
        if status_code != 0:
            print(f"Error running {file}... continuing")

    print(f"Finished folding on gpu {pmi_rank} of rank {node_rank}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--fasta",
        type=Path,
        required=True,
        help="Directory of fastas or single fasta file",
    )
    parser.add_argument("-o", "--out_dir", type=Path, help="Path to output directory")
    parser.add_argument(
        "-g",
        "--glob_pattern",
        type=str,
        help="Glob pattern to search directory for fasta files (defaults to *.fasta)",
        default="*.fasta",
    )

    parser.add_argument(
        "--nextclade",
        action="store_true",
        help="Run nextclade to get RBD region of spike",
    )
    parser.add_argument("-t", "--test", action="store_true")

    args = parser.parse_args()

    main(args.fasta, args.out_dir, args.glob_pattern, args.test, args.nextclade)
