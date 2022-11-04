from pathlib import Path
from argparse import ArgumentParser
import os
from typing import List, Union
import subprocess

# This assumes we are running in VOC environment...
# Can change to a more standard fasta reader if we want
from voc.utils import read_fasta, write_fasta

node_rank = int(os.environ.get("NODE_RANK", 0))  # zero indexed
pmi_rank = int(os.environ.get("PMI_LOCAL_RANK", 0))


def find_workfiles(in_files: List[Union[Path, str]]) -> List[Union[Path, str]]:

    num_nodes = int(os.environ.get("NRANKS", 1))

    gpu_rank = (node_rank * 4) + pmi_rank
    num_gpus = num_nodes * 4
    if num_gpus > 1:
        chunk_size = len(in_files) // num_gpus
        start_idx = node_rank * chunk_size
        end_idx = start_idx + chunk_size
        if node_rank + 1 == num_gpus:
            end_idx = len(in_files)

        print(
            f"GPU {gpu_rank}/ {num_gpus} starting at {start_idx}, ending at {end_idx} ({len(in_files)=}"
        )
        node_data = in_files[start_idx:end_idx]
    else:
        node_data = in_files[:]

    return node_data


def run_openfold(in_fasta_dir: Path, out_dir: Path, test: bool = False) -> int:
    command = f"""python run_pretrained_openfold.py \\
"{in_fasta_dir}" \\
/lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/data/pdb_mmcif/mmcif_files/ \\
--uniref90_database_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/data/uniref90/uniref90.fasta \\
--mgnify_database_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/data/mgnify/mgy_clusters_2018_12.fa \\
--pdb70_database_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/data/pdb70/pdb70 \\
--uniclust30_database_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \\
--output_dir "{out_dir}" \\
--bfd_database_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \\
--model_device "cuda:{pmi_rank}" \\
--jackhmmer_binary_path /lus/eagle/projects/CVD-Mol-AI/hippekp/conda/envs/voc/bin/jackhmmer \\
--hhblits_binary_path /lus/eagle/projects/CVD-Mol-AI/hippekp/conda/envs/voc/bin/hhblits \\
--hhsearch_binary_path /lus/eagle/projects/CVD-Mol-AI/hippekp/conda/envs/voc/bin/hhsearch \\
--kalign_binary_path /lus/eagle/projects/CVD-Mol-AI/hippekp/conda/envs/voc/bin/kalign \\
--config_preset "model_1_ptm" \\
--openfold_checkpoint_path /lus/eagle/projects/CVD-Mol-AI/hippekp/workflow_data/openfold/resources/openfold_params/finetuning_ptm_2.pt"""

    if test:
        out_dir.mkdir(exist_ok=True, parents=True)
        with open(out_dir / "test_out.out", "w") as f:
            f.write("Out data")
        print("*" * 50, f"Testing, command: \n{command}")
        return 0

    res = subprocess.run(command.split())
    return res.returncode


def main(fasta: Path, out_dir: Path, glob_pattern: str, test: bool):
    out_dir.mkdir(exist_ok=True, parents=True)
    fasta_temp_dir = out_dir / "tmp_fasta"
    fasta_temp_dir.mkdir(exist_ok=True, parents=True)
    fasta_files = []
    if fasta.is_file():
        # Write each seq to a temp fasta file inside a dir
        for seq in read_fasta(fasta):
            temp_dir = fasta_temp_dir / seq.tag
            temp_dir.mkdir(exist_ok=True, parents=True)
            fasta_temp_file = temp_dir / f"{seq.tag.split('|')[0]}.fasta"
            write_fasta(seq, fasta_temp_file)
            fasta_files.append(fasta_temp_file)

    else:  # Is a directory of fasta files
        # Assuming just one seq per fasta file
        for file in fasta.glob(glob_pattern):
            seq = read_fasta(file)[0]  # Here is the one seq assumption
            temp_dir = fasta_temp_dir / seq.tag
            temp_dir.mkdir(exist_ok=True, parents=True)
            fasta_temp_file = temp_dir / f"{seq.tag.split('|')[0]}.fasta"
            write_fasta(seq, fasta_temp_file)
            fasta_files.append(fasta_temp_file)

    node_files = find_workfiles(fasta_files)

    for file in node_files:
        file_dir = file.parent.name
        file_out_dir = out_dir / file_dir

        status_code = run_openfold(file_dir, file_out_dir, test)
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
    parser.add_argument("-t", "--test", action="store_true")

    args = parser.parse_args()

    main(args.fasta, args.out_dir, args.glob_pattern, args.test)
