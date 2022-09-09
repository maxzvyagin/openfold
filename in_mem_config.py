from pydantic import BaseSettings as _BaseSettings
from pathlib import Path
from typing import Union, Type, TypeVar
from datetime import date
import yaml
import json

PathLike = Union[str, Path]
_T = TypeVar("_T")


class BaseSettings(_BaseSettings):
    """Base settings to provide an easier interface to read/write YAML files."""

    def dump_yaml(self, cfg_path: PathLike) -> None:
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class FoldingArguments(BaseSettings):
    """Class for all needed folding object arguments"""
    config_preset: str
    template_mmcif_dir: Path
    model_device: str
    openfold_checkpoint_path: Path
    obsolete_pdbs_path: Path
    uniref90_database_path: Path = Path("data/uniref90/uniref90.fasta")
    mgnify_database_path: Path = Path("data/mgnify/mgy_clusters_2018_12.fa ")
    bfd_database_path: Path = Path("data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt")
    pdb70_database_path: Path = Path("data/pdb70/pdb70")
    jackhmmer_binary_path: str = "jackhmmer"
    hhblits_binary_path: str = "hhblits"
    hhsearch_binary_path: str = "hhsearch"
    kalign_binary_path: str = "kalign"
    random_seed: int = None  # will be selected by program if not provided
    multimer_ri_gap: int = 200  # Residue index offset between multiple sequences, if provided
    skip_relaxation: bool = False  # whether to perform relaxation on output pdb
    save_outputs: bool = True  # save pkl files as well as pdbs
    max_template_date: str = date.today().strftime("%Y-%m-%d")
    release_dates_path: str = None

if __name__ == "__main__":
    settings = FoldingArguments(
        config_preset="model_1_ptm",
        template_mmcif_dir=Path("data/pdb_mmcif/mmcif_files/"),
        model_device="cuda:0",
        openfold_checkpoint_path=Path("openfold/resources/openfold_params/finetuning_ptm_2.pt"),
        obsolete_pdbs_path=Path("data/pdb_mmcif/obsolete.dat"),

    )
    settings.dump_yaml("settings_template.yaml")
