import logging
import numpy as np
import os

from openfold.utils.script_utils import (
    parse_fasta,
    run_model,
    prep_output,
    relax_protein,
)
from openfold.model.model import AlphaFold
from in_mem_config import FoldingArguments

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import pickle

import random
import torch
from pathlib import Path

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if torch_major_version > 1 or (torch_major_version == 1 and torch_minor_version >= 12):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.np import protein

from openfold.utils.tensor_utils import (
    tensor_tree_map,
)


def load_model(config, path, model_device):
    model = AlphaFold(config)
    model = model.eval()
    ckpt_path = path
    d = torch.load(ckpt_path)
    model.load_state_dict(d)
    model = model.to(model_device)
    logger.info(f"Loaded OpenFold parameters at {path}...")
    return model


def compute_alignments(input_file, tag, alignment_dir, args, local_alignment_dir):
    """Compute alignment for a single sequence"""
    if not os.path.isdir(local_alignment_dir):
        logger.info(f"Generating alignments for {tag}...")

        alignment_runner = data_pipeline.AlignmentRunner(
            jackhmmer_binary_path=args.jackhmmer_binary_path,
            hhblits_binary_path=args.hhblits_binary_path,
            hhsearch_binary_path=args.hhsearch_binary_path,
            uniref90_database_path=args.uniref90_database_path,
            mgnify_database_path=args.mgnify_database_path,
            bfd_database_path=args.bfd_database_path,
            uniclust30_database_path=args.uniclust30_database_path,
            pdb70_database_path=args.pdb70_database_path,
            no_cpus=args.cpus,
        )
        alignment_runner.run(input_file, local_alignment_dir)
    else:
        logger.info(f"Using precomputed alignments for {tag} at {alignment_dir}...")


class FoldingObject:
    def __init__(self, args: FoldingArguments):
        self.args = args
        self.config = model_config(self.args.config_preset)
        self.template_featurizer = templates.TemplateHitFeaturizer(
            mmcif_dir=self.args.template_mmcif_dir,
            max_template_date=self.args.max_template_date,
            max_hits=self.config.data.predict.max_templates,
            kalign_binary_path=self.args.kalign_binary_path,
            release_dates_path=self.args.release_dates_path,
            obsolete_pdbs_path=self.args.obsolete_pdbs_path,
        )

        self.data_processor = data_pipeline.DataPipeline(
            template_featurizer=self.template_featurizer,
        )

        if self.args.random_seed is None:
            self.args.random_seed = random.randrange(2 ** 32)
        np.random.seed(self.args.random_seed)
        torch.manual_seed(self.args.random_seed + 1)

        self.feature_processor = feature_pipeline.FeaturePipeline(self.config.data)

        self.model = load_model(
            self.config, self.args.openfold_checkpoint_path, self.args.model_device
        )

        self.feature_dicts = {}

    def fold(self, input_file: Path, output_dir_base: Path) -> None:
        with open(input_file, "r") as fp:
            data = fp.read()
        tags, seqs = parse_fasta(data)
        tag = "-".join(tags)
        output_name = f"{tag}_{self.args.config_preset}"
        if not os.path.exists(output_dir_base):
            os.makedirs(output_dir_base)
        alignment_dir = os.path.join(output_dir_base, "alignments")
        local_alignment_dir = os.path.join(alignment_dir, tag)
        compute_alignments(
            input_file, tags, alignment_dir, self.args, local_alignment_dir
        )
        # compute features
        feature_dict = self.feature_dicts.get(tag, None)
        if feature_dict is None:
            feature_dict = self.data_processor.process_fasta(
                fasta_path=input_file, alignment_dir=local_alignment_dir
            )
            self.feature_dicts[tag] = feature_dict
        processed_feature_dict = self.feature_processor.process_features(
            feature_dict,
            mode="predict",
        )
        processed_feature_dict = {
            k: torch.as_tensor(v, device=self.args.model_device)
            for k, v in processed_feature_dict.items()
        }
        # run the model inference
        out = run_model(self.model, processed_feature_dict, tag, output_dir_base)
        # Now write to file
        # Toss out the recycling dimensions --- we don't need them anymore
        processed_feature_dict = tensor_tree_map(
            lambda x: np.array(x[..., -1].cpu()), processed_feature_dict
        )
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)
        unrelaxed_protein = prep_output(
            out,
            processed_feature_dict,
            feature_dict,
            self.feature_processor,
            self.args.config_preset,
            self.args.multimer_ri_gap,
            self.args.subtract_plddt
        )
        unrelaxed_output_path = os.path.join(
            output_dir_base, f"{output_name}_unrelaxed.pdb"
        )
        with open(unrelaxed_output_path, "w") as fp:
            fp.write(protein.to_pdb(unrelaxed_protein))
        logger.info(f"Output written to {unrelaxed_output_path}...")

        # now perform relaxation if requested
        # Relax the prediction.
        if not self.args.skip_relaxation:
            logger.info(f"Running relaxation on {unrelaxed_output_path}...")
            relax_protein(
                self.config,
                self.args.model_device,
                unrelaxed_protein,
                output_dir_base,
                output_name,
            )

        if self.args.save_outputs:
            output_dict_path = os.path.join(
                output_dir_base, f"{output_name}_output_dict.pkl"
            )
            with open(output_dict_path, "wb") as fp:
                pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Model output written to {output_dict_path}...")
