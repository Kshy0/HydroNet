"""Step 3: spin up CaMa-Flood-GPU on a HydroNet map and save its state as HydroNet's restart.

    python -m scripts.s03_run_cama_spinup --params inp/glb_15min/parameters.nc \\
        --runoff-dir RUNOFF_DIR --runoff-mapping inp/glb_15min/runoff_mapping_bin.npz \\
        --start 1979-01-01 --end 1979-12-31 --output out/cama_spinup

Training starts from the CaMa-Flood storage and outflow at the end of this run
(``paths.cama_restart``; loaded by ``scripts.training.data.load_restart_state``). The run
also writes CaMa-Flood's daily mean total outflow for comparison with HydroNet.
Forcing is daily binary runoff on a regular grid, as in cmfgpu's
scripts/run_daily_bin.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import torch
import torch.distributed as dist
from cmfgpu.models import CaMaFlood
from hydroforge.data import InputProxy
from hydroforge.data.datasets import DailyBinDataset
from hydroforge.model import OutputConfig
from hydroforge.parallel import setup_distributed
from torch.utils.data import DataLoader


def run(args) -> None:
    distributed = setup_distributed(allowed_devices=("cuda",))
    device = distributed.device
    day = timedelta(days=1)
    dataset = DailyBinDataset(
        base_dir=args.runoff_dir, shape=tuple(args.runoff_shape),
        start_date=args.start, end_date=args.end, time_interval=day, model_step=day,
        spin_up_cycles=args.spin_up_cycles,
        spin_up_start_date=args.start if args.spin_up_cycles else None,
        spin_up_end_date=args.end if args.spin_up_cycles else None,
        source_units="mm day-1", target_units="m s-1", bin_dtype="float32",
        prefix=args.prefix, suffix=args.suffix, lat_south_to_north=False, lon_0_to_360=False,
    )
    model = CaMaFlood(
        device=device,
        output=OutputConfig(experiment=args.experiment, dir=args.output,
                            variables={"mean": ["total_outflow"]}),
        input_proxy=InputProxy.from_nc(args.params),
        opened_modules=(("base", "adaptive_time", "bifurcation") if args.bifurcation
                        else ("base", "adaptive_time")),
        simulation_schedule=dataset.simulation_schedule,
    )
    with model:
        dataset, mapping = dataset.build_local_mapping(
            args.runoff_mapping, model.base.catchment_id.cpu().numpy(), device=device)
        for chunk in DataLoader(dataset, batch_size=None, num_workers=2, pin_memory=True):
            for runoff in dataset.shard_forcing(chunk.to(device, non_blocking=True), mapping):
                model.set_inputs(runoff=runoff)
                model.step_advance()
        restart = model.save_state()
        if distributed.rank == 0:
            print("CaMa-Flood restart:", restart.file_path, flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    date = lambda text: datetime.fromisoformat(text)  # noqa: E731
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bifurcation", action="store_true",
                        help="include bifurcation pathways and save their recurrent discharge")
    parser.add_argument("--params", required=True, help="CaMa-Flood parameters.nc of the HydroNet map")
    parser.add_argument("--runoff-dir", required=True)
    parser.add_argument("--runoff-mapping", required=True, help="runoff_mapping_bin.npz of the map")
    parser.add_argument("--runoff-shape", type=int, nargs=2, default=(180, 360))
    parser.add_argument("--prefix", default="Roff____")
    parser.add_argument("--suffix", default=".one")
    parser.add_argument("--start", type=date, required=True)
    parser.add_argument("--end", type=date, required=True)
    parser.add_argument("--spin-up-cycles", type=int, default=0,
                        help="repeat the period this many times before the recorded run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", default="cama_spinup")
    torch.set_num_threads(2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
