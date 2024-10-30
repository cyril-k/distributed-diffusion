import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from xfuser import xFuserCogVideoXPipeline, xFuserArgs
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
)
from diffusers.utils import export_to_video


def setup_dist(rank, world_size):
    dist.init_process_group(
        backend='nccl', 
        # init_method='env://', 
        rank=rank, 
        world_size=world_size
    )

def cleanup_dist():
    dist.destroy_process_group()

def run_dist(
        # rank,
        # world_size
    ):
    # setup_dist(rank, world_size)
    
    args = xFuserArgs(
        model="THUDM/CogVideoX-5b",
        tensor_parallel_degree=1,
        ulysses_degree=2,
        ring_degree=1,
        # use_cfg_parallel=True,
        # use_cfg_parallel=False,
        height=480,
        width=720,
        num_frames=40,
        num_inference_steps=30,
        warmup_steps=0,
        prompt=[
            "a wide ribbon converges on itself like a moebius tape, it has a color gradient.",
            """The scene is set in a maze-like structure filled with staircases that twist and turn in impossible directions. Multiple figures walk up and down the stairs, defying traditional notions of gravity; some walk horizontally, others vertically, yet all appear to inhabit the same space seamlessly. The architecture is geometric and precise, with arches, doorways, and railings that give a sense of balance, despite the disorienting perspectives.""",
            """Escher's hand is prominently in the foreground, holding the sphere delicately between his fingers, while his face is reflected centrally, gazing intently at the viewer. The surrounding room, including a window, shelves, and furniture, wraps around the sphere's surface, bending into exaggerated curves due to the reflection's distortion. The meticulous shading and attention to detail bring a strikingly realistic quality to the reflection, creating a sense of depth and immersion."""
            ],
    )

    engine_config, input_config = args.create_config()
    # cleanup_dist()

    local_rank = get_world_group().local_rank

    pipe = xFuserCogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.bfloat16,
    )

    device = torch.device(f"cuda:{local_rank}")
    pipe = pipe.to(device)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    output_type = "latents"
    # output_type = "pil"

    output = pipe(
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        prompt=input_config.prompt,
        num_inference_steps=input_config.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        guidance_scale=6,
        output_type=output_type,
    ).frames[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    parallel_info = (
        f"dp{args.data_parallel_degree}_cfg{engine_config.parallel_config.cfg_degree}_"
        f"ulysses{args.ulysses_degree}_ring{args.ring_degree}_"
        f"tp{args.tensor_parallel_degree}_"
        f"pp{args.pipefusion_parallel_degree}_patch{args.num_pipeline_patch}"
    )
    if is_dp_last_group():
        if output_type != "latents":
            world_size = get_data_parallel_world_size()
            resolution = f"{input_config.width}x{input_config.height}"
            output_filename = f"results/cogvideox_{parallel_info}_{resolution}.mp4"
            export_to_video(output, output_filename, fps=8)
            print(f"output saved to {output_filename}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()

if __name__ == "__main__":
    run_dist()