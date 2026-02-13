
# from openpi.training import config as config_pi
# from openpi.policies import policy_config
# from openpi_client import image_tools
import numpy as np


from accelerate import Accelerator
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import key_board_control, get_fk_solution

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import wandb
import json
from decord import VideoReader, cpu
import swanlab
import imageio
import sys
from scipy.spatial.transform import Rotation as R



class agent():
    def __init__(self,args):
          
        # args = Args()
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # # load pi policy
        # if 'pi05' in args.policy_type:
        #     config = config_pi.get_config("pi05_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid' 
        # elif 'pi0fast' in args.policy_type:
        #     config = config_pi.get_config("pi0fast_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0fast_droid'
        # elif 'pi0' in args.policy_type:
        #     config = config_pi.get_config("pi0_droid")
        #     checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0_droid'
        # else:
        #     raise ValueError(f"Unknown policy type: {args.policy_type}")
        # self.policy = policy_config.create_trained_policy(config, checkpoint_dir)

        # load ctrl-world model
        # map_location handles checkpoints saved on CUDA when loading on different device
        self.model = CrtlWorld(args)
        self.model.load_state_dict(torch.load(args.val_model_path, map_location=self.accelerator.device))
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        print("load world model success")
        with open(f"{args.data_stat_path}", 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None,:]
            self.state_p99 = np.array(data_stat['state_99'])[None,:]
        

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def get_traj_info(self, id, start_idx=0, steps=8, split='val'):
        val_dataset_dir = self.args.val_dataset_dir
        args = self.args
        skip = args.skip_step
        num_frames = steps
        annotation_path = f"{val_dataset_dir}/annotation/{split}/{id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno['action'])
            except:
                length = anno["video_length"]
        frames_ids = np.arange(start_idx, start_idx + num_frames * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        print("Ground truth frames ids", frames_ids)

        # get action and joint pos
        instruction = anno['texts'][0]
        car_action = np.array(anno['states'])
        car_action = car_action[frames_ids]
        
        # Handle both 'joints' and 'states' (some datasets only have states)
        if 'joints' in anno:
            joint_pos = np.array(anno['joints'])
            joint_pos = joint_pos[frames_ids]
        else:
            # Use states as joint positions if joints not available
            # Pad to 8 dimensions if needed (states are 7D, joints are 8D)
            joint_pos = np.array(anno['states'])
            if joint_pos.shape[-1] == 7:
                # Add an extra dimension (set to 0 for gripper joint)
                joint_pos = np.pad(joint_pos, ((0, 0), (0, 1)), mode='constant', constant_values=0)
            joint_pos = joint_pos[frames_ids]

        # get videos
        video_dict =[]
        video_latent = []
        for id in range(len(anno['videos'])):
            video_path = anno['videos'][id]['video_path']
            video_path = f"{val_dataset_dir}/{video_path}"
            # load videos from all views
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except:
                true_video = vr.get_batch(range(length)).numpy()
            true_video = true_video[frames_ids]
            video_dict.append(true_video)

            # encode video
            device = self.device
            true_video = torch.from_numpy(true_video).to(self.dtype).to(device)
            x = true_video.permute(0,3,1,2).to(device) / 255.0*2-1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                batch_size = 32
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
    
            video_latent.append(x)

        
        return car_action, joint_pos, video_dict, video_latent, instruction

    def forward_wm(self, action_cond, video_latent_true, video_latent_cond, his_cond=None, text=None, 
                   action_noise=0.0, latent_noise=0.0, num_inference_steps_override=None):
        args = self.args
        image_cond = video_latent_cond

        # Add action noise BEFORE normalization (simulates state estimation error)
        if action_noise > 0.0:
            noise = np.random.randn(*action_cond.shape) * action_noise
            action_cond = action_cond + noise
            
        # action should be normed
        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames+args.num_history, args.action_dim)

        # Use override inference steps if provided
        inference_steps = num_inference_steps_override if num_inference_steps_override is not None else args.num_inference_steps

        # predict future frames
        with torch.no_grad():
            bsz = action_cond.shape[0]
            if text is not None:
                text_token = self.model.action_encoder(action_cond, text, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)           
            pipeline = self.model.pipeline
            
            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height*3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
            )
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3,n=1) # (B, 8, 4, 32,32)
        
        # Add latent noise AFTER generation (direct quality degradation)
        if latent_noise > 0.0:
            noise = torch.randn_like(latents) * latent_noise
            latents = latents + noise


        # decode ground truth video
        true_video = torch.stack(video_latent_true, dim=0) # (bsz, 8,32,32)
        decoded_video = []
        bsz,frame_num = true_video.shape[:2]
        true_video = true_video.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,true_video.shape[0],args.decode_chunk_size):
            chunk = true_video[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        true_video = torch.cat(decoded_video,dim=0)
        true_video = true_video.reshape(bsz,frame_num,*true_video.shape[1:])
        true_video = ((true_video / 2.0 + 0.5).clamp(0, 1)*255)
        true_video = true_video.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(2,16,256,256,3)

        # decode predicted video
        decoded_video = []
        bsz,frame_num = latents.shape[:2]
        x = latents.flatten(0,1)
        decode_kwargs = {}
        for i in range(0,x.shape[0],args.decode_chunk_size):
            chunk = x[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video,dim=0)
        videos = videos.reshape(bsz,frame_num,*videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1)*255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8)

        # concatenate true videos and video
        videos_cat = np.concatenate([true_video,videos],axis=-3) # (3, 8, 256, 256, 3)
        videos_cat = np.concatenate([video for video in videos_cat],axis=-2).astype(np.uint8) 

        return videos_cat, true_video, videos, latents  # np.uint8:(3, 8, 128, 256, 3) or (3, 8, 192, 320, 3)

        
if __name__ == "__main__":
    from config import wm_args
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--svd_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--dataset_root_path', type=str, default=None)
    parser.add_argument('--dataset_meta_info_path', type=str, default=None)
    parser.add_argument('--dataset_names', type=str, default=None)
    parser.add_argument('--task_type', type=str, default='replay')
    parser.add_argument('--val_id', type=str, default=None, help='Trajectory ID to replay')
    parser.add_argument('--start_idx', type=int, default=None, help='Start frame index')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Dataset split (train or val)')
    parser.add_argument('--interact_num', type=int, default=None, help='Number of interactions (will auto-calculate for full video if not set)')
    # Noise injection parameters
    parser.add_argument('--action_noise', type=float, default=0.0, help='Gaussian noise std for action perturbation (0.0 = no noise)')
    parser.add_argument('--latent_noise', type=float, default=0.0, help='Gaussian noise std for latent space perturbation (0.0 = no noise)')
    parser.add_argument('--num_inference_steps', type=int, default=None, help='Override diffusion steps. Use -1 for video_length (full length per video)')
    parser.add_argument('--noise_config', type=str, default=None, help='Noise config name for output filename')
    parser.add_argument('--repeat_id', type=int, default=None, help='Repeat/sample ID for generating multiple samples per (task, noise)')
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None:
                args.__dict__[k] = v
        return args
    
    args = merge_args(args, args_new)

    # Override val_id, start_idx, and val_dataset_dir if provided via command line
    if args_new.val_id is not None:
        args.val_id = [args_new.val_id]
    if args_new.start_idx is not None:
        args.start_idx = [args_new.start_idx]
    if args_new.split is not None:
        # Update dataset directory to use the correct split
        base_dir = args.val_dataset_dir.split('/annotation')[0] if '/annotation' in args.val_dataset_dir else args.val_dataset_dir
        args.val_dataset_dir = f"{args.dataset_root_path}/{args.dataset_names}"
    
    # Ensure instruction list matches
    if hasattr(args, 'val_id'):
        args.instruction = [""] * len(args.val_id)

    # Store noise parameters
    action_noise = args_new.action_noise if args_new.action_noise is not None else 0.0
    latent_noise = args_new.latent_noise if args_new.latent_noise is not None else 0.0
    num_inference_steps_override = args_new.num_inference_steps  # Can be None
    noise_config = args_new.noise_config if args_new.noise_config else f"an{action_noise}_ln{latent_noise}_steps{num_inference_steps_override or 'default'}"
    
    print(f"Noise config: {noise_config}")
    print(f"  Action noise: {action_noise}")
    print(f"  Latent noise: {latent_noise}")
    print(f"  Inference steps: {num_inference_steps_override or 'default (50)'}")
    import os
    cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"  CUDA_VISIBLE_DEVICES={cuda_dev} (physical GPU index)")

    # create rollout agent
    Agent = agent(args)
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    print(f'rollout with {args.task_type}')
    
    # Get split from args (default to 'val' for backward compatibility)
    split = getattr(args, 'split', 'val')

    for val_id_i, text_i, start_idx_i in zip(args.val_id, args.instruction, args.start_idx):
        # Read annotation for video length (needed for interact_num and/or inference steps)
        annotation_path = f"{args.val_dataset_dir}/annotation/{split}/{val_id_i}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            video_length = anno.get("video_length", len(anno.get("states", [])))

        # Auto-calculate interact_num for full video if not specified
        if args_new.interact_num is not None:
            interact_num = args_new.interact_num
        else:
            # Formula: (interact_num * (pred_step - 1)) + pred_step >= video_length - start_idx_i
            remaining_frames = video_length - start_idx_i
            interact_num = max(1, int(np.ceil((remaining_frames - pred_step) / (pred_step - 1))) + 1)
            print(f"Auto-calculated interact_num={interact_num} for video_length={video_length}, start_idx={start_idx_i}")

        # Per-vide o inference steps: -1 means use video_length (entire video)
        if num_inference_steps_override == -1:
            steps_for_video = video_length
            print(f"Using inference_steps={steps_for_video} (video_length for full video)")
        else:
            steps_for_video = num_inference_steps_override
        
        # read ground truth trajectory informations
        eef_gt, joint_pos_gt, video_dict, video_latents, instruction = Agent.get_traj_info(val_id_i, start_idx=start_idx_i, steps=int(pred_step*interact_num+8), split=split)
        text_i = instruction
        print("text_i:",instruction, "eef pose at t=0", eef_gt[0], "joint at t=0", joint_pos_gt[0])

        # create buffers and push first frames to history buffer
        predicted_latents = None
        video_to_save = []
        info_to_save = []
        his_cond = []
        his_joint = []
        his_eef = []
        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
        assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
        for i in range(Agent.args.num_history*4):
            his_cond.append(first_latent)  # (1, 4, 72, 40)
            his_joint.append(joint_pos_gt[0:1])  # (1, 7)
            his_eef.append(eef_gt[0:1])  # (1, 7)

        # interact loop
        for i in range(interact_num):
            # ground truth video
            start_id = int(i*(pred_step-1))
            end_id = start_id + pred_step
            video_latent_true = [v[start_id:end_id] for v in video_latents]
            
            # prepare input for policy
            joint_first = his_joint[-1][0]
            state_first = his_eef[-1][0]
            if i==0:
                video_first = [v[0] for v in video_dict]
            else:
                video_first = [v[-1] for v in video_dict_pred]
            assert joint_first.shape == (8,), f"Expected joint_first shape (8,), got {joint_first.shape}"
            assert state_first.shape == (7,), f"Expected state_first shape (7,), got {state_first.shape}"
            
            # forward policy
            print("################ policy forward ####################")
            # in the trajectory replay model, we use action recorded in trajetcory
            cartesian_pose = eef_gt[start_id:end_id]  # (pred_step, 7)
            print("cartesian space action", cartesian_pose[0]) # output xyz and gripper for debug
            print("cartesian space action", cartesian_pose[-1]) # output xyz and gripper for debug
            
            print("################ world model forward ################")
            print(f'traj_id:{val_id_i}, interact step: {i}/{interact_num}')
            # retrive history cond and action cond
            history_idx = [0,0,-8,-6,-4,-2]
            his_pose = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)  # (4, 7)
            action_cond = np.concatenate([his_pose, cartesian_pose], axis=0)
            his_cond_input = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]  # (1, 4, 72, 40)
            assert current_latent.shape == (1, 4, 72, 40), f"Expected current_latent shape (1, 4, 72, 40), got {current_latent.shape}"
            assert action_cond.shape == (int(num_history+num_frames), 7), f"Expected action_cond shape ({int(num_history+num_frames)}, 7), got {action_cond.shape}"
            assert his_cond_input.shape == (1, int(num_history), 4, 72, 40), f"Expected his_cond_input shape (1, {int(num_history)}, 72, 40), got {his_cond_input.shape}"
            # forward world model
            videos_cat, true_videos, video_dict_pred, predicted_latents = Agent.forward_wm(
                action_cond, video_latent_true, current_latent, 
                his_cond=his_cond_input,
                text=text_i if Agent.args.text_cond else None,
                action_noise=action_noise,
                latent_noise=latent_noise,
                num_inference_steps_override=steps_for_video
            )

            print("################ record information ################")
            # push current step to history buffer
            his_eef.append(cartesian_pose[pred_step-1:pred_step]) #(1,7)
            his_cond.append(torch.cat([v[pred_step-1] for v in predicted_latents], dim=1).unsqueeze(0))  # (1, 4, 72, 40)
            if i == interact_num - 1:
                video_to_save.append(videos_cat)  # save all frames for the last interaction step
            else:
                video_to_save.append(videos_cat[:pred_step-1]) # last frame is the first frame of next step, so we remove it here
                
        
        # save rollout video and info with parameters
        video = np.concatenate(video_to_save, axis=0)
        task_name = args.task_name
        text_id = text_i.replace(' ', '_').replace(',', '').replace('.', '').replace('\'', '').replace('\"', '')[:30]
        videos_dir = args.val_model_path.split('/')[:-1]
        videos_dir = '/'.join(videos_dir)
        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include noise config and optional repeat_id in filename
        repeat_suffix = f"_rep{args_new.repeat_id}" if args_new.repeat_id is not None else ""
        filename_video = f"{args.save_dir}/{task_name}/video/{noise_config}/time_{uuid}_traj_{split}_{val_id_i}_{start_idx_i}_{text_id}{repeat_suffix}.mp4"
        os.makedirs(os.path.dirname(filename_video), exist_ok=True)
        if video.dtype in (np.float32, np.float64):
            video = (np.clip(video, 0, 1) * 255).astype(np.uint8)
        imageio.mimwrite(filename_video, video, fps=4, codec="libx264")
        print(f"Saving video to {filename_video}")
        print("##########################################################################")


# CUDA_VISIBLE_DEVICES=0 python rollout_replay_traj.py
        
        
