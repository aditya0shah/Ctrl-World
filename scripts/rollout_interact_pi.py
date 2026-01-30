
from openpi.training import config as config_pi
from openpi.policies import policy_config
from openpi_client import image_tools
# from openpi.shared import download

import numpy as np


from accelerate import Accelerator
import torch
from diffusers import StableVideoDiffusionPipeline
import numpy as np
import cv2
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

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CrtlWorld
from models.utils import key_board_control, get_fk_solution
    

class agent():
    def __init__(self,args):
          
        # args = Args()
        args.val_model_path = args.ckpt_path
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # load pi policy
        if 'pi05' in args.policy_type:
            config = config_pi.get_config("pi05_droid")
            # checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid' 
        elif 'pi0fast' in args.policy_type:
            config = config_pi.get_config("pi0fast_droid")
            # checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0fast_droid'
        elif 'pi0' in args.policy_type:
            config = config_pi.get_config("pi0_droid")
            # checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0_droid'
        else:
            raise ValueError(f"Unknown policy type: {args.policy_type}")
        self.policy = policy_config.create_trained_policy(config, args.pi_ckpt)

        # load ctrl-world model

        self.model = CrtlWorld(args)
        self.model.load_state_dict(torch.load(args.val_model_path, map_location="cpu"))
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        
        # Optimize: Use torch.compile for PyTorch 2.0+ to speed up UNet inference
        # Note: Enable by setting compile_unet=True in args if desired (disabled by default for compatibility)
        if hasattr(torch, 'compile') and hasattr(args, 'compile_unet') and args.compile_unet:
            try:
                self.model.pipeline.unet = torch.compile(self.model.pipeline.unet, mode='reduce-overhead')
                print("UNet compiled with torch.compile for faster inference")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}. Continuing without compilation.")
        
        print("load world model success")
        with open(f"{args.data_stat_path}", 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None,:]
            self.state_p99 = np.array(data_stat['state_99'])[None,:]
        
        # Since the official Pi-Droid model output joint velocity, and crtl-world is train on cartesian space, we need to load an light-weight adapter to transform joint velocity action into cartesian pose action. 
        if args.action_adapter is not None:
            from models.action_adapter.train2 import Dynamics
            self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(self.device)
            self.dynamics_model.load_state_dict(torch.load(args.action_adapter, map_location=self.device))        

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

    def init_from_images(self, image_paths, joint_position, cartesian_position=None, gripper_position=None):
        """
        Initialize from initial images and joint state.
        
        Args:
            image_paths: List of paths to initial images (e.g., ['ext_1.png', 'ext_2.png', 'wrist.png'])
            joint_position: Initial joint positions (7 values)
            cartesian_position: Initial cartesian pose (6 values: xyz + euler) or None to compute from FK
            gripper_position: Initial gripper position (1 value) or None to use 0.0
        
        Returns:
            video_dict: List of initial video frames (numpy arrays)
            video_latents: List of encoded latents
            initial_eef: Initial end-effector pose (7 values: xyz + euler + gripper)
            initial_joint: Initial joint state (8 values: 7 joints + gripper)
        """
        import cv2
        
        # Load and encode images
        video_dict = []
        video_latent = []
        
        for img_path in image_paths:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not load image: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to expected dimensions (height x width from config)
            target_h, target_w = self.args.height, self.args.width
            img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            video_dict.append(img_resized[None, ...])  # Add time dimension: (1, H, W, 3)
            
            # Encode to latent
            device = self.device
            img_tensor = torch.from_numpy(img_resized).to(self.dtype).to(device)
            x = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device) / 255.0 * 2 - 1  # (1, 3, H, W)
            vae = self.model.pipeline.vae
            with torch.no_grad():
                latent = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
            video_latent.append(latent)
        
        # Compute initial end-effector pose if not provided
        if cartesian_position is None:
            from scipy.spatial.transform import Rotation as R
            current_state_fk = get_fk_solution(joint_position[:7])
            xyz = current_state_fk[:3, 3]
            rotation_matrix = current_state_fk[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz')
            initial_eef = np.concatenate([xyz, euler], axis=0)
        else:
            initial_eef = np.array(cartesian_position)
        
        # Add gripper position
        if gripper_position is None:
            gripper_pos = 0.0
        else:
            gripper_pos = gripper_position[0] if isinstance(gripper_position, (list, np.ndarray)) else gripper_position
        
        initial_eef = np.concatenate([initial_eef, [gripper_pos]])  # (7,)
        initial_joint = np.concatenate([joint_position[:7], [gripper_pos]])  # (8,)
        
        return video_dict, video_latent, initial_eef, initial_joint


    def get_traj_info(self, id, start_idx=0, steps=8,skip=1):
        val_dataset_dir = self.args.val_dataset_dir
        num_frames = steps
        annotation_path = f"{val_dataset_dir}/annotation/val/{id}.json"
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
        joint_pos = np.array(anno['joints'])
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

    def forward_wm(self, action_cond, video_latent_true, video_latent_cond, his_cond=None, text=None):
        # action_cond, video_latent_true, current_latent, his_cond=his_latent,text=text_i
        # Supports both single samples and batched inputs
        args = self.args
        image_cond = video_latent_cond

        # action should be normed
        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).to(self.device).to(self.dtype)
        
        # Handle single sample vs batch: ensure batch dimension exists
        if action_cond.ndim == 2:
            # Single sample: (num_frames+num_history, action_dim) -> (1, num_frames+num_history, action_dim)
            action_cond = action_cond.unsqueeze(0)
        
        # Handle image_cond: ensure batch dimension exists
        if image_cond.ndim == 3:
            # Single sample: (4, 72, 40) -> (1, 4, 72, 40)
            image_cond = image_cond.unsqueeze(0)
        
        # Handle his_cond: ensure batch dimension exists
        if his_cond is not None:
            if his_cond.ndim == 4:
                # Single sample: (num_history, 4, 72, 40) -> (1, num_history, 4, 72, 40)
                his_cond = his_cond.unsqueeze(0)
        
        # Verify shapes
        assert image_cond.shape[1:] == (4, 72, 40), f"Expected image_cond shape (B, 4, 72, 40), got {image_cond.shape}"
        assert action_cond.shape[1:] == (args.num_frames+args.num_history, args.action_dim), \
            f"Expected action_cond shape (B, {args.num_frames+args.num_history}, {args.action_dim}), got {action_cond.shape}"
        if his_cond is not None:
            assert his_cond.shape[2:] == (4, 72, 40), f"Expected his_cond shape (B, num_history, 4, 72, 40), got {his_cond.shape}"

        # Handle text: convert single string to list for batching
        batch_size = action_cond.shape[0]
        if text is not None:
            if isinstance(text, str):
                text = [text] * batch_size
            elif isinstance(text, list):
                assert len(text) == batch_size, f"Text list length {len(text)} doesn't match batch size {batch_size}"
            else:
                raise TypeError(f"Text must be str or list, got {type(text)}")

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
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
            )
        # Optimize: Use reshape + permute instead of einops for better performance
        # latents: (B, f, c, m*h, n*w) where m=3, n=1 -> (B*3, f, c, h, w)
        B, f, c, H_full, W = latents.shape
        h = H_full // 3  # Split height into 3 views
        # Reshape more efficiently: (B, f, c, 3*h, w) -> (B, 3, f, c, h, w) -> (B*3, f, c, h, w)
        latents = latents.view(B, f, c, 3, h, W).permute(0, 3, 1, 2, 4, 5).contiguous().view(B*3, f, c, h, W)
        
        # Split latents back into camera views: (B*3, f, c, h, w) -> list of 3 tensors
        # Optimize: Use view/reshape instead of slicing where possible
        original_batch_size = bsz
        num_views = 3
        latents_per_view = []
        for view_idx in range(num_views):
            # More efficient: reshape then select batch dimension
            view_latents = latents[view_idx::num_views]  # Extract every 3rd element starting from view_idx
            # Remove batch dimension for single batch to match expected format (matches video_latents format)
            if original_batch_size == 1:
                view_latents = view_latents.squeeze(0)  # (f, c, h, w) = (f, 4, 24, 40)
            latents_per_view.append(view_latents)


        # decode ground truth video (optional - only if provided)
        if video_latent_true is not None:
            # video_latent_true is a list of tensors (one per camera view), each (num_frames, channels, height, width)
            true_video = torch.stack(video_latent_true, dim=0) # (num_views, num_frames, channels, height, width)
            decoded_video = []
            num_views_gt, frame_num_gt = true_video.shape[:2]
            true_video = true_video.flatten(0,1)  # (num_views*frame_num, channels, height, width)
            decode_kwargs = {}
            for i in range(0,true_video.shape[0],args.decode_chunk_size):
                chunk = true_video[i:i+args.decode_chunk_size]/pipeline.vae.config.scaling_factor
                decode_kwargs["num_frames"] = chunk.shape[0]
                decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
            true_video = torch.cat(decoded_video,dim=0)
            true_video = true_video.reshape(num_views_gt, frame_num_gt, *true_video.shape[1:])
            true_video = ((true_video / 2.0 + 0.5).clamp(0, 1)*255)
            true_video = true_video.detach().to(torch.float32).cpu().numpy().transpose(0,1,3,4,2).astype(np.uint8) #(num_views, frame_num, height, width, channels)
        else:
            true_video = None

        # decode predicted video
        # Optimize: Use larger chunks when possible, keep on GPU longer
        decoded_video = []
        bsz_all_views, frame_num = latents.shape[:2]
        x = latents.flatten(0,1)
        
        # Optimize chunk size for better GPU utilization with larger batches
        # For larger batches, use larger chunks to amortize overhead
        effective_chunk_size = args.decode_chunk_size
        if bsz_all_views > 4:
            # Increase chunk size for larger batches to reduce loop overhead
            effective_chunk_size = min(args.decode_chunk_size * 2, x.shape[0])
        
        decode_kwargs = {}
        # Pre-scale all latents once instead of per chunk
        x_scaled = x / pipeline.vae.config.scaling_factor
        
        for i in range(0, x_scaled.shape[0], effective_chunk_size):
            chunk = x_scaled[i:i+effective_chunk_size]
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        
        # Concatenate on GPU, then do all conversions at once
        videos = torch.cat(decoded_video, dim=0)
        videos = videos.reshape(bsz_all_views, frame_num, *videos.shape[1:])
        # Keep operations on GPU: normalize and convert to uint8 range, then move to CPU once
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).round()
        videos = videos.detach().to(torch.uint8).cpu().numpy().transpose(0,1,3,4,2)
        
        # Split decoded videos back into camera views: (B*3, f, h, w, c) -> list of 3 arrays
        # For single batch: each is (f, h, w, c); for batched: each is (B, f, h, w, c)
        videos_per_view = []
        for view_idx in range(num_views):
            view_videos = videos[view_idx::num_views]  # Extract every 3rd element starting from view_idx
            # Remove batch dimension for single batch to match expected format
            if original_batch_size == 1:
                view_videos = view_videos.squeeze(0)  # (f, h, w, c)
            videos_per_view.append(view_videos)

        # concatenate true videos and video (if ground truth available)
        # Optimize: Reduce list comprehensions and use vectorized operations
        # For single batch (B=1), videos has shape (3, f, h, w, c) which matches original behavior
        # For batched (B>1), videos has shape (B*3, f, h, w, c) - reshape to (B, 3, f, h, w, c) for processing
        if original_batch_size == 1:
            # Single batch: use videos directly (already in shape (3, f, h, w, c))
            if true_video is not None:
                videos_cat = np.concatenate([true_video, videos], axis=-3) # Concatenate along frame dimension
                # Optimize: Use np.concatenate with axis instead of list comprehension
                videos_cat = np.concatenate(videos_cat, axis=-2).astype(np.uint8)
            else:
                # If no ground truth, just use predicted videos
                videos_cat = np.concatenate(videos, axis=-2).astype(np.uint8)
        else:
            # Multiple batches: reshape videos from (B*3, f, h, w, c) to (B, 3, f, h, w, c)
            videos_batched = videos.reshape(original_batch_size, num_views, frame_num, *videos.shape[2:])
            if true_video is not None:
                # true_video: (num_views, f, h, w, c) - broadcast to (B, num_views, f, h, w, c)
                true_video_batched = np.broadcast_to(true_video[None, ...], (original_batch_size, *true_video.shape))
                videos_cat = np.concatenate([true_video_batched, videos_batched], axis=2)  # (B, 3, 2*f, h, w, c)
                # Flatten for return: concatenate all batches and views
                # Optimize: Reshape then concatenate views in one operation
                B, num_v, F, H, W, C = videos_cat.shape
                videos_cat = videos_cat.reshape(B * num_v, F, H, W, C)
                videos_cat = np.concatenate(videos_cat, axis=-2).astype(np.uint8)
            else:
                # Optimize: Reshape and concatenate in one step
                B, num_v, F, H, W, C = videos_batched.shape
                videos_cat = videos_batched.reshape(B * num_v, F, H, W, C)
                videos_cat = np.concatenate(videos_cat, axis=-2).astype(np.uint8)

        return videos_cat, true_video, videos_per_view, latents_per_view  # Return latents as list per view

    def forward_policy(self, videos, state, joints, text, time_step=1):
        
        # inference policy
        image1 = videos[1]
        image2 = videos[2]
        image1 = torch.from_numpy(image1).to(torch.uint8)  # convert to torch tensor
        image2 = torch.from_numpy(image2).to(torch.uint8)  # convert to torch tensor
        assert image1.shape == (192, 320, 3), "Image 1 shape should be (192, 320, 3), got {}".format(image1.shape)
        image1 = torch.nn.functional.interpolate(image1.permute(2, 0, 1).unsqueeze(0).float(), size=(180, 320), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).to(torch.uint8)
        image2 = torch.nn.functional.interpolate(image2.permute(2, 0, 1).unsqueeze(0).float(), size=(180, 320), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).to(torch.uint8)
        image1 = image1.numpy()  # convert back to numpy array
        image2 = image2.numpy()  # convert back to numpy array
        example = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(image1, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(image2, 224, 224),
            "observation/joint_position": joints[:7],
            "observation/gripper_position": joints[-1:],
            "prompt": text,
        }
        action_chunk = self.policy.infer(example)["actions"] #(10,8) velocity

        # action adapater
        current_joint = joints[None,:][:,:7]
        current_gripper = joints[None,:][:,7:]
        if 'pi05' in self.args.policy_type:
            idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]  # for dynamics model, we need 15 steps
        else:
            idx = [0,1,2,3,4,5,6,7,8,9,9,9,9,9,9]
        # policy output joint velocity and gripper position
        joint_vel = action_chunk[:,:7] # (15, 7)
        gripper_pos = action_chunk[:,7:] # (15, 1)
        joint_vel = joint_vel[idx]  # (15, 7)
        gripper_pos = gripper_pos[idx]  # (15, 1)
        gripper_max = self.args.gripper_max
        gripper_pos = np.clip(gripper_pos, 0, gripper_max)
        # calculate future joint positions
        joint_pos = self.dynamics_model(current_joint, joint_vel,None, training=False)
        # fk
        state_fk = []
        joint_pos = np.concatenate([current_joint, joint_pos], axis=0)[:15]  # (15, 7)
        gripper_pos = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]  # (15, 1)
        joint_vel = joint_vel  # (15, 7)
        for i in range(joint_pos.shape[0]):
            current_state_fk = get_fk_solution(joint_pos[i,:7])
            xyz = current_state_fk[:3, 3]
            rotation_matrix = current_state_fk[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz') 
            state_fk.append(np.concatenate([xyz, euler, gripper_pos[i]], axis=0))
        state_fk = np.array(state_fk) # (15,7)

        # prepare output
        skip = self.args.policy_skip_step
        valid_num = int(skip*(self.args.pred_step-1))
        policy_in_out = {
            'joint_pos': joint_pos[:valid_num],  # (12, 7)
            'joint_vel': joint_vel[:valid_num],  # (12, 7)
            'state_fk': state_fk[:valid_num],  # (12, 7)
        }
        state_fk_skip = state_fk[::skip][:self.args.pred_step]  # (5, 7)
        joint_pos_skip = joint_pos[::skip][:self.args.pred_step]  # (5, 7)
        joint_pos_skip = np.concatenate([joint_pos_skip, state_fk_skip[:,-1:]], axis=-1) # (5, 8) add gripper pos

        return policy_in_out, joint_pos_skip, state_fk_skip

    def forward_policy_batch(self, videos_batch, states_batch, joints_batch, texts_batch, time_step=1):
        """
        Batched version of forward_policy that processes multiple trajectories at once.
        
        Args:
            videos_batch: List of video lists, one per trajectory. Each video list contains [ext1, ext2, wrist] images.
            states_batch: List of states, one per trajectory. Each state is (7,) array.
            joints_batch: List of joint positions, one per trajectory. Each is (8,) array.
            texts_batch: List of text prompts, one per trajectory. Can be single string (repeated) or list.
            time_step: Time step (unused but kept for compatibility)
        
        Returns:
            policy_outputs: List of policy_in_out dicts, one per trajectory
            joint_positions: List of joint_pos_skip arrays, one per trajectory
            cartesian_poses: List of state_fk_skip arrays, one per trajectory
        """
        batch_size = len(videos_batch)
        
        # Handle text: if single string, repeat for all trajectories
        if isinstance(texts_batch, str):
            texts_batch = [texts_batch] * batch_size
        assert len(texts_batch) == batch_size, f"Texts batch size {len(texts_batch)} doesn't match batch size {batch_size}"
        
        # Batch image preprocessing - this is where we get speedup
        image1_batch = []
        image2_batch = []
        for videos in videos_batch:
            image1 = videos[1]
            image2 = videos[2]
            image1_batch.append(image1)
            image2_batch.append(image2)
        
        # Stack images for batched processing
        image1_batch = np.stack(image1_batch, axis=0)  # (batch_size, 192, 320, 3)
        image2_batch = np.stack(image2_batch, axis=0)  # (batch_size, 192, 320, 3)
        
        # Convert to torch and batch interpolate
        image1_tensor = torch.from_numpy(image1_batch).to(torch.uint8)  # (batch_size, 192, 320, 3)
        image2_tensor = torch.from_numpy(image2_batch).to(torch.uint8)  # (batch_size, 192, 320, 3)
        
        # Batch interpolate: (batch_size, 192, 320, 3) -> (batch_size, 3, 192, 320) -> interpolate -> (batch_size, 3, 180, 320) -> (batch_size, 180, 320, 3)
        image1_tensor = image1_tensor.permute(0, 3, 1, 2).float()  # (batch_size, 3, 192, 320)
        image2_tensor = image2_tensor.permute(0, 3, 1, 2).float()  # (batch_size, 3, 192, 320)
        image1_tensor = torch.nn.functional.interpolate(image1_tensor, size=(180, 320), mode='bilinear', align_corners=False)
        image2_tensor = torch.nn.functional.interpolate(image2_tensor, size=(180, 320), mode='bilinear', align_corners=False)
        image1_tensor = image1_tensor.permute(0, 2, 3, 1).to(torch.uint8)  # (batch_size, 180, 320, 3)
        image2_tensor = image2_tensor.permute(0, 2, 3, 1).to(torch.uint8)  # (batch_size, 180, 320, 3)
        image1_batch = image1_tensor.numpy()  # (batch_size, 180, 320, 3)
        image2_batch = image2_tensor.numpy()  # (batch_size, 180, 320, 3)
        
        # Batch resize_with_pad - process each image individually (image_tools.resize_with_pad doesn't support batching)
        image1_resized = []
        image2_resized = []
        for i in range(batch_size):
            image1_resized.append(image_tools.resize_with_pad(image1_batch[i], 224, 224))
            image2_resized.append(image_tools.resize_with_pad(image2_batch[i], 224, 224))
        
        # Stack joint positions and gripper positions
        joints_array = np.stack(joints_batch, axis=0)  # (batch_size, 8)
        joint_positions_batch = joints_array[:, :7]  # (batch_size, 7)
        gripper_positions_batch = joints_array[:, 7:8]  # (batch_size, 1)
        
        # Create batched example dict for policy inference
        # Use infer_batch for true batched inference (if available), otherwise fall back to loop
        if hasattr(self.policy, 'infer_batch'):
            # Prepare batched observation dict
            batched_example = {
                "observation/exterior_image_1_left": np.stack(image1_resized, axis=0),  # (batch_size, 224, 224, 3)
                "observation/wrist_image_left": np.stack(image2_resized, axis=0),  # (batch_size, 224, 224, 3)
                "observation/joint_position": joint_positions_batch,  # (batch_size, 7)
                "observation/gripper_position": gripper_positions_batch,  # (batch_size, 1)
                "prompt": texts_batch,  # List of strings
            }
            # Call batched inference
            batched_results = self.policy.infer_batch(batched_example)
            action_chunks = batched_results["actions"]  # (batch_size, action_horizon, 8)
        else:
            # Fallback to individual inference if infer_batch not available
            action_chunks = []
            for i in range(batch_size):
                example = {
                    "observation/exterior_image_1_left": image1_resized[i],
                    "observation/wrist_image_left": image2_resized[i],
                    "observation/joint_position": joint_positions_batch[i],
                    "observation/gripper_position": gripper_positions_batch[i],
                    "prompt": texts_batch[i],
                }
                action_chunk = self.policy.infer(example)["actions"]  # (10, 8) or (15, 8) velocity
                action_chunks.append(action_chunk)
            action_chunks = np.stack(action_chunks, axis=0)  # (batch_size, 10, 8) or (batch_size, 15, 8)
        
        # Process action adapter and FK for each trajectory
        policy_outputs = []
        joint_positions = []
        cartesian_poses = []
        
        if 'pi05' in self.args.policy_type:
            idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]  # for dynamics model, we need 15 steps
        else:
            idx = [0,1,2,3,4,5,6,7,8,9,9,9,9,9,9]
        
        gripper_max = self.args.gripper_max
        skip = self.args.policy_skip_step
        valid_num = int(skip*(self.args.pred_step-1))
        
        for i in range(batch_size):
            # Handle both batched array and list formats
            if isinstance(action_chunks, np.ndarray):
                action_chunk = action_chunks[i]  # (action_horizon, 8)
            else:
                action_chunk = action_chunks[i]  # (action_horizon, 8)
            
            # Check action_chunk shape and adjust indices if needed
            action_horizon = action_chunk.shape[0]
            if action_horizon < max(idx) + 1:
                # If action horizon is shorter than expected, adjust indices
                # For indices beyond the horizon, repeat the last valid index
                idx_adjusted = [min(j, action_horizon - 1) for j in idx]
            else:
                idx_adjusted = idx
            
            current_joint = joints_batch[i][None, :][:, :7]  # (1, 7)
            current_gripper = joints_batch[i][None, :][:, 7:]  # (1, 1)
            
            # Policy output joint velocity and gripper position
            joint_vel = action_chunk[:, :7]  # (action_horizon, 7)
            gripper_pos = action_chunk[:, 7:]  # (action_horizon, 1)
            joint_vel = joint_vel[idx_adjusted]  # (15, 7)
            gripper_pos = gripper_pos[idx_adjusted]  # (15, 1)
            gripper_pos = np.clip(gripper_pos, 0, gripper_max)
            
            # Calculate future joint positions
            joint_pos = self.dynamics_model(current_joint, joint_vel, None, training=False)
            
            # FK
            state_fk = []
            joint_pos = np.concatenate([current_joint, joint_pos], axis=0)[:15]  # (15, 7)
            gripper_pos = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]  # (15, 1)
            for j in range(joint_pos.shape[0]):
                current_state_fk = get_fk_solution(joint_pos[j, :7])
                xyz = current_state_fk[:3, 3]
                rotation_matrix = current_state_fk[:3, :3]
                r = R.from_matrix(rotation_matrix)
                euler = r.as_euler('xyz')
                state_fk.append(np.concatenate([xyz, euler, gripper_pos[j]], axis=0))
            state_fk = np.array(state_fk)  # (15, 7)
            
            # Prepare output
            policy_in_out = {
                'joint_pos': joint_pos[:valid_num],  # (12, 7)
                'joint_vel': joint_vel[:valid_num],  # (12, 7)
                'state_fk': state_fk[:valid_num],  # (12, 7)
            }
            state_fk_skip = state_fk[::skip][:self.args.pred_step]  # (5, 7)
            joint_pos_skip = joint_pos[::skip][:self.args.pred_step]  # (5, 7)
            joint_pos_skip = np.concatenate([joint_pos_skip, state_fk_skip[:, -1:]], axis=-1)  # (5, 8) add gripper pos
            
            policy_outputs.append(policy_in_out)
            joint_positions.append(joint_pos_skip)
            cartesian_poses.append(state_fk_skip)
        
        return policy_outputs, joint_positions, cartesian_poses

    
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
    parser.add_argument('--task_type', type=str, default=None)
    parser.add_argument('--pi_ckpt', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1, help='Number of parallel trajectories to run from same initial state')
    parser.add_argument('--trajectory_index', type=int, default=None, help='Index of this trajectory when running y parallel trajectories (0-based)')
    parser.add_argument('--num_trajectories', type=int, default=None, help='Total number of parallel trajectories (y) when using multi-GPU rollout')
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)

    def merge_args(cfg, cli_args):
        for k, v in vars(cli_args).items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    args = merge_args(args, args_new)

    # Expand (val_id, instruction, start_idx) when num_trajectories > n scenes, then optionally slice to trajectory_index
    n = len(args.val_id)
    if getattr(args, 'num_trajectories', None) is not None and args.num_trajectories > n:
        rep = (args.num_trajectories + n - 1) // n
        args.val_id = (args.val_id * rep)[:args.num_trajectories]
        args.instruction = (args.instruction * rep)[:args.num_trajectories]
        args.start_idx = (args.start_idx * rep)[:args.num_trajectories]
    if getattr(args, 'trajectory_index', None) is not None:
        i = args.trajectory_index
        assert 0 <= i < len(args.val_id), f"trajectory_index {i} out of range [0, {len(args.val_id)})"
        args.val_id = [args.val_id[i]]
        args.instruction = [args.instruction[i]]
        args.start_idx = [args.start_idx[i]]
    
    # Debug: print the pi_ckpt value being used
    print(f"Using pi_ckpt: {args.pi_ckpt}")

    # create agent
    Agent = agent(args)
    interact_num = args.interact_num
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    history_idx = args.history_idx
    batch_size = args.batch_size  # Number of parallel trajectories

    # run len(val_id) trajectory
    for val_id_i, text_i, start_idx_i in zip(args.val_id, args.instruction, args.start_idx):

        # Check if using image-based initialization
        use_image_init = hasattr(args, 'init_from_images') and args.init_from_images
        
        if use_image_init:
            # Initialize from images directly
            scene_dir = os.path.join(args.val_dataset_dir, val_id_i)
            metadata_path = os.path.join(scene_dir, "metadata.json")
            
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Get image paths (assuming standard naming: ext_1.png, ext_2.png, wrist.png)
            image_paths = [
                os.path.join(scene_dir, "ext_1.png"),
                os.path.join(scene_dir, "ext_2.png"), 
                os.path.join(scene_dir, "wrist.png")
            ]
            
            # Check if images exist, if not try alternative names
            if not all(os.path.exists(p) for p in image_paths):
                # Try to find images in the directory
                import glob
                png_files = glob.glob(os.path.join(scene_dir, "*.png"))
                if len(png_files) >= 3:
                    image_paths = sorted(png_files)[:3]
                    print(f"Using images: {image_paths}")
                else:
                    raise FileNotFoundError(f"Need at least 3 images in {scene_dir}, found {len(png_files)}: {png_files}")
            
            joint_pos = np.array(metadata['joint_position'])
            cartesian_pos = metadata.get('cartesian_position', None)
            gripper_pos = metadata.get('gripper_position', None)
            
            video_dict, video_latents, initial_eef, initial_joint = Agent.init_from_images(
                image_paths, joint_pos, cartesian_pos, gripper_pos
            )
            
            # Initialize history buffers - separate lists for each trajectory
            video_to_save_batch = [[] for _ in range(batch_size)]
            info_to_save_batch = [[] for _ in range(batch_size)]
            his_cond_batch = [[] for _ in range(batch_size)]  # List of lists - one per trajectory
            his_joint_batch = [[] for _ in range(batch_size)]  # List of lists - one per trajectory
            his_eef_batch = [[] for _ in range(batch_size)]  # List of lists - one per trajectory
            first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
            assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
            # Initialize same initial state for all trajectories
            # Convert numpy arrays to tensors for .clone() method
            initial_joint_single = torch.from_numpy(initial_joint[None, :]).to(Agent.device).to(Agent.dtype)  # (1, 8)
            initial_eef_single = torch.from_numpy(initial_eef[None, :]).to(Agent.device).to(Agent.dtype)  # (1, 7)
            for traj_idx in range(batch_size):
                for i in range(Agent.args.num_history*4):
                    his_cond_batch[traj_idx].append(first_latent.clone())  # (1, 4, 72, 40)
                    his_joint_batch[traj_idx].append(initial_joint_single.clone())  # (1, 8)
                    his_eef_batch[traj_idx].append(initial_eef_single.clone())  # (1, 7)
            video_dict_pred_batch = [[v[0:1] for v in video_dict] for _ in range(batch_size)]
            
            print(f"text_i: {text_i}, eef pose at t=0: {initial_eef}, joint at t=0: {initial_joint}")
            print(f"Initialized from images: {image_paths}")
            print(f"Running {batch_size} parallel trajectories from same initial state")

            # No ground truth for comparison
            video_latents_gt = None
            
        else:
            # Original dataset-based initialization
            # get initial state and groud truth
            id = val_id_i
            eef_gt, joint_pos_gt, video_dict, video_latents,_ = Agent.get_traj_info(val_id_i, start_idx=start_idx_i, steps=int(pred_step*interact_num+8))
            print("text_i:",text_i, "eef pose at t=0", eef_gt[0], "joint at t=0", joint_pos_gt[0])

            # initialize all history buffer - separate lists for each trajectory
            video_to_save_batch = [[] for _ in range(batch_size)]
            info_to_save_batch = [[] for _ in range(batch_size)] 
            his_cond_batch = [[] for _ in range(batch_size)]  # List of lists - one per trajectory
            his_joint_batch = [[] for _ in range(batch_size)]  # List of lists - one per trajectory
            his_eef_batch = [[] for _ in range(batch_size)]  # List of lists - one per trajectory
            first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
            assert first_latent.shape == (1, 4, 72, 40), f"Expected first_latent shape (1, 4, 72, 40), got {first_latent.shape}"
            # Initialize same initial state for all trajectories
            # Convert numpy arrays to tensors for .clone() method
            initial_joint_single = torch.from_numpy(joint_pos_gt[0:1]).to(Agent.device).to(Agent.dtype)  # (1, 8)
            initial_eef_single = torch.from_numpy(eef_gt[0:1]).to(Agent.device).to(Agent.dtype)  # (1, 7)
            for traj_idx in range(batch_size):
                for i in range(Agent.args.num_history*4):
                    his_cond_batch[traj_idx].append(first_latent.clone())  # (1, 4, 72, 40)
                    his_joint_batch[traj_idx].append(initial_joint_single.clone())  # (1, 8)
                    his_eef_batch[traj_idx].append(initial_eef_single.clone())  # (1, 7)
            video_dict_pred_batch = [[v[0:1] for v in video_dict] for _ in range(batch_size)]
            video_latents_gt = video_latents
            print(f"Running {batch_size} parallel trajectories from same initial state")


        # start rollout - batched version
        for i in range(interact_num):
            print(f"################ Step {i+1}/{interact_num} - Batched Rollout (batch_size={batch_size}) ################")
            
            # Collect inputs from all trajectories for batched policy and world model forward pass
            # Prepare batched inputs for policy
            videos_batch = []
            states_batch = []
            joints_batch = []
            
            for traj_idx in range(batch_size):
                # Get current state for this trajectory
                current_joint = his_joint_batch[traj_idx][-1][0]  # (8,)
                current_pose = his_eef_batch[traj_idx][-1][0]  # (7,)
                # Convert tensors to numpy arrays if needed (forward_policy_batch expects numpy arrays)
                if isinstance(current_joint, torch.Tensor):
                    current_joint = current_joint.cpu().float().numpy()
                if isinstance(current_pose, torch.Tensor):
                    current_pose = current_pose.cpu().float().numpy()
                # Extract last frame from each video view
                # Handle both (1, H, W, 3) and (num_frames, h, w, c) shapes
                current_obs = []
                for v in video_dict_pred_batch[traj_idx]:
                    if len(v.shape) == 4:
                        # Shape is (num_frames, h, w, c) or (1, H, W, 3)
                        # Use v[0] if first dim is 1, otherwise v[-1]
                        current_obs.append(v[0] if v.shape[0] == 1 else v[-1])
                    else:
                        # Already 3D: (h, w, c)
                        current_obs.append(v)
                
                videos_batch.append(current_obs)
                states_batch.append(current_pose)
                joints_batch.append(current_joint)
            
            # BATCHED policy forward pass - processes all trajectories at once
            print(f"################ Batched policy forward (batch_size={batch_size}) ################")
            policy_outputs, joint_positions, cartesian_poses = Agent.forward_policy_batch(
                videos_batch, states_batch, joints_batch, text_i
            )
            
            # Prepare world model inputs for each trajectory (after policy divergence)
            action_conds = []
            current_latents = []
            his_latents = []
            history_idx = args.history_idx
            
            for traj_idx in range(batch_size):
                # Convert tensors to numpy arrays if needed for numpy operations
                his_pose_elems = []
                for idx in history_idx:
                    elem = his_eef_batch[traj_idx][idx][0]  # (7,)
                    if isinstance(elem, torch.Tensor):
                        elem = elem.cpu().float().numpy()
                    # Ensure element is 2D: (1, 7)
                    if elem.ndim == 1:
                        elem = elem[None, :]  # (1, 7)
                    his_pose_elems.append(elem)
                his_pose = np.concatenate(his_pose_elems, axis=0)  # (num_history, 7)
                action_cond = np.concatenate([his_pose, cartesian_poses[traj_idx]], axis=0)  # (num_history+num_frames, 7)
                his_latent = torch.cat([his_cond_batch[traj_idx][idx] for idx in history_idx], dim=0).unsqueeze(0)  # (1, num_history, 4, 72, 40)
                current_latent = his_cond_batch[traj_idx][-1]  # (1, 4, 72, 40)
                
                action_conds.append(action_cond)
                current_latents.append(current_latent)
                his_latents.append(his_latent)
            
            # Stack into batches for world model
            action_cond_batch = np.stack(action_conds, axis=0)  # (batch_size, num_history+num_frames, 7)
            current_latent_batch = torch.cat(current_latents, dim=0)  # (batch_size, 4, 72, 40)
            his_latent_batch = torch.cat(his_latents, dim=0)  # (batch_size, num_history, 4, 72, 40)
            text_batch = [text_i if Agent.args.text_cond else None] * batch_size
            
            print(f"################ Batched world model forward (batch_size={batch_size}) ################")
            # BATCHED world model forward pass - this is where we get speedup!
            videos_cat_batch, true_videos, video_dict_pred_batch_output, predict_latents = Agent.forward_wm(
                action_cond_batch,
                None,  # No ground truth for batched rollouts
                current_latent_batch,
                his_cond=his_latent_batch,
                text=text_batch
            )
            
            print("################ Update history buffers for each trajectory ################")
            # Split outputs and update history buffers for each trajectory
            for traj_idx in range(batch_size):
                # Extract this trajectory's latents
                # predict_latents is a list of 3 tensors (one per camera view)
                # For batch_size > 1, each tensor has shape (batch_size, num_frames, 4, 24, 40)
                # For batch_size = 1, each tensor has shape (num_frames, 4, 24, 40)
                if batch_size == 1:
                    traj_latents = predict_latents  # Already single sample format
                else:
                    traj_latents = [v[traj_idx:traj_idx+1] for v in predict_latents]  # Each: (1, num_frames, 4, 24, 40)
                
                # Update history buffers for this trajectory
                # joint_positions[traj_idx] has shape (pred_step, 8), extract the last step
                joint_update = joint_positions[traj_idx][pred_step-1:pred_step]  # (1, 8)
                his_joint_batch[traj_idx].append(joint_update)  # Append to this trajectory's history
                
                cartesian_update = cartesian_poses[traj_idx][pred_step-1:pred_step]  # (1, 7)
                his_eef_batch[traj_idx].append(cartesian_update)  # Append to this trajectory's history
                
                # Combine latents from all camera views for this trajectory
                if batch_size == 1:
                    # traj_latents is list of (num_frames, 4, 24, 40)
                    combined_latent = torch.cat([v[pred_step-1:pred_step] for v in traj_latents], dim=2)  # (1, 4, 72, 40)
                else:
                    # traj_latents is list of (1, num_frames, 4, 24, 40)
                    combined_latent = torch.cat([v[0, pred_step-1:pred_step] for v in traj_latents], dim=2)  # (1, 4, 72, 40)
                
                his_cond_batch[traj_idx].append(combined_latent)  # Append to this trajectory's history
                
                # Update video predictions for this trajectory
                # video_dict_pred_batch_output is a list of arrays (one per camera view)
                # Each array has shape (batch_size, num_frames, h, w, c) for batched or (num_frames, h, w, c) for single
                if batch_size == 1:
                    video_dict_pred_batch[traj_idx] = video_dict_pred_batch_output
                    # videos_cat_batch is already in single trajectory format
                    video_to_save_batch[traj_idx].append(videos_cat_batch[:pred_step-1])
                else:
                    # Extract this trajectory's video predictions from batched output
                    # video_dict_pred_batch_output is list of (batch_size, num_frames, h, w, c) arrays
                    video_dict_pred_batch[traj_idx] = [v[traj_idx] for v in video_dict_pred_batch_output]  # Each: (num_frames, h, w, c)
                    # Extract this trajectory's videos from each view and concatenate
                    traj_videos_per_view = [v[traj_idx, :pred_step-1] for v in video_dict_pred_batch_output]  # Each: (pred_step-1, h, w, c)
                    # Concatenate views horizontally for this trajectory (same as original format)
                    traj_video_cat = np.concatenate(traj_videos_per_view, axis=-2)  # (pred_step-1, h, 3*w, c)
                    video_to_save_batch[traj_idx].append(traj_video_cat)
                
                info_to_save_batch[traj_idx].append(policy_outputs[traj_idx])
            

        # save rollout videos and info for each trajectory in batch
        print("##########################################################################")
        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        text_id = text_i.replace(' ', '_').replace(',', '').replace('.', '').replace('\'', '').replace('\"', '')[:40]
        
        for traj_idx in range(batch_size):
            video = np.concatenate(video_to_save_batch[traj_idx], axis=0)
            run_id = args.trajectory_index if getattr(args, 'trajectory_index', None) is not None else traj_idx
            suffix = f"run{run_id}" if getattr(args, 'trajectory_index', None) is not None else f"batch{traj_idx}"
            filename_video = f"{args.save_dir}/{args.task_name}/video/{args.task_type}_time_{uuid}_traj_{val_id_i}_{start_idx_i}_{args.policy_skip_step}_{text_id}_{suffix}.mp4"
            os.makedirs(os.path.dirname(filename_video), exist_ok=True)
            # Use imageio + imageio-ffmpeg (bundled ffmpeg, no system install needed)
            if video.dtype in (np.float32, np.float64):
                video = (np.clip(video, 0, 1) * 255).astype(np.uint8)
            imageio.mimwrite(filename_video, video, fps=4, codec="libx264")
            print(f"Saving video for trajectory {traj_idx} to {filename_video}")
            
            info = {'success': 1, 'start_idx': 0, 'end_idx': video.shape[0]-1, 'instructions': text_i, 'batch_idx': traj_idx}
            if len(info_to_save_batch[traj_idx]) > 0:
                for key in info_to_save_batch[traj_idx][0].keys():
                    info[key] = []
                    for i in range(len(info_to_save_batch[traj_idx])):
                        info[key] += info_to_save_batch[traj_idx][i][key].tolist()
            
            # save to json
            filename_info = f"{args.save_dir}/{args.task_name}/info/{args.task_type}_time_{uuid}_traj_{val_id_i}_{start_idx_i}_{pred_step}_{text_id}_{suffix}.json"
            os.makedirs(os.path.dirname(filename_info), exist_ok=True)
            with open(filename_info, 'w') as f:
                json.dump(info, f, indent=4)
            print(f"Saving trajectory info for trajectory {traj_idx} to {filename_info}")
        print("##########################################################################")


# CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python rollout_interact_pi.py --task_type pickplace
        
        
