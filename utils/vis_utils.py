
import os
import textwrap
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import tqdm

from habitat.core.logging import logger
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
import cv2

'''
        top_down_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=info["top_down_map"]["agent_angle"],
            agent_radius_px=5,
        )
'''
def clip_map_birdseye_view(image, clip_size, pixel_pose):
    half_clip_size = clip_size//2

    delta_x = pixel_pose[0] - half_clip_size
    delta_y = pixel_pose[1] - half_clip_size
    min_x = max(delta_x, 0)
    max_x = min(pixel_pose[0] + half_clip_size, image.shape[0])
    min_y = max(delta_y, 0)
    max_y = min(pixel_pose[1] + half_clip_size, image.shape[1])

    return_image = np.zeros([clip_size, clip_size, 3],dtype=np.uint8)
    cliped_image = image[min_x:max_x, min_y:max_y]
    start_x = max(-delta_x,0)
    start_y = max(-delta_y,0)
    try:
        return_image[start_x:start_x+cliped_image.shape[0],start_y:start_y+cliped_image.shape[1]] = cliped_image
    except:
        print('image shape ', image.shape, 'min_x', min_x,'max_x', max_x,'min_y',min_y,'max_y',max_y, 'return_image.shape',return_image.shape, 'cliped', cliped_image.shape, 'start_x,y', start_x, start_y)
    return return_image

#python train_bc.py --run-type eval --gpu 7 --exp-config IL_configs/gmt.yaml --resume data/eval/GMT_fix_90.pt --version-name GMT_FIX
# python train_bc.py --run-type eval --exp-config IL_configs/gmt.yaml --gpu 8 --data-dir /disk4/obin/vistarget_demo_gibson/train/ --stop --version-name 0725_gmt_goalemb --diff hard --resume data/eval/GMT_goalemb_85.pt


def append_text_to_image(image: np.ndarray, text: str, font_size=0.5, font_line=cv2.LINE_AA):
    r""" Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)
    linetype = font_line if font_line is not None else cv2.LINE_8

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        if y % 2 == 1 :
            y += 1
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=linetype,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final
def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view


def observations_to_image(observation: Dict, info: Dict, mode='panoramic', local_imgs=None, clip=None, center_agent = True) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    if "rgb" in observation and mode != 'panoramic':
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)
    elif "panoramic_rgb" in observation and mode == 'panoramic':
        observation_size = observation["panoramic_rgb"].shape[0]
        rgb = observation["panoramic_rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)
        #egocentric_view.append((observation['panoramic_depth'].repeat(3,2)*255).astype(np.uint8))

    if "objectgoal" in observation:
        goal_rgb = (observation['objectgoal'][:,:,:3]*255)
        if not isinstance(goal_rgb, np.ndarray):
            goal_rgb = goal_rgb.cpu().numpy()
        egocentric_view.append(goal_rgb.astype(np.uint8))
    if "target_goal" in observation:
        goal_rgb = (observation['target_goal'][:,:,:3]*255)
        if not isinstance(goal_rgb, np.ndarray):
            goal_rgb = goal_rgb.cpu().numpy()
        egocentric_view.append(goal_rgb.astype(np.uint8))

    if local_imgs is not None:
        blank_img = np.zeros_like(rgb)
        small_imgs = []
        for i in range(4):
            if i >= len(local_imgs):
                small_shape = [int(blank_img.shape[0]/2), int(blank_img.shape[1]/2)]
                small_img = np.zeros([small_shape[0],small_shape[1],3])
            else:
                small_img = cv2.resize(local_imgs[i], dsize=None, fx=0.5, fy=0.5)
            small_imgs.append(small_img)
        small_img = np.concatenate([np.concatenate([small_imgs[0],small_imgs[1]],0),np.concatenate([small_imgs[2],small_imgs[3]],0)],1)
        egocentric_view.append(small_img.astype(np.uint8))

    if mode == 'panoramic':
        egocentric_view = np.concatenate(egocentric_view, axis=0)
    else:
        egocentric_view = np.concatenate(egocentric_view, axis=1)
    if "collisions" in info and info['collisions'] is not None:
        if info["collisions"]["is_collision"]:
            egocentric_view = draw_collision(egocentric_view)
    #if 'direction_img' in observation.keys():
    #    int_img = (observation['direction_img'] * 255).astype(np.uint8)
    #    egocentric_view = np.concatenate([int_img, egocentric_view],0)
    frame = egocentric_view


    top_down_height = frame.shape[0]
    if info is not None and "top_down_map" in info:
        if info['top_down_map'] is not None:
            #trg = observation['neighbors']
            #targetable = np.zeros([21, frame.shape[1], 3], dtype=np.uint8)
            #d = 21
            #for i in range(12):
            #    targetable[:, d * i:d * (i + 1)] = [0, 255 * trg[i], 0]
            #frame = np.concatenate([targetable, frame], 0)
            '''
            exp = info["top_down_map"]['unexplored']
            trg = observation['neighbors']
            delta = len(exp)
            kk = np.arange(delta).tolist()
            ss = kk[int(delta/2):] + kk[:int(delta/2)]
            targetable = np.zeros([22, frame.shape[1], 3],dtype=np.uint8)
            explorable = np.zeros([22, frame.shape[1], 3],dtype=np.uint8)
            exp,trg = exp[ss], trg[ss]
            d = int(frame.shape[1]/delta)
            for i in range(delta):
                explorable[:,d*i:d*(i+1)] = 255 * exp[i]
                targetable[:, d * i:d * (i + 1)] = [0, 255 * trg[i], 0]
            frame = np.concatenate([targetable, explorable,frame],0)
            '''
            top_down_height = frame.shape[0]
            top_down_map = info["top_down_map"]["map"]
            top_down_map = maps.colorize_topdown_map(
                top_down_map, info["top_down_map"]["fog_of_war_mask"]
            )
            map_agent_pos = info["top_down_map"]["agent_map_coord"]
            top_down_map = maps.draw_agent(
                image=top_down_map,
                agent_center_coord=map_agent_pos,
                agent_rotation=info["top_down_map"]["agent_angle"],
                agent_radius_px=5,
            )
            clip_size = 80#200 if local_imgs is None else 80
            if clip is not None: clip_size = clip
            if center_agent:
                top_down_map = clip_map_birdseye_view(top_down_map, clip_size, map_agent_pos)
            if top_down_map.shape[0] > top_down_map.shape[1]:
                top_down_map = np.rot90(top_down_map, 1)

            # scale top down map to align with rgb view
            old_h, old_w, _ = top_down_map.shape
            top_down_width = int(float(top_down_height) / old_h * old_w)
            # cv2 resize (dsize is width first)
            top_down_map = cv2.resize(
                top_down_map,
                (top_down_height, top_down_height),
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            top_down_map = np.zeros([top_down_height, top_down_height, 3],dtype=np.uint8)

        frame = np.concatenate((frame, top_down_map), axis=1)

    return frame
