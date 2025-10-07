from math import log
from loguru import logger

import torch
from einops import repeat
from kornia.utils import create_meshgrid
from kornia.geometry.linalg import transform_points

from .geometry import warp_kpts

##############  ↓  Coarse-Level supervision  ↓  ##############


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, "n h w -> n (h w) c", c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }

    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data["image0"].device
    N, _, H0, W0 = data["image0"].shape
    _, _, H1, W1 = data["image1"].shape
    scale = config["EDM"]["LOCAL_RESOLUTION"]

    scale0 = scale * data["scale0"][:, None] if "scale0" in data else scale
    scale1 = scale * data["scale1"][:, None] if "scale0" in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = (
        create_meshgrid(h0, w0, False, device).reshape(
            1, h0 * w0, 2).repeat(N, 1, 1)
    )  # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = (
        create_meshgrid(h1, w1, False, device).reshape(
            1, h1 * w1, 2).repeat(N, 1, 1)
    )
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if "mask0" in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data["mask0"])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data["mask1"])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (no depth consistency check, since it leads to worse results experimentally)
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts(
        grid_pt0_i,
        data["depth0"],
        data["depth1"],
        data["T_0to1"],
        data["K0"],
        data["K1"],
    )
    _, w_pt1_i = warp_kpts(
        grid_pt1_i,
        data["depth1"],
        data["depth0"],
        data["T_1to0"],
        data["K1"],
        data["K0"],
    )
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. check if mutual nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary

    def out_bound_mask(pt, w, h):
        return (
            (pt[..., 0] < 0) + (pt[..., 0] >= w) +
            (pt[..., 1] < 0) + (pt[..., 1] >= h)
        )

    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    loop_back = torch.stack(
        [nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0
    )
    correct_0to1 = loop_back == torch.arange(
        h0 * w0, device=device)[None].repeat(N, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0 * w0, h1 * w1, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({"conf_matrix_gt": conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(
            f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({"spv_b_ids": b_ids, "spv_i_ids": i_ids, "spv_j_ids": j_ids})

    # 6. save intermediate results (for fast fine-level computation)
    data.update(
        {
            "spv_w_pt0_i": w_pt0_i,
            "spv_pt1_i": grid_pt1_i,
            "spv_w_pt1_i": w_pt1_i,
            "spv_pt0_i": grid_pt0_i,
        }
    )


def compute_supervision_coarse(data, config):
    assert (
        len(set(data["dataset_name"])) == 1
    ), "Do not support mixed datasets training!"
    data_source = data["dataset_name"][0]
    if data_source.lower() in ["scannet", "megadepth"]:
        spvs_coarse(data, config)
    elif data_source.lower() == "synthetichomography":
        spvs_coarse_homography(data, config)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


##############  ↓  Fine-Level supervision  ↓  ##############
@torch.no_grad()
def spvs_fine(data, pad=0, window_size=8.0):
    """
    Update:
        data (dict):{
            "expec_f_gt": [M, 2]}
    """
    # 1. get coarse prediction
    w_pt0_i, pt1_i = data["spv_w_pt0_i"], data["spv_pt1_i"]
    w_pt1_i, pt0_i = data["spv_w_pt1_i"], data["spv_pt0_i"]
    b_ids, i_ids, j_ids, mconf = (
        data["b_ids"].clone(),
        data["i_ids"].clone(),
        data["j_ids"].clone(),
        data["mconf"].clone(),
    )

    scale0 = data["scale0"][b_ids] if "scale0" in data else 1.0
    scale1 = data["scale1"][b_ids] if "scale0" in data else 1.0

    # 2. random sampling of training samples for fine-level supervision
    # (optional) In the early stages of training
    # the coarse matching predictions are not good enough
    # pad samples with gt coarse-level matches
    if pad > 0:
        bs = int(b_ids.max()) + 1
        k = b_ids.shape[0] // bs
        offset01_gt = (
            w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]
        ) / scale1  # [M, 2] [NK, 2]
        offset01_gt_mask = (offset01_gt[:, 0].abs() < (window_size // 2)) & (
            offset01_gt[:, 1].abs() < (window_size // 2)
        )
        spv_b_ids, spv_i_ids, spv_j_ids = (
            data["spv_b_ids"].clone(),
            data["spv_i_ids"].clone(),
            data["spv_j_ids"].clone(),
        )

        # each pair in a batch
        for bi in range(bs):
            b_mask = b_ids == bi
            spv_b_mask = spv_b_ids == bi
            pad_num = 0
            max_pad_num = pad
            pred_true_num = offset01_gt_mask[b_mask].sum()

            # Only pad when the number of coarse matches is insufficient
            if pred_true_num < pad:
                max_pad_num = pad - pred_true_num
            else:
                continue

            candi_spv_i_ids = spv_i_ids[spv_b_mask]
            candi_spv_j_ids = spv_j_ids[spv_b_mask]
            shuffle_indices = torch.randperm(spv_b_mask.sum())
            candi_spv_i_ids = candi_spv_i_ids[shuffle_indices]
            candi_spv_j_ids = candi_spv_j_ids[shuffle_indices]

            max_pad_num = min(int(spv_b_mask.sum()), max_pad_num)
            for i in range(k):
                if pad_num < max_pad_num:
                    if offset01_gt_mask[b_mask].tolist()[i] is False:
                        i_ids[bi * k + i] = candi_spv_i_ids[pad_num]
                        j_ids[bi * k + i] = candi_spv_j_ids[pad_num]
                        # Just assisting fine-level training
                        mconf[bi * k + i] = 0.0
                        pad_num += 1
                else:
                    break

        # update data
        data.update(
            {
                "b_ids": b_ids,
                "i_ids": i_ids,
                "j_ids": j_ids,
                "mconf": mconf,
            }
        )

    # 3. compute fine-level offset gt
    offset01_gt = (
        w_pt0_i[b_ids, i_ids] - pt1_i[b_ids, j_ids]
    ) / scale1  # [M, 2] [NK, 2]
    offset10_gt = (
        w_pt1_i[b_ids, j_ids] - pt0_i[b_ids, i_ids]
    ) / scale0  # [M, 2] [NK, 2]
    data.update({"offset01_gt": offset01_gt})
    data.update({"offset10_gt": offset10_gt})

    target_uv = torch.cat([offset01_gt, offset10_gt])  # [2NK, 2] range [-4, 4]
    pos_mask = (target_uv[:, 0].abs() < (window_size // 2)) & (
        target_uv[:, 1].abs() < (window_size // 2)
    )
    # [M, 2] range [-0.5, 0.5]
    data.update({"target_uv": target_uv / window_size})
    data.update({"target_uv_weight": pos_mask})  # [M, 1]


def compute_supervision_fine(data, pad=0, window_size=8.0):
    spvs_fine(data, pad, window_size)


# TODO: check if fine level loss is fine
# TODO: check if this funcition makes sense
@torch.no_grad()
def spvs_coarse_homography(data, config):
    """
    Generate coarse-level supervision for a pair of images related by a homography.
    """
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    scale = config['EDM']['LOCAL_RESOLUTION']
    h0, w0 = H0 // scale, W0 // scale
    
    # 1. Create a grid of points in the coarse resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1) # [N, hw, 2]

    # 2. Scale coarse points to original image resolution for homography transformation
    # The homography is defined on the full resolution images (W, H)
    grid_pt0_i = grid_pt0_c * scale

    # 3. Warp points using the homography
    # kornia.transform_points expects homography of shape (N, 3, 3)
    homography = data['homography'].to(device)
    w_pt0_i = transform_points(homography, grid_pt0_i) # [N, hw, 2]

    # 4. Scale warped points back to coarse resolution
    w_pt0_c = w_pt0_i / scale
    
    # 5. Find nearest grid point for each warped point to establish correspondence
    w_pt0_c_round = w_pt0_c.round().long()
    
    # Create masks for points that are warped outside the image boundaries
    # These are invalid correspondences.
    in_bound_mask = (w_pt0_c_round[..., 0] >= 0) & (w_pt0_c_round[..., 0] < w0) & \
                    (w_pt0_c_round[..., 1] >= 0) & (w_pt0_c_round[..., 1] < h0)

    # Get the 1D index for the target points
    j_ids_gt = w_pt0_c_round[..., 1] * w0 + w_pt0_c_round[..., 0] # [N, hw0]

    # 6. Construct the ground-truth confidence matrix
    conf_matrix_gt = torch.zeros(N, h0 * w0, h0 * w0, device=device)
    
    # Get batch and source point indices for valid correspondences
    b_ids, i_ids = torch.where(in_bound_mask)
    # Get the corresponding target point indices
    j_ids = j_ids_gt[b_ids, i_ids]

    conf_matrix_gt[b_ids, i_ids, j_ids] = 1
    data.update({'conf_matrix_gt': conf_matrix_gt})

    # For fine-level supervision (not fully implemented here, but necessary keys are provided)
    if len(b_ids) == 0:
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({'spv_b_ids': b_ids, 'spv_i_ids': i_ids, 'spv_j_ids': j_ids})
    
    # Dummy values for fine supervision keys used by the original datasets
    grid_pt1_i = create_meshgrid(H0, W0, False, device).reshape(1, H0 * W0, 2).repeat(N, 1, 1)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i,
        # These two are not accurate for homography but are needed to prevent crashes.
        # Fine-level loss might be incorrect without further changes.
        'spv_w_pt1_i': grid_pt0_i,
        'spv_pt0_i': grid_pt0_i,
    })

