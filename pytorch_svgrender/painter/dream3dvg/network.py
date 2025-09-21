import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import pydiffvg

import random
from typing import List, Dict, Union, Tuple, Any

from pytorch_svgrender.utils.misc import (
    PROJ_TYPES,
    RAND_TYPES,
    get_rand_fn,
    get_mean_dist,
    blender2world,
    rand_on_line,
    HWF,
)
from pytorch_svgrender.utils.math_utils import euclidean_distance
from pytorch_svgrender.utils.graphics_utils import fov2focal
from pytorch_svgrender.utils.sh_utils import eval_sh
import open3d as o3d
from torchvision.utils import save_image

class CurveRenderer(nn.Module):
    def __init__(
        self,
        device: str = "cuda:0",
        style: str = 'sketch',
        proj_mode: str = "ortho",
        proj_scale: float = 1.0,
        blender: bool = True,
        use_sfm: bool = True,
        optim_color: bool = False,
        gap: float = 0.01,
        center: List[float] = [0.0, 0.0, 0.0],
        rand_mode: str = "hemisphere",
        upside: bool = True,
        boundaries: List[float] = [0.2],
        stroke_width: float = 1.5,
        num_strokes: int = 32,
        num_segments: int = 4,
        pts_per_seg: int = 4,
        eps: float = 1.0e-8,
        add_noise: bool = False,
        noise_thres: float = 0.5,
        color_thres: float = 0.0,
        pts_dist_thres: float = 0.015,
        use_viewnet: bool = False,
    ):
        """Module to compute view-independent lines"""

        super().__init__()

        self.style = style
        self.device = device

        assert proj_mode in PROJ_TYPES
        self.proj_mode = proj_mode
        self.projection = (
            self.perspective if self.proj_mode == "persp" else self.orthographic
        )
        self.proj_scale = proj_scale
        self.blender = blender

        self.use_sfm = use_sfm
        self.start_points: torch.Tensor = None
        self.start_colors: torch.Tensor = None
        
        self.gap = gap
        self.center = torch.Tensor(center).to(self.device)

        assert rand_mode in RAND_TYPES
        self.set_random_point = get_rand_fn(rand_mode == "hemisphere", z_nonneg=upside)
        self.boundaries = boundaries[0] if rand_mode == "hemisphere" else boundaries

        self.stroke_width = stroke_width

        self.num_strokes = num_strokes
        
        self.pts_per_seg = pts_per_seg
        self.eps = eps
        
        # svg style
        if self.style == 'sketch':
            self.num_segments = 1 
            self.num_points = self.num_strokes * self.num_segments * self.pts_per_seg
        
        elif self.style == 'iconography':
            self.num_segments = num_segments
            self.num_points = self.num_strokes * self.num_segments * 3
         
        # set optimized params via style
        self.optim_point, self.optim_color, self.optim_width = {
            'sketch': (True, False, False),
            'iconography': (True, True, False),
        }.get(self.style, (False, False, False))
        
        # alpha optimization - 视角相关的透明度/颜色优化
        self.use_viewnet = use_viewnet
        
        if self.use_viewnet:
            if self.style == 'sketch':
                # 素描风格：只预测透明度（单一值）
                # 原因：素描通常使用固定颜色（黑色），只需控制线条的显示强度
                self.alpha_net = ViewpointModel(output_dim=1).to(self.device)
            elif self.style == 'iconography':
                # 图标风格：预测完整RGBA四通道
                # 原因：图标需要丰富色彩，不同视角可能需要不同颜色表现
                self.alpha_net = ViewpointModel(output_dim=4).to(self.device)

        self.add_noise = add_noise
        self.add_noise_init = add_noise  # backup values
        self.noise_thres = noise_thres
        self.color_thres = color_thres
        self.pts_dist_thres = pts_dist_thres

        # initailize parameters as zero or an empty set
        # 初始化SVG渲染相关的核心数据结构 - SVG rendering core data structures
        

        # SVG几何存储 - SVG geometry storage
        self.shapes = {}  # 字典：存储3D控制点到SVG路径的映射 {torch.Tensor(3D_points): pydiffvg.Path}
                         # Dictionary: Maps 3D control points to SVG path objects
        self.shapes_xing = []  # 列表：扩展形状存储（预留接口，当前未使用）
                              # List: Extended shape storage (reserved interface, currently unused)
        
        # SVG属性管理 - SVG attributes management  
        self.shape_groups = {}  # 字典：存储3D控制点到SVG形状组的映射 {torch.Tensor(3D_points): pydiffvg.ShapeGroup}
                               # Dictionary: Maps 3D control points to SVG shape groups (colors, stroke, fill properties)
        
        # 优化参数容器 - Optimization parameter containers
        self.point_params = []  # 列表：存储所有可优化的3D控制点参数 [torch.Tensor([N,3]), ...]
                               # List: All optimizable 3D control point parameters for gradient optimization
        self.color_params = []  # 列表：存储所有可优化的颜色参数 [torch.Tensor([R,G,B,A]), ...]
                               # List: All optimizable color parameters (RGBA values)
        self.width_params = []  # 列表：存储所有可优化的线宽参数（预留接口）
                               # List: All optimizable stroke width parameters (reserved interface)
        
        # 优化控制标志 - Optimization control flags
        self.optimize_flag = []  # 列表：标记每个形状是否参与优化 [bool, bool, ...]
                                # List: Boolean flags indicating which shapes participate in optimization

        # initialize basic attributes related to dataset
        self.H: int = None
        self.W: int = None
        self.intrinsic: torch.Tensor = None
        
        # for fps
        self.sfm_voxel_size: float = 0.02
        self.sfm_filter_fn: str = 'radius'
        self.sfm_filter_params: List[Union[int, float]] = [10, 0.2]
        self.sfm_second_filter: bool = True
        self.sfm_filter_ratio: float = 0.03 
        
        # for iconography style
        self.strokes_counter = 0 

    def init_properties_viewer(self, hwf: HWF) -> None:
        self.set_intrinsic(hwf)

    def init_properties(self, dataset):
        """Reflect properties of given virtual scene"""

        # camera properties
        self.H, self.W = dataset.test_cameras[0].height, dataset.test_cameras[0].width
        FovX, FovY = dataset.test_cameras[0].FovX, dataset.test_cameras[0].FovY
        self.F = fov2focal(FovX, self.W) # here H = W, just when init, camera param will change in each iteration
        
        if self.proj_mode == "persp":  # perspective
            self.intrinsic = (
                torch.Tensor([[self.F, 0, self.W / 2], [0, self.F, self.H / 2], [0, 0, 1]])
                .float()
                .to(self.device)
            )
        else:  # orthographic
            self.intrinsic = (
                torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                .float()
                .to(self.device)
            )
        
        # get extracted points or randomize points to set initial strokes
        if self.start_points is None:
            if self.use_sfm:
                self.start_points, self.num_strokes, self.start_colors = self.fps_from_sfm(
                    points=dataset.point_cloud.points, num_points=self.num_strokes, colors=dataset.point_cloud.colors
                )
            else:
                print("You are not using SfM points or the points are not available. Randomly initialize the starting points.")
                randomized = [
                    self.set_random_point(self.boundaries, device=self.device)
                    + self.center
                    for _ in range(self.num_strokes)
                ]
                self.start_points = torch.stack(randomized)

            # print final status of starting points
            print(
                f"Initialized curves: [num_strokes] {self.num_strokes} | [use_point_clouds] {self.use_sfm}"
            )

    def init_properties_dataset(self, dataset):
        """Initialize properties for dataset-based training mode"""
        # Similar to init_properties but adapted for dataset mode
        self.init_properties(dataset)  # Reuse the existing logic
            
    def fps_from_sfm(self, points, num_points: int, colors=None, init_alpha: float = 1.0) -> Tuple[Union[torch.Tensor, int]]:
        """FPS from points in sfm point cloud."""
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # add color if not None
        if colors is not None:
            assert len(points) == len(colors), "Points and colors must have the same length"
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # remove outliers
        sampled_pcd = pcd.voxel_down_sample(voxel_size=self.sfm_voxel_size)
        p1, p2 = self.sfm_filter_params
        if self.sfm_filter_fn == "statistic":
            cl, _ = sampled_pcd.remove_statistical_outlier(
                nb_neighbors=p1, std_ratio=p2
            )
        elif self.sfm_filter_fn == "radius":
            cl, _ = sampled_pcd.remove_radius_outlier(nb_points=p1, radius=p2)
        else:  # fn must be in ["statistic", "radius"]
            raise ValueError
        cl_pts = np.asarray(cl.points)
        cl_colors = np.asarray(cl.colors) if colors is not None else None
        
        # second filtering
        if self.sfm_second_filter:
            lower, upper = (
                self.sfm_filter_ratio * 100,
                (1 - self.sfm_filter_ratio) * 100,
            )
            perc_low = np.percentile(cl_pts, lower, axis=0, keepdims=True)
            perc_high = np.percentile(cl_pts, upper, axis=0, keepdims=True)
            low_filter = cl_pts > perc_low
            low_filter = np.logical_and(
                np.logical_and(low_filter[:, 0], low_filter[:, 1]), low_filter[:, 2]
            )
            high_filter = cl_pts < perc_high
            high_filter = np.logical_and(
                np.logical_and(high_filter[:, 0], high_filter[:, 1]), high_filter[:, 2]
            )
            filter = np.logical_and(low_filter, high_filter)
            cl.points = o3d.utility.Vector3dVector(cl_pts[filter])
            if colors is not None:
                cl.colors = o3d.utility.Vector3dVector(cl_colors[filter])
                
        # apply fps to get initial coordinates of points
        n_all = len(cl.points)
        if n_all > num_points:
            computed = cl.farthest_point_down_sample(num_points)
            points = computed.points
            if colors is not None:
                colors = np.asarray(computed.colors)
        else:
            raise ValueError("Not enough points")
            # num_points = n_all
            # points = cl_filtered.points
        
        # Add alpha channel to colors
        if colors is not None:
            alpha_channel = np.full((colors.shape[0], 1), init_alpha)  # Create an alpha channel of the same length
            colors = np.concatenate([colors, alpha_channel], axis=1)  # Add the alpha channel to the colors
        
        sampled = torch.from_numpy(np.asarray(points)).float().to(self.device)
        sampled_colors = torch.from_numpy(colors).float().to(self.device) if colors is not None else None
        
        return sampled, num_points, sampled_colors if colors is not None else None
    
    def get_pts_3d(self, pt0: torch.Tensor, radius: float = 0.001) -> torch.Tensor:
        
        """Initialize the path starting with pt0"""
        # if have 2 points, set them as start and end, and choose 2 mid points as control points
        # if only 1 point, set a small radius, and set 3 points together with pt0 as a bezier curve
        
        if self.style in ['sketch']:
            if len(pt0.shape) == 2:
                start, end = pt0[0], pt0[1]
                # mid_points = rand_on_circle(start, end, num_points=2)
                mid_points = rand_on_line(start, end, num_points=2)
                pts_ls = [start] + mid_points + [end]
            else:
                radius = torch.ones([3]).to(self.device) * radius
                pts_ls = [pt0]
                
                for _ in range(self.num_segments):
                    for _ in range(self.pts_per_seg - 1):
                        pt1 = pt0 + radius + torch.rand([3]).to(self.device) * self.gap
                        pts_ls.append(pt1)
                        pt0 = pt1
            
            # stack all points to control easily
            pts = torch.stack(pts_ls)  # [N_pts, 3]
            
        elif self.style == 'iconography':

            theta = torch.linspace(0, 2 * torch.pi, self.num_segments * 3).to(self.device) 
            phi = torch.linspace(0, torch.pi, self.num_segments * 3).to(self.device) 
            
            x = radius * torch.sin(phi) * torch.cos(theta)
            y = radius * torch.sin(phi) * torch.sin(theta)
            z = radius * torch.cos(phi)
                
            pts = torch.stack((x, y, z), dim=-1) + pt0

        return pts

    def perspective(self, pts_3d: torch.Tensor, pose: torch.Tensor, w2c=False, intrinsic=None) -> torch.Tensor:
        """Perspective projection."""
        if not w2c:
            extrinsic = torch.linalg.inv(pose)  # w2c
        else:
            extrinsic = pose
            
        if self.blender:
            extrinsic = blender2world(
                extrinsic[:3, :3], extrinsic[:3, -1:], self.device
            )
            
        intrinsic = intrinsic if intrinsic is not None else self.intrinsic
        pts_3d_hg = torch.concat([pts_3d, torch.ones_like(pts_3d[..., -1:])], dim=-1)
        world_matrix = intrinsic @ extrinsic[:3, ...]
        pts_2d_aug = pts_3d_hg @ world_matrix.T
        pts_2d_aug = pts_2d_aug / self.proj_scale
        pts_2d = pts_2d_aug[..., :-1] / pts_2d_aug[..., -1:]  # [N_pts, 2]

        return pts_2d.contiguous()

    def initialize(self, pose: torch.Tensor) -> torch.Tensor:
        """Initialize strokes with the given view."""
        # pose here w2c
        
        if self.style == 'sketch':
            self.num_control_pts = torch.zeros(self.num_segments, dtype=torch.int32) + 2
            stroke_color = torch.Tensor([0.0, 0.0, 0.0, 1.0]).to(self.device)
            
            if len(self.point_params) == 0:
                for pt0 in self.start_points:
                    pt, path = self.init_path(pose, pt0)
                    self.shapes[pt] = path
                    
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.Tensor([len(self.shapes) - 1]).int(),
                        fill_color=None,
                        stroke_color=stroke_color.clone(),
                    )
                    
                    self.shape_groups[pt] = path_group
                
                self.optimize_flag = [True for _ in range(len(self.shapes))]
            else:
                self.load_paths(pose, stroke_color)
              
        elif self.style == 'iconography':
            self.num_control_pts = [2] * self.num_segments

            if self.optim_color:
                for pt0, color0 in zip(self.start_points, self.start_colors):
                    pt, path = self.init_path(pose, pt0)
                    self.shapes[pt] = path
                    
                    path_group = pydiffvg.ShapeGroup(
                            shape_ids=torch.Tensor([self.strokes_counter - 1]).int(),
                            fill_color=color0.clone(),
                            stroke_color=None,
                        )
                    self.shape_groups[pt] = path_group
                    self.optimize_flag = [True for _ in range(len(self.shapes))]
            else:
                raise NotImplementedError(f"{self.style} is only supported for color optimization in current version.")
                     
        else:
                raise NotImplementedError(f"{self.style} is only supported in current version.")
        
        img = self.render_warp(pose, w2c=True, init=True)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            self.H, self.W, 3, device=self.device
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0).permute(0, 3, 1, 2)  # [1, H, W, C] -> [1, C, H, W]
        
        return img

    def initialize_dataset(self, pose: torch.Tensor) -> torch.Tensor:
        """Initialize strokes for dataset-based training mode."""
        # For dataset mode, use the same initialization as regular mode
        return self.initialize(pose)

    def init_path(
        self, pose: torch.Tensor, pt: torch.Tensor=None, coord: List =None
    ) -> Tuple[torch.Tensor, pydiffvg.Path]:
        """Set a path based on the starting point"""
        
        if self.style in ['sketch']:
            
            # get 3D control points
            points = self.get_pts_3d(pt)
        
            path = pydiffvg.Path(
                num_control_points=self.num_control_pts,
                points=self.projection(points, pose, w2c=True),  # initialized -> project the 3D points to 2D plane, with the pose
                stroke_width=torch.tensor(self.stroke_width, device=self.device),
                is_closed=False,
            )
            
        elif self.style == 'iconography':
            points = self.get_pts_3d(pt, radius=0.1)
            path = pydiffvg.Path(
                num_control_points=torch.LongTensor(self.num_control_pts),
                points=self.projection(points, pose, w2c=True),  # initialized -> project the 3D points to 2D plane, with the pose
                stroke_width=torch.tensor(0.0, device=self.device),
                is_closed=True,
            )
            
        self.strokes_counter += 1   
        return points, path


    def load_paths(self, pose: torch.Tensor, stroke_color: torch.Tensor):
        """Load paths from given control points"""

        if len(self.color_params) == 0:
            color_params = [stroke_color for _ in range(len(self.point_params))]
        else:
            color_params = self.color_params

        new_idx = 0
        for pt, flag, color in zip(self.point_params, self.optimize_flag, color_params):
            if flag:
                path = pydiffvg.Path(
                    num_control_points=self.num_control_pts,
                    points=self.projection(pt, pose),  # initialized
                    stroke_width=torch.tensor(self.stroke_width, device=self.device),
                    is_closed=False,
                )
                self.shapes[pt] = path
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.Tensor([new_idx]).int(),
                    fill_color=None,
                    stroke_color=color,
                )
                self.shape_groups[pt] = path_group
                new_idx += 1
    
    def sample_depth_value(self, points_3d, pose, depth, intrinsic):
        points_2d = self.projection(points_3d, pose, True, intrinsic) # (num_paths, *, 2)
        points_2d = torch.clamp((points_2d * 2 - self.H) / self.H, -1., 1.).unsqueeze(0)
        points_depth = F.grid_sample(depth.unsqueeze(0), points_2d, mode='bilinear', align_corners=False) 
        return points_depth.squeeze(0).squeeze(0)
        
    def bezier_curve_3d(self, points: torch.Tensor, num_points: int = 100):
        """Evaluates a cubic Bezier curve at `num_points` points in 3D.
    """
        p0, p1, p2, p3 = points[0], points[1], points[2], points[3]
        t = torch.linspace(0, 1, num_points, device=points.device).unsqueeze(1)
        points = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
        return points
    
    def bezier_curve_3d_icon(self, points: torch.Tensor, num_points: int = 100):
        """Evaluates a cubic Bezier curve at `num_points` points in 3D.
    """
        line0 = points[0:4, ...]
        line1 = points[3:7, ...]
        line2 = points[6:10, ...]
        line3 =  torch.cat([points[9:, ...], points[0, ...].unsqueeze(0)], dim=0)
        lines = [line0, line1, line2, line3]
        curve_points = []
        for line in lines:
            curve_points.append(self.bezier_curve_3d(line, num_points))
        curve_points = torch.cat(curve_points, dim=0)
        return curve_points
            
    def render_warp(self, 
                    pose: torch.Tensor, 
                    w2c=False, 
                    intrinsic=None, 
                    depth=None,
                    opposite_pose=None,
                    opposite_depth=None,
                    init=False,
                    is_test=False,
                    ) -> torch.Tensor:
        
        """Render sketches with projected 2D points."""
        if self.style == 'sketch':
            for group in self.shape_groups.values():
                group.stroke_color.data[:3].clamp_(0.0, 0.0)  # to force black strokes
                group.stroke_color.data[-1].clamp_(0.0, 1.0)  # opacity
            
            # apply 2D projection with given position
            for pt, path in self.shapes.items():
                path.points = self.projection(pt, pose, w2c=w2c, intrinsic=intrinsic)
            
            shapes_2d = list(self.shapes.values())
            shape_groups = list(self.shape_groups.values())
            
            # =====================================================================
            # 视角相关透明度控制系统 - View-dependent Transparency Control System
            # =====================================================================
            # 该系统实现了智能的笔画可见性控制，结合神经网络预测和几何深度信息，
            # 确保从不同视角观看时只显示相关的笔画，提高3D感知效果。
            # 
            # This system implements intelligent stroke visibility control by combining
            # neural network predictions with geometric depth information to ensure
            # only relevant strokes are shown from different viewpoints, enhancing 3D perception.
            #
            # 工作流程 - Workflow:
            # 1. 神经网络重要性预测：使用ViewpointModel预测每个笔画的重要性分数
            # 2. 几何深度投票：通过多视角深度一致性检验验证笔画的几何合理性  
            # 3. 双重过滤决策：结合两者信息做出最终的透明度决策
            # =====================================================================
            if self.use_viewnet:
                if not init:
                    # ===== 第一步：神经网络重要性预测过滤 =====
                    # Importance filtering using neural network (MLP version)
                    num_points_per_path = 32  # 每条贝塞尔曲线采样的点数 - Each bezier curve is sampled into 32 points
                    
                    with torch.no_grad():
                        # 为每个参数化的控制点生成3D空间中的贝塞尔曲线采样点
                        # Generate 3D bezier curve sample points for each parameterized control point
                        points_3d = []
                        for point in self.point_params:
                            points_3d.append(self.bezier_curve_3d(point, num_points=num_points_per_path))
                        points_3d = torch.cat(points_3d, dim=0).view(-1, num_points_per_path, 3)  # (num_path, k, 3)
                    
                        # 计算每个3D点到相机的深度值 - Calculate depth from each 3D point to camera
                        # 使用相机姿态矩阵将3D点变换到相机坐标系，提取Z轴深度
                        point_depths = (torch.einsum("hwn,mn->hwm", points_3d, pose[:3, :3]) + pose[:3,-1].view(1, 1, 3))[..., -1]

                    # 使用ViewpointModel神经网络预测每个点的重要性分数
                    # Use ViewpointModel neural network to predict importance score for each point
                    # SKETCH风格：只预测透明度值（单通道输出） - SKETCH style: only predict alpha values (single channel output)
                    point_alphas = self.alpha_net(points_3d, point_depths).squeeze(-1)  # [B*N, 1] -> [B*N] 预测点的重要性
                    curve_alphas = torch.mean(point_alphas, dim=-1)  # 计算每条曲线的平均重要性 - Calculate average importance for each curve

                    # 神经网络重要性阈值过滤：重要性>0.75的点被认为是重要的
                    # Neural network importance threshold filtering: points with importance > 0.75 are considered important
                    alpha_filter = point_alphas > 0.75
                    
                    # 为每个笔画组分配神经网络预测的透明度，保持原有RGB颜色不变
                    # Assign neural network predicted alpha values to each stroke group while preserving original RGB colors
                    for ind, shape_group in enumerate(shape_groups):
                        curve_alpha = curve_alphas[ind]  # 获取该曲线的神经网络预测平均透明度 - Get NN predicted average alpha for this curve
                        rgb = shape_group.stroke_color[:3].detach()  # 保持原有RGB颜色（通常是黑色）- Preserve original RGB color (usually black)
                        # 只更新alpha通道，实现视角相关的透明度控制 - Only update alpha channel for view-dependent transparency control
                        adaptive_color = torch.cat((rgb, curve_alpha.unsqueeze(-1)), dim=-1)  # [R,G,B,A]
                        shape_group.stroke_color = adaptive_color
                    
                    # ===== 第二步：几何深度投票机制 =====
                    # Geometric depth voting mechanism for view consistency
                    
                    if depth is not None and opposite_depth is not None:
                        # 计算对向视角下每个3D点的深度值 - Calculate depth of each 3D point from opposite viewpoint
                        point_depths_opposite = (torch.einsum("hwn,mn->hwm", points_3d, opposite_pose[:3, :3]) + opposite_pose[:3, -1].view(1, 1, 3))[..., -1]
                        
                        # 从深度图中采样得到像素级的真实深度值 - Sample real pixel-level depth values from depth maps
                        pixel_depths = self.sample_depth_value(points_3d, pose, depth, intrinsic)  # 当前视角深度图采样
                        pixel_depths_opposite = self.sample_depth_value(points_3d, opposite_pose, opposite_depth, intrinsic)  # 对向视角深度图采样
                        
                        # 计算两个视角间像素深度的差异范围，用于容差判断
                        # Calculate pixel depth difference range between two viewpoints for tolerance judgment
                        depth_range = torch.abs(pixel_depths - pixel_depths_opposite)
                        
                        # 几何一致性判断：比较几何深度与像素深度的差异
                        # Geometric consistency judgment: compare differences between geometric and pixel depths
                        diff_pos = torch.abs(point_depths - pixel_depths) - depth_range * 0.25  # 当前视角的深度误差（带容差）
                        diff_neg = torch.abs(point_depths_opposite - pixel_depths_opposite)     # 对向视角的深度误差
                        
                        # 深度投票：当前视角误差小于对向视角误差时，认为该点在前景
                        # Depth voting: when current view error < opposite view error, consider the point as foreground
                        depth_filter = diff_pos < diff_neg
                        
                        # ===== 第三步：双重过滤的最终透明度决策 =====
                        # Final transparency decision with dual filtering
                        for ind, shape_group in enumerate(shape_groups):
                            # 对于神经网络判断为不重要的曲线，使用深度投票进行进一步筛选
                            # For curves deemed unimportant by neural network, use depth voting for further filtering
                            if not alpha_filter[ind].all():  
                                # 深度投票：如果75%以上的采样点通过深度一致性检验，认为在前景
                                # Depth voting: if >75% sample points pass depth consistency test, consider as foreground
                                if depth_filter[ind].sum() >= num_points_per_path * 0.75: 
                                    shape_group.stroke_color.data[-1].clamp_(1.0, 1.0)  # 前景：完全不透明 -> front view: fully opaque
                                else:
                                    # 背景：训练时完全透明，测试时保持0.2透明度便于观察
                                    # Background: fully transparent during training, 0.2 alpha during testing for visibility
                                    shape_group.stroke_color.data[-1].clamp_(0.0, 1.0) if not is_test else shape_group.stroke_color.data[-1].clamp_(0.2, 0.2)
                            else: 
                                # 神经网络判断为重要的曲线：强制设为完全不透明，确保关键笔画可见
                                # Curves deemed important by neural network: force fully opaque to ensure key strokes are visible
                                shape_group.stroke_color.data[-1].clamp_(1.0, 1.0)
        
                # ===== 初始化阶段：显示所有曲线 =====
                # Initialization phase: show all curves
                else:
                    # 初始化时所有笔画都设为完全不透明，确保完整的初始SVG可见
                    # During initialization, all strokes are set to fully opaque to ensure complete initial SVG visibility
                    for ind, shape_group in enumerate(shape_groups):
                        shape_group.stroke_color.data[-1].clamp_(1.0, 1.0)
            
            # render 
                     
            scene_args = pydiffvg.RenderFunction.serialize_scene(
                self.W, self.H, shapes_2d, shape_groups
            )
               
        elif self.style == 'iconography':
            # iconography optimize color
            for group in self.shape_groups.values():
                group.fill_color.data.clamp_(0.0, 1.0)

            for pt, path in self.shapes.items():
                path.points = self.projection(pt, pose, w2c=w2c, intrinsic=intrinsic)
            
            shapes_2d = list(self.shapes.values())
            shape_groups = list(self.shape_groups.values())
            
            if not init:
                num_points_per_path = 32 # k
                points_3d = []
                
                with torch.no_grad():
                    for point in self.point_params:
                        points_3d.append(self.bezier_curve_3d_icon(point, num_points=num_points_per_path)) # [[4*num_pts, 3]]
                    # all curves of an icon, then will use pts_3d to calculate importance
                    points_3d = torch.cat(points_3d, dim=0).view(-1, num_points_per_path, 3) # (num_paths*4, k, 3)
                    
                    # point to camera depth
                    point_depths = (torch.einsum("hwn,mn->hwm", points_3d, pose[:3, :3]) + pose[:3, -1].view(1, 1, 3))[..., -1] # (num_paths*4, k)

                # ICONOGRAPHY风格：预测完整RGBA四通道
                point_alphas = self.alpha_net(points_3d, point_depths)  # (num_paths*4, k, 4) 预测每个点的RGBA值
                curve_alphas = torch.mean(point_alphas, dim=1)  # （num_paths*4, 4）计算每条曲线的平均RGBA
                
                alpha_filter = (point_alphas[:,:,-1] > 0.75).view(len(self.shapes), -1)  # 基于alpha通道的重要性过滤
                
                # 为每个图标形状分配完整的RGBA颜色
                for ind, shape_group in enumerate(shape_groups):
                    # 每个图标由4条贝塞尔曲线组成，计算这4条曲线的平均RGBA
                    adaptive_color = curve_alphas[ind*4: (ind+1)*4, ...].mean(dim=0)  # 平均RGBA值
                    shape_group.fill_color = adaptive_color  # 直接设置填充颜色（包含RGB和A）
                
                # depth voting
                if depth is not None and opposite_depth is not None:
                    point_depths_opposite = (torch.einsum("hwn,mn->hwm", points_3d, opposite_pose[:3, :3]) + opposite_pose[:3, -1].view(1, 1, 3))[..., -1]
                    pixel_depths = self.sample_depth_value(points_3d, pose, depth, intrinsic)
                    pixel_depths_opposite = self.sample_depth_value(points_3d, opposite_pose, opposite_depth, intrinsic)
                    
                    # pixel-level depth range
                    depth_range = torch.abs(pixel_depths - pixel_depths_opposite)
                    
                    diff_pos = torch.abs(point_depths - pixel_depths) - depth_range * 0.25
                    diff_neg = torch.abs(point_depths_opposite - pixel_depths_opposite)
                    
                    depth_filter = (diff_pos < diff_neg).view(len(self.shapes), -1)
                    
                    for ind, shape_group in enumerate(shape_groups):
                        if not alpha_filter[ind].all(): # for curves not important, depth voting
                            if depth_filter[ind].sum() >= num_points_per_path * self.num_segments * 0.5:
                                shape_group.fill_color.data[-1].clamp_(1.0, 1.0)
                            else:
                                shape_group.fill_color.data[-1].clamp_(0.0, 1.0) if not is_test else shape_group.fill_color.data[-1].clamp_(0.2, 0.2)
                        else:
                            shape_group.fill_color.data[-1].clamp_(1.0, 1.0)
                            
                path_depths = point_depths.view(len(self.shapes), -1).mean(dim=-1)
                render_orders = torch.argsort(path_depths, descending=True)
                
                # re-order
                reordered_shapes_2d = []
                reordered_shape_groups = []
                for ind, render_order in enumerate(render_orders):
                    reordered_shapes_2d.append(shapes_2d[render_order.item()])
                    reordered_shape_groups.append(shape_groups[render_order])
                    reordered_shape_groups[-1].shape_ids = torch.tensor(ind).unsqueeze(0).to(torch.int32).to(self.device)
                
                # render                      
                scene_args = pydiffvg.RenderFunction.serialize_scene(
                    self.W, self.H, reordered_shapes_2d, reordered_shape_groups
                )      
                
            else:                          
                scene_args = pydiffvg.RenderFunction.serialize_scene(
                    self.W, self.H, shapes_2d, shape_groups
                )
        
        _render = pydiffvg.RenderFunction.apply#diffvg渲染流程：serialize_scene && apply
        
        img = _render(
            self.W,  # width
            self.H,  # height
            2,  # num_samples_x
            2,  # num_samples_y
            0,  # seed
            None,
            *scene_args,
        )
        
        return img

    def alpha_blending(self, img):
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            self.H, self.W, 3, device=self.device
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0).permute(0, 3, 1, 2)
        return img
        
    def get_opposite_camera_matrix(self, w2c):
        
        rotation = w2c[:3, :3]
        position = w2c[:3, 3]
        
        new_rotation = rotation @ torch.diag(torch.tensor([-1., -1., 1.])).to(self.device)
        new_position = position
        
        new_w2c = torch.eye(4).to(self.device)
        new_w2c[:3, :3] = new_rotation
        new_w2c[:3, 3] = new_position

        return new_w2c

    def sketch(self, pose: torch.Tensor, w2c=False, intrinsic=None, depth=None, opposite_pose=None, opposite_depth=None, is_test=False) -> torch.Tensor:
        # self.sort_strokes(pose)  # XXX remove this if needed
        sketch = self.render_warp(pose, w2c=w2c, intrinsic=intrinsic, 
                    depth=depth, opposite_pose=opposite_pose,
                    opposite_depth=opposite_depth,
                    is_test=is_test)
        
        alpha = sketch[..., -1:]
        sketch = sketch[..., :3] * alpha + (1.0 - alpha)
        sketch = sketch.unsqueeze(0).permute(0, 3, 1, 2)  # [1, H, W, 3] -> [1, 3, H, W]

        return sketch

    def save_svg(self, fname: str):
        shapes = list(self.shapes.values())
        shape_groups = list(self.shape_groups.values())
        pydiffvg.save_svg(fname, self.W, self.H, shapes, shape_groups)

    def cleaning(self, show: bool = True):
        counts = 0
        for i, pts in enumerate(self.point_params):
            mean_dist = get_mean_dist(pts)
            if mean_dist <= self.pts_dist_thres:
                self.inactive_stroke(i)
                counts += 1
        self.num_strokes -= counts
        if show:
            print(f"number of strokes: {len(self.point_params)} -> {self.num_strokes}")

    def inactive_stroke(self, idx: int):
        self.point_params[idx].requires_grad = False
        self.optimize_flag[idx] = False

    def set_point_params(self) -> List[torch.Tensor]:
        if len(self.point_params) == 0:
            self.point_params = []
            for i, pts in enumerate(self.shapes.keys()):
                if self.optimize_flag[i]:
                    pts.requires_grad = True
                    self.point_params.append(pts)

        return self.point_params

    def set_color_params(self) -> List[torch.Tensor]:
        if self.style in ['sketch', 'painting']:
            if len(self.color_params) == 0:
                self.color_params = []
                for i, group in enumerate(self.shape_groups.values()):
                    if self.optimize_flag[i]:
                        group.stroke_color.requires_grad = True
                        self.color_params.append(group.stroke_color)
        elif self.style == 'iconography':
            if self.use_viewnet:
                if len(self.color_params) == 0:
                    self.color_params = []
                    for i, group in enumerate(self.shape_groups.values()):
                        if self.optimize_flag[i]:
                            group.fill_color.requires_grad = True
                            self.color_params.append(group.fill_color)
            else:
                if len(self.color_params) == 0:
                    self.color_params = []
                    for i, group in enumerate(self.shape_groups.values()):
                        if self.optimize_flag[i]:
                            group.fill_color.requires_grad = True
                            self.color_params.append(group.fill_color)
        return self.color_params

    def set_random_noise(self, save: bool = False):
        self.add_noise = False if save else self.add_noise_init

    def load_state_dict(self, ckpt: Dict[str, Any]):
        self.point_params = ckpt["point_params"]
        self.color_params = ckpt["color_params"]
        self.optimize_flag = ckpt["optimize_flag"]

    def state_dict(self) -> Dict[str, List[Any]]:
        states = {
            "point_params": self.point_params,
            "color_params": self.color_params,
            "optimize_flag": self.optimize_flag,
        }

        return states

    def gui(self) -> None:
        pass

# 功能：基于3D点位置和深度信息，预测该点在当前视角下的重要性/透明度。                
class ViewpointModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=1):
        super(ViewpointModel, self).__init__()
        # 网络结构：
        # 输入维度：3D坐标 (x,y,z) + 深度值
        # 位置编码：使用正弦余弦函数进行高频编码
        # 隐藏层：128 → 64 → output_dim
        # 激活函数：ReLU + Sigmoid输出
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.freq_embed, mlp_input_dim = self.get_embedder()
        self.fc1 = nn.Linear(mlp_input_dim + 1, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, output_dim)
        self.act = nn.Sigmoid()
        self.init_weights()
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_embedder(self, multires=6):
        
        embed_kwargs = {
            'include_input': True,
            'input_dim': self.input_dim,
            'max_freq_log2': multires-1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        
        embedder_obj = Embedder(**embed_kwargs)
        def embed(x, eo=embedder_obj): return eo.embed(x)
        
        return embed, embedder_obj.out_dim
    
    def norm(self, x):
        return (x - x.mean()) / x.std()
        
    def forward(self, x, depth):
        B, N, C = x.shape
        x = x.view(-1, C)
        x = self.freq_embed(x)
        x = torch.cat([x, depth.view(B*N, -1)], dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = self.act(x)
        return x.view(B, N, self.output_dim)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dim']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
        out_dim += d
        
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class CurveOptimizer:
    def __init__(
        self,
        module: CurveRenderer,
        point_lr: float = 1.0,
        color_lr: float = 0.01,
    ):
        """Optimizer used in CLIPasso.
        mainly referred to https://github.com/yael-vinker/CLIPasso/blob/main/models/painter_params.py
        """

        # renderer to optimize
        self.module = module
        # variables related to an optimizer
        self.point_lr = point_lr
        self.color_lr = color_lr
        self.optim_point = module.optim_point
        self.optim_color = module.optim_color
        print(f"Optimize colors: {self.optim_color}")

    def initialize(self):
        if self.optim_point:
            self.point_optim = optim.Adam(self.module.set_point_params(), lr=self.point_lr)
        if self.optim_color:
            self.color_optim = optim.Adam(self.module.set_color_params(), lr=self.color_lr)
        if self.module.use_viewnet:
            self.alpha_optim = optim.Adam(self.module.alpha_net.parameters(), lr=self.color_lr)

    def zero_grad(self):
        if self.optim_point:
            self.point_optim.zero_grad()
        if self.optim_color:
            self.color_optim.zero_grad()
        if self.module.use_viewnet:
            self.alpha_optim.zero_grad()
        
    def step(self):
        if self.optim_point:
            self.point_optim.step()
        if self.optim_color:
            self.color_optim.step()    
        if self.module.use_viewnet:
            self.alpha_optim.step()

    def state_dict(self) -> Dict[str, Any]:
        params = {}
        if self.optim_point:
            params["point"] = self.point_optim.state_dict()
        if self.optim_color:
            params["color"] = self.color_optim.state_dict()
        if self.module.use_viewnet:
            params["alpha"] = self.alpha_optim.state_dict()
            
        return params

    def load_state_dict(self, ckpt: Dict[str, Any]):
        self.point_optim.load_state_dict(ckpt["point"])
        if self.optim_color and "color" in ckpt.keys():
            self.color_optim.load_state_dict(ckpt["color"])

    def get_lr(self) -> float:
        return self.point_optim.param_groups[0]["lr"]

