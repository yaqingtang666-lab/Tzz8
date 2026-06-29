# SMPL LBS 实验 — 线性混合蒙皮过程可视化

## 实验概述

本项目基于 [SMPL](https://smpl.is.tue.mpg.de/) 参数化人体模型，实现了完整的 LBS 蒙皮流水线，将官方 lbs() 函数中的每一个关键中间量单独提取出来，分别在四个阶段进行 3D 可视化渲染。最后将手写实现的结果与官方前向传播进行数值对比验证。

## 实验目标

1. 理解参数化人体模型中模板网格、形状参数（β）、姿态参数（θ）、关节回归器和蒙皮权重之间的关系。
2. 掌握 LBS 的四个阶段

---

## 核心架构：四阶段 LBS 流水线

### 阶段 (a) — 模板网格与蒙皮权重

```
T̄  (N_V × 3)         →  T-pose 下的模板顶点
W  (N_V × K)          →  每个顶点对各关节的蒙皮权重
```

- 模板 `T̄` 是处于 T-pose 的中性人体网格（6890 个顶点，13776 个面片）。
- `W` 是顶点-关节权重矩阵（SMPL 共 24 个关节），每行之和为 1。
- 此时网格**尚未**根据人物体型改变，也**尚未**根据姿态弯曲——但每个顶点已经知道"将来应该主要跟着哪些骨骼走"。

### 阶段 (b) — 形状校正与关节回归

```
v_shaped = T̄ + B_S(β)                     // β ∈ R¹⁰：形状系数
J        = J_regressor(v_shaped)           // 从校正后网格回归出 24 个关节
```

- `β` 通过线性形变基（`shapedirs`）编码人物体型（高矮、胖瘦、肩宽等）。
- `B_S(β) = blend_shapes(β, shapedirs)` 将这些位移叠加到模板上。
- 关节位置 `J` **不是固定常数**——它通过 `vertices2joints(J_regressor, v_shaped)` 从体型变化后的网格回归得到。体型的改变会导致关节位置相应移动。

### 阶段 (c) — 姿态校正

```
full_pose     = cat(global_orient, body_pose)      // [1, 72]  轴角表示
rot_mats      = batch_rodrigues(full_pose)          // [1, 24, 3, 3]
pose_feature  = flatten(rot_mats[:, 1:, :, :] − I)  // 旋转矩阵偏离单位阵的程度
pose_offsets  = matmul(pose_feature, posedirs)       // [1, N_V, 3]
v_posed       = v_shaped + pose_offsets
```

- 在将顶点真正绑定到骨骼之前，SMPL 会加入**姿态相关的校正偏移**（`B_P(θ)`），以处理关节弯曲处（肩、肘、膝）的额外几何变化——这些仅靠骨骼刚体旋转无法表达。
- `pose_feature` 是每个关节旋转矩阵与单位阵的偏差，编码了关节弯曲的*幅度*。
- `posedirs` 将该偏差通过线性模型映射为顶点位移。
- **v_posed 还不是最终结果**——网格虽已变形，但尚未绑定到骨骼上。

### 阶段 (d) — 线性混合蒙皮

```
J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
T                 = matmul(W, A)                     // 每个顶点一个 4×4 变换矩阵
v_homo            = [v_posed; 1]                     // 齐次坐标
verts             = T · v_homo                        // 最终蒙皮顶点
```

- `batch_rigid_transform` 沿运动学树（由 `parents` 定义）为每个关节计算**全局**刚体变换 `G_k`。
- 每个顶点的最终变换 `T_i` 是影响它的多个关节全局变换的*加权混合*：
  ```
  v_i′ = Σ_{k=1}^{K}  w_{ik} · G_k · [v_i^{posed}; 1]
  ```
- 这也就是 **Linear Blend Skinning** 名字的来源：每个顶点不是只跟着一个骨骼走，而是跟着多个关节做加权平均后的变换。

### 五个核心变量

| 变量 | 维度 | 含义 |
|---|---|---|
| `v_template` | [N_V, 3] | T-pose 下的原始模板网格 |
| `v_shaped` | [N_V, 3] | 模板 + 形状相关形变 |
| `J` | [K, 3] | 由 v_shaped 回归得到的关节位置 |
| `v_posed` | [N_V, 3] | v_shaped + 姿态相关校正偏移 |
| `verts` | [N_V, 3] | 经过 LBS 之后的最终蒙皮网格 |

---

## 项目结构

```
Tzz8/
├── run_lbs_lab.py                       # 主脚本（手动 LBS + 可视化 + 验证）
├── outputs/                             # 所有生成结果
│   ├── stage_a_template_weights.png     # 模板网格 + 单关节权重热力图
│   ├── all_joint_weights.png            # 每面片主导关节分布图（可选）
│   ├── stage_b_shaped_joints.png        # 形状校正后网格 + 回归关节点
│   ├── stage_c_pose_offsets.png         # 姿态校正网格（按偏移量着色）
│   ├── stage_d_lbs_result.png           # 最终蒙皮网格 + 变换后关节
│   ├── comparison_grid.png              # 四阶段 2×2 对比图
│   └── summary.txt                      # 模型元信息 + 数值验证结果
├── .gitignore
└── README.md
```

---

## 验证结果

在与官方前向传播使用完全相同输入的前提下，对手写 LBS 实现进行了逐顶点数值对比：

```
manual_vs_official_mean_abs_error: 0.0000000000
manual_vs_official_max_abs_error:  0.0000000000
```

两项误差均精确为 0（`float32` 精度范围内），说明手写流水线与官方 `smplx` 实现在数值上**完全一致**。

### 模型基础信息

| 属性 | 数值 |
|---|---|
| 顶点数 | 6890 |
| 面片数 | 13776 |
| 关节数 | 24 |
| 形状参数（`β`）维度 | 10 |

---

## 关键设计决策

1. **`_ChumpyArrayShim`**—— 一个轻量级 pickle 兼容层，允许在**不安装**已废弃的 `chumpy` 包的情况下加载旧版 SMPL `.pkl` 文件。通过注册一个伪造的 `chumpy.ch` 模块，从被 pickle 对象的内部状态重建 numpy 数组来实现。

2. **`prepare_posedirs`**—— 某些 SMPL 变体可能以转置形式存储 `posedirs`（如 `[V×3, P]` 而非 `[P, V×3]`）。该辅助函数自动检测形状并在必要时进行转置，确保与官方 `lbs()` 的约定一致。

3. **坐标系转换**—— SMPL 使用 Y-up 坐标系，而 matplotlib 使用 Z-up。辅助函数 `smpl_to_plot_coords` 交换 Y 轴与 Z 轴，保证 3D 渲染方向正确。

---

## 输出可视化说明

| 文件 | 说明 |
|---|---|
| `stage_a_template_weights.png` <img width="1077" height="1119" alt="stage_a_template_weights" src="https://github.com/user-attachments/assets/8a4a2f37-01b1-4db0-a5b8-fb09933d4ebc" />
| T-pose 网格 + 单个关节的蒙皮权重热力图（颜色越亮表示该关节对该区域影响越强） |
| `all_joint_weights.png`<img width="1517" height="1559" alt="all_joint_weights" src="https://github.com/user-attachments/assets/d44d3b42-7179-4155-8516-1b72e596b2bb" />
 | 每个面片按主导关节着色 —— 展示全局关节影响区域分布 |
| `stage_b_shaped_joints.png`<img width="1077" height="1119" alt="stage_b_shaped_joints" src="https://github.com/user-attachments/assets/c139d170-e0ee-4892-b0de-4193dc14126f" />
 | 通过 `β` 改变体型后的网格，白色圆点为回归出的关节点位置 |
| `stage_c_pose_offsets.png`<img width="1077" height="1119" alt="stage_c_pose_offsets" src="https://github.com/user-attachments/assets/cf9a7f5c-f4dc-4495-8814-f8b3c4317bec" />
 | 按 `|pose_offsets|` 着色的姿态校正网格 —— 热色区域对应弯曲关节附近的额外几何修正 |
| `stage_d_lbs_result.png`<img width="1077" height="1119" alt="stage_d_lbs_result" src="https://github.com/user-attachments/assets/354255bc-13d3-43e6-a414-549e9444c57a" />
 | 目标姿态下的最终蒙皮网格与变换后的关节 |
| `comparison_grid.png`<img width="2417" height="2176" alt="comparison_grid" src="https://github.com/user-attachments/assets/0f100af0-8f24-4ef5-9b49-6c5f0eb7a2e4" />
 | 四阶段 2×2 并排对比图，直观展示流水线演变过程 |

---

## 实验任务对应关系

本实验包含 7 项任务，均已在 `run_lbs_lab.py` 中实现：

| 任务 | 内容 | 对应输出 |
|---|---|---|
| 任务 1 | 加载 SMPL 并输出基础信息 | 控制台打印 + `summary.txt` |
| 任务 2 | 可视化模板网格与蒙皮权重 | `stage_a_template_weights.png`、`all_joint_weights.png` |
| 任务 3 | 可视化形状校正与关节回归 | `stage_b_shaped_joints.png` |
| 任务 4 | 可视化姿态校正 `B_P(θ)` | `stage_c_pose_offsets.png` |
| 任务 5 | 可视化完整 LBS 结果 | `stage_d_lbs_result.png` |
| 任务 6 | 生成四阶段总对比图 | `comparison_grid.png` |
| 任务 7 | 手写 LBS 与官方前向一致性验证 | `summary.txt` |

---
