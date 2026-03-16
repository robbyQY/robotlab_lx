from PIL import Image, ImageDraw, ImageFont

W, H = 2200, 1300
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)


def font(size=28, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_box(x, y, w, h, text, fill, outline=(31, 58, 95), title=False):
    d.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=fill, outline=outline, width=3)
    f = font(34 if title else 25, bold=title)
    tw, th = d.multiline_textbbox((0, 0), text, font=f, align="center")[2:]
    tx = x + (w - tw) / 2
    ty = y + (h - th) / 2
    d.multiline_text((tx, ty), text, font=f, fill=(11, 19, 43), align="center", spacing=6)


def arrow(x1, y1, x2, y2, color=(31, 58, 95), width=5):
    d.line((x1, y1, x2, y2), fill=color, width=width)
    # simple triangular head
    import math

    ang = math.atan2(y2 - y1, x2 - x1)
    l = 18
    a1 = ang + 2.6
    a2 = ang - 2.6
    p1 = (x2 + l * math.cos(a1), y2 + l * math.sin(a1))
    p2 = (x2 + l * math.cos(a2), y2 + l * math.sin(a2))
    d.polygon([ (x2,y2), p1, p2 ], fill=color)


# Title
d.text((W // 2 - 540, 30), "Unitree B2 强化学习训练工作流（含策略结构）", font=font(52, bold=True), fill=(11, 19, 43))
d.text((W // 2 - 230, 95), "RobotLab + Isaac Lab + RSL-RL PPO", font=font(30), fill=(58, 80, 107))

# left column
xL, wL = 70, 520
ys = [180, 360, 540, 720, 900]
textsL = [
    "1) 训练入口\ntrain.py + Hydra\n--task / --agent",
    "2) 环境构建\nUnitreeB2RoughEnvCfg\nManagerBasedRLEnv",
    "3) 并行仿真\n4096 env × 24 steps\nIsaac Sim 物理推进",
    "4) 奖励与终止\n速度跟踪 + 能耗/接触惩罚\nepisode/time-out",
    "5) PPO 更新\n优势估计 + clipping\n迭代到 checkpoint",
]
for i, y in enumerate(ys):
    draw_box(xL, y, wL, 120, textsL[i], fill=(234, 244, 255), title=(i in [0, 4]))
    if i < len(ys) - 1:
        arrow(xL + wL // 2, y + 120, xL + wL // 2, ys[i + 1] - 14)

# middle
xM, wM = 760, 620
draw_box(xM, 180, wM, 120, "环境配置层\nrough terrain / events / curriculum", fill=(244, 241, 255), title=True)
draw_box(xM, 360, 295, 120, "观测\nbase ang vel / gravity\njoint pos-vel / actions", fill=(244, 241, 255))
draw_box(xM + 325, 360, 295, 120, "动作\n12D 关节位置\nhip:0.125 others:0.25", fill=(244, 241, 255))
draw_box(xM, 540, wM, 120, "命令生成\nUniformThresholdVelocityCommand\n(vx, vy, wz)", fill=(244, 241, 255))
draw_box(xM, 720, wM, 120, "机器人模型\nUNITREE_B2_CFG\nURDF + 电机参数", fill=(244, 241, 255))
arrow(xM + wM // 2, 300, xM + 150, 350, color=(109, 89, 122))
arrow(xM + wM // 2, 300, xM + 470, 350, color=(109, 89, 122))
arrow(xM + wM // 2, 480, xM + wM // 2, 530, color=(109, 89, 122))
arrow(xM + wM // 2, 660, xM + wM // 2, 710, color=(109, 89, 122))

# right
xR, wR = 1540, 590
draw_box(xR, 180, wR, 120, "PPO Actor-Critic 结构\n（共享输入，不共享参数）", fill=(255, 244, 232), title=True)
draw_box(xR, 360, 275, 120, "Actor MLP\n512→256→128\nELU", fill=(255, 244, 232))
draw_box(xR + 315, 360, 275, 120, "Critic MLP\n512→256→128\nELU", fill=(255, 244, 232))
draw_box(xR, 540, 275, 120, "动作分布\nGaussian\ninit std=1.0", fill=(255, 244, 232))
draw_box(xR + 315, 540, 275, 120, "值函数 V(s)\nvalue coef=1.0", fill=(255, 244, 232))
draw_box(xR, 720, wR, 120, "PPO 超参\nγ=0.99, λ=0.95, clip=0.2\nepoch=5, minibatch=4, lr=1e-3", fill=(255, 244, 232))
arrow(xR + wR // 2, 300, xR + 130, 350, color=(188, 108, 37))
arrow(xR + wR // 2, 300, xR + 460, 350, color=(188, 108, 37))
arrow(xR + 130, 480, xR + 130, 530, color=(188, 108, 37))
arrow(xR + 460, 480, xR + 460, 530, color=(188, 108, 37))
arrow(xR + 130, 660, xR + wR // 2, 710, color=(188, 108, 37))
arrow(xR + 460, 660, xR + wR // 2, 710, color=(188, 108, 37))

# cross-links
arrow(xL + wL, 420, xM - 20, 420, color=(109, 89, 122))
arrow(xM + wM + 20, 420, xR - 20, 420, color=(188, 108, 37))
arrow(xL + wL, 780, xM - 20, 780, color=(109, 89, 122))
arrow(xM + wM + 20, 780, xR - 20, 780, color=(188, 108, 37))
arrow(xR + wR // 2, 840, xL + wL // 2, 960, color=(31, 58, 95))

# footer
d.text((70, 1170), "左：训练闭环   中：B2环境结构   右：策略网络结构", font=font(26), fill=(58, 80, 107))
d.text((70, 1210), "依据：unitree_b2/rough_env_cfg.py, velocity_env_cfg.py, rsl_rl_ppo_cfg.py, train.py", font=font(22), fill=(92, 103, 125))

out = "docs/imgs/b2_rl_training_workflow.png"
img.save(out)
print(out)