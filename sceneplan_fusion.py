from layout_repair import read_layout, repair_layout


# Fusion：在原 HCD 最终融合处调用，k 从 1 开始。
def make_blender(plan, scheduled=False):
    boxes = []
    if scheduled:
        centers, half_sizes, pairs = read_layout(plan, skip_unknown_objects=True)
        for i in sorted({index // 2 for pair in pairs for index in pair}):
            section = slice(2 * i, 2 * i + 2)
            boxes.append((*centers[section], *half_sizes[section]))

    masks = {}

    def blend(z_full, z_smooth, k):
        if not isinstance(k, int) or not 1 <= k <= 20:
            raise ValueError("expected a denoising step k from 1 to 20")
        if (z_full.shape != z_smooth.shape or z_full.device != z_smooth.device
                or z_full.dtype != z_smooth.dtype or z_full.ndim != 4):
            raise ValueError("HCD latents must share BCHW shape, device, and dtype")
        if k >= 10 or not boxes:
            return 0.8 * z_full + 0.2 * z_smooth

        import torch

        height, width = z_full.shape[-2:]
        device = z_full.device
        key = (height, width, device)
        if key not in masks:
            u = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
            v = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
            mask = torch.zeros((height, width), device=device, dtype=torch.float32)
            for x, y, half_w, half_h in boxes:
                mx = (1 - ((u - x) / half_w).abs()).clamp_min(0)
                my = (1 - ((v - y) / half_h).abs()).clamp_min(0)
                mask = torch.maximum(mask, my[:, None] * mx[None, :])
            masks[key] = mask

        weight = 0.20 + 0.20 * masks[key] * ((10 - k) / 9)
        weight = weight.to(z_full.dtype)
        return (1 - weight) * z_full + weight * z_smooth

    return blend


# Matched conditions：同一份 cached MCP plan 生成四个对照，只做一次 joint repair。
def prepare_conditions(plan):
    repaired, status = repair_layout(plan)
    variants = {
        "mccd": (plan, False),
        "joint_repair": (repaired, False),
        "scheduled_fusion": (plan, True),
        "sceneplan": (repaired, True),
    }
    conditions = {
        name: (layout, make_blender(layout, scheduled))
        for name, (layout, scheduled) in variants.items()
    }
    return conditions, status
