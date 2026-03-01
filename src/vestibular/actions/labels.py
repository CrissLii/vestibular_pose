ACTION_LABELS_ZH = {
    "wheelbarrow_walk": "小推车",
    "forward_roll": "前滚翻",
    "head_up_prone": "抬头向上",
    "jump_in_place": "原地向上跳跃",
    "run_straight": "直线加速跑",
    "spin_in_place": "原地旋转",
    "unknown": "未识别",
}

def zh(action_id: str) -> str:
    return ACTION_LABELS_ZH.get(action_id, action_id)
