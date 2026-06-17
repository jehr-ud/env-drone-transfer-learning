def build_obs_dict(obs):
    return {
        "goal_rel": obs[0:3],
        "vel_norm": obs[3:6],
        "rp": obs[6:8],
        "yaw": obs[8:10],
        "ang_vel_norm": obs[10:13],
        "alignment": obs[13],
        "dist": obs[14],
        "rel_vec": obs[15:18],
        "dist_norm": obs[18],
        "size_norm": obs[19],
    }