from matplotlib.patheffects import withStroke

def draw_agent_ids(ax, world, info, frame_idx):
    """エージェントの位置に ID を描画するコールバック関数"""
    # print("draw_agent_ids called!")
    # print(ax, world)
    positions = [a.state.p_pos for a in world.agents]

    if not hasattr(ax, "_id_texts"):
        # print("初回描画")
        texts = []
        for i, (x, y) in enumerate(positions):
            t = ax.text(x, y, str(i), color="white", fontsize=9, ha="center", va="center",
                           path_effects=[withStroke(linewidth=2, foreground="black")], 
                           zorder=5)
            texts.append(t)
        ax._id_texts = texts
    else:
        # print("更新描画")
        for t, (x, y) in zip(ax._id_texts, positions):
            t.set_position((x, y))