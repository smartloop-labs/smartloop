"""Generate horizontal Orchestrator DAG diagram (Smartloop docs theme)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

BG      = "#0D1117"
FG      = "#E6EDF3"
MUTED   = "#8B949E"
ACCENT  = "#FF3478"
CARD    = "#161B22"
NODE    = "#14B8A6"
STEP    = "#3B82F6"
RESULT  = "#22C55E"
PROMPT  = "#F59E0B"

fig, ax = plt.subplots(figsize=(17, 6.6))
fig.patch.set_alpha(0)
ax.set_facecolor("none")
ax.set_xlim(0, 17)
ax.set_ylim(-0.6, 6.0)
ax.axis("off")


def box(cx, cy, w, h, text, color, fontsize=12):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.20",
        linewidth=1.6, edgecolor=color, facecolor="none",
    ))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=FG)


def circle(cx, cy, r, text, color, fontsize=11):
    ax.add_patch(Circle((cx, cy), r,
                        linewidth=1.6, edgecolor=color, facecolor="none"))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=FG)


def arrow(p1, p2, color=MUTED, curve=0.0, dashed=False, lw=1.6):
    style = (0, (5, 4)) if dashed else "-"
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=16,
        color=color, lw=lw, linestyle=style,
        connectionstyle=f"arc3,rad={curve}",
    ))


X_PROMPT, X_ORCH, X_NODES, X_SYNTH, X_RESULT = 1.4, 4.4, 8.5, 12.4, 15.4
Y = 3.0
LABEL_Y = 5.4


# Caption at bottom (centered with image)
ax.text(8.5, -0.35, "Fig: Orchestration Framework",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=FG, style="italic")

# Only DAG annotation
ax.text(X_NODES, LABEL_Y, "DAG", ha="center", va="center",
        fontsize=10, color=MUTED, style="italic", fontweight="bold")

# User prompt (circle)
circle(X_PROMPT, Y, 0.8, "Prompt", PROMPT, fontsize=11)

# Orchestrator
box(X_ORCH, Y, 2.4, 1.2, "Orchestrator", ACCENT, fontsize=13)

# Nodes (stacked)
node_ys = [4.4, 3.0, 1.6]
node_labels = ["Node A", "Node B", "Node C"]
for label, y in zip(node_labels, node_ys):
    box(X_NODES, y, 1.8, 0.8, label, NODE, fontsize=11)

# Synthesize
box(X_SYNTH, Y, 2.0, 1.2, "Synthesize", STEP, fontsize=12)

# Result
box(X_RESULT, Y, 1.6, 1.2, "Result", RESULT, fontsize=12)

# Prompt -> Orchestrator
arrow((X_PROMPT + 0.8, Y), (X_ORCH - 1.2, Y))

# Orchestrator -> each node
for y in node_ys:
    arrow((X_ORCH + 1.2, Y), (X_NODES - 0.9, y),
          curve=0.05 if y > Y else -0.05)

# Nodes -> Synthesize
for y in node_ys:
    arrow((X_NODES + 0.9, y), (X_SYNTH - 1.0, Y),
          curve=-0.05 if y > Y else 0.05)

# Synthesize -> Result
arrow((X_SYNTH + 1.0, Y), (X_RESULT - 0.8, Y))

# Feedback loop: Synthesize -> Orchestrator (dashed, arcs above)
fb = FancyArrowPatch(
    (X_SYNTH, Y + 0.6), (X_ORCH, Y + 0.6),
    arrowstyle="-|>", mutation_scale=15,
    color=ACCENT, lw=1.4, linestyle=(0, (5, 4)),
    connectionstyle="arc3,rad=0.35",
)
ax.add_patch(fb)

plt.tight_layout()
plt.savefig("/Users/mehfuzh/source/smartloop/img/orchestrator-dag.png",
            dpi=200, bbox_inches="tight", transparent=True)
print("Saved: img/orchestrator-dag.png")
