with open('generate_plots.py', 'r') as f:
    code = f.read()

# Enhance main seaborn context
code = code.replace(
    'sns.set_theme(style="whitegrid")\nplt.rcParams["font.family"] = "sans-serif"',
    'sns.set_theme(style="darkgrid", context="talk")\nplt.rcParams["font.family"] = "sans-serif"\nsns.set_palette("husl")'
)

# Enhance plot 1 layout explicitly
code = code.replace(
    'ax1.plot(episodes, rewards, color=color, marker=\'o\', linestyle=\'-\', linewidth=2, label="Agent Reward", alpha=0.8)',
    'ax1.plot(episodes, rewards, color="#1ABC9C", marker=\'o\', markersize=8, linestyle=\'-\', linewidth=3, label="Agent Reward", alpha=0.9)\nax1.fill_between(episodes, 0, rewards, color="#1ABC9C", alpha=0.1)'
)

code = code.replace('color = \'tab:blue\'', 'color = "#1ABC9C"')
code = code.replace('color = \'tab:orange\'', 'color = "#F39C12"')

code = code.replace(
    'ax2.step(episodes, difficulties, color=color, linewidth=3, where=\'mid\', label="AutoRL Difficulty Layer")',
    'ax2.step(episodes, difficulties, color=color, linewidth=4, where=\'mid\', alpha=0.8, label="AutoRL Difficulty Layer")'
)

# Enhance plot 2 colors
code = code.replace("color='crimson'", "color='#E74C3C'")
code = code.replace("color='forestgreen'", "color='#2ECC71'")
code = code.replace("color='lightgreen'", "color='#2ECC71'")

# Title font sizes
code = code.replace("fontsize=14", "fontsize=16")

with open('generate_plots.py', 'w') as f:
    f.write(code)

print("Updated generate_plots.py successfully.")
