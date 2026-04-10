import re

# ─── Fix 1: generate_plots.py canvas sizing ───────────────────────────────────
with open('generate_plots.py', 'r', encoding='utf-8') as f:
    plots = f.read()

plots = plots.replace(
    'sns.set_theme(style="whitegrid")\nplt.rcParams["font.family"] = "sans-serif"',
    'sns.set_theme(style="darkgrid", context="notebook", font_scale=1.0)\nplt.rcParams["font.family"] = "sans-serif"'
)
plots = plots.replace("sns.set_palette", "# sns.set_palette")  # remove if exists from before
plots = plots.replace('figsize=(10, 5)', 'figsize=(14, 6)')
plots = plots.replace('figsize=(9, 5)',  'figsize=(14, 6)')
# ensure tight_layout before each savefig
plots = plots.replace(
    'plt.savefig("docs/curriculum_scaling.png"',
    'fig.tight_layout(rect=[0, 0, 0.90, 1])\nplt.savefig("docs/curriculum_scaling.png"'
)
plots = plots.replace(
    'plt.savefig("docs/static_vs_dynamic.png"',
    'plt.tight_layout()\nplt.savefig("docs/static_vs_dynamic.png"'
)

with open('generate_plots.py', 'w', encoding='utf-8') as f:
    f.write(plots)
print("generate_plots.py patched.")

# ─── Fix 2: README.md tables ──────────────────────────────────────────────────
with open('README.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Table A (Episode Bundles — detailed with Confidence) — insert after row 12
table_a_anchor = "| 12 | **Rainy Street** | raincoat | bus, car, person, umbrella | 0.38 | request_rescan |"
table_a_insert = """\n| 13 | **Blizzard Whiteout** | truck | person, car, stop sign | 0.22 | discard |
| 14 | **Sensor Glare** | motorcycle | ambulance, person, traffic light | 0.46 | request_rescan |"""

if "Blizzard Whiteout" not in text:
    text = text.replace(table_a_anchor, table_a_anchor + table_a_insert)
    print("Table A patched (Episode Bundles detailed).")
else:
    print("Table A already has Blizzard Whiteout — skipped.")

# Table B (Notable Objects under Why 3-step) — insert after row 12
table_b_anchor = "| 12 | **Rainy Street** | umbrella, car, bus, raincoat |"
table_b_insert = """\n| 13 | **Blizzard Whiteout** | truck, person, car, stop sign |
| 14 | **Sensor Glare** | motorcycle, ambulance, person, traffic light |"""

if table_b_anchor in text and "Blizzard Whiteout" not in text.split(table_b_anchor)[1][:200]:
    text = text.replace(table_b_anchor, table_b_anchor + table_b_insert)
    print("Table B patched (Notable Objects).")
else:
    print("Table B already correct or anchor not found.")

# Fix stale counts
text = text.replace("All 12 bundles are hardcoded", "All 14 bundles are hardcoded")
text = text.replace("The **12 themed bundles**", "The **14 themed bundles**")

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(text)
print("README.md patched.")
