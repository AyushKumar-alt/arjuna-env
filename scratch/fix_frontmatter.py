with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

frontmatter = """---
title: Arjuna Perception Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

"""

# Only prepend if frontmatter is missing
if not content.startswith('---'):
    content = frontmatter + content
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Frontmatter restored successfully.")
else:
    print("Frontmatter already present, no change needed.")
