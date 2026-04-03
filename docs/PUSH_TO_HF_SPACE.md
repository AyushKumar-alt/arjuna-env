# Pushing to Hugging Face Space (`space` remote)

HF **rejects** pushes when `main`’s **history** contains certain **binary blobs** (e.g. PNGs under `docs/images/`), even if a later commit deletes them.

HF also validates **README** YAML: **`sdk`** must be exactly **`docker`** (lowercase), not `Docker`, and keys should match [Space metadata](https://huggingface.co/docs/hub/spaces-config-reference) (e.g. `title`, not `Title`).

## Recommended: one-commit snapshot for Space only

From repo root, with a clean tree on `main`:

```powershell
git checkout main
git pull github main

git checkout --orphan hf-space-publish
git add -A
git commit -m "Snapshot for HF Space (no legacy binary blobs in history)"
git push space hf-space-publish:main --force

git checkout main
git branch -D hf-space-publish
```

- **GitHub** `main` keeps full history (`git push github main` as usual).
- **Space** `main` is replaced by a **single** commit whose tree matches your working tree (no old PNG objects).

## Do not

- `git push space main` if `main` still has PNGs anywhere in ancestry (pre-receive hook will fail).

## README frontmatter (Docker Space)

```yaml
---
title: Your Space Title
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
```

Ensure **`sdk: docker`** is lowercase.
