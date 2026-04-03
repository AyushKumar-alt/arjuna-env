# `docs/images/`

This directory exists as a **documentation anchor** for the repository. It does **not** contain image binaries (no PNG, JPEG, GIF, or WebP files are committed here).

## Why there are no screenshots in Git

1. **Hugging Face Hub** may reject **Space** `git push` when the **history** of `main` includes certain **binary blobs** (common with PNGs committed in the past), even after those files are deleted in a later commit.
2. Keeping the Space repo **text-only** avoids fighting **Xet** / **Git LFS** for a few static images.
3. **Reviewers and judges** get a better experience by using the **live deployment** (same UI you would screenshot).

## What to use instead of static images

| Goal | Where |
|------|--------|
| Interactive **OpenEnv Playground** (Reset / Step / JSON) | **[Live `/web`](https://calpol500mg-arjuna-env.hf.space/web)** |
| **OpenAPI** (`/reset`, `/step`, schemas) | **[Live `/docs`](https://calpol500mg-arjuna-env.hf.space/docs)** |
| Space project page | **[HF Space](https://huggingface.co/spaces/Calpol500mg/arjuna-env)** |
| Source tree | **[GitHub](https://github.com/AyushKumar-alt/arjuna-env)** |

Run the stack **locally** for the same URLs on `http://127.0.0.1:7860` (see root **`README.md`**). The **`Dockerfile`** sets **`ENABLE_WEB_INTERFACE=true`** so **`/web`** is available in containers by default.

## Related docs

- **`docs/PUSH_TO_HF_SPACE.md`** — how to push to the **Space** remote when you must avoid binary history (orphan snapshot workflow).
- Root **`README.md`** — full setup, tasks, grading, troubleshooting.

## If you ever need images in version control

- Prefer **[Hub Xet](https://huggingface.co/docs/hub/xet)** for binaries on HF, **or**
- Host assets outside this repo (release attachments, wiki, external CDN) and link from documentation **without** committing binaries to **`main`**.
