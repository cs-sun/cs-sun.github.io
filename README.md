# sunchangsheng.com

Personal homepage, served by GitHub Pages from `master`.

Plain HTML with no stylesheet: browser defaults, plus HTML attributes
(`<font>`, `<table cellpadding>`, `<img width>`) where layout needs them.
Each page's body is wrapped in
`<font face="Helvetica Neue, Helvetica, Arial, sans-serif">`, because a
browser set to Chinese would otherwise default to PingFang for English text
too; this keeps one font everywhere (Chinese characters fall back to the
system sans-serif).

## Layout

```
index.html          about page (hand-written)
research.html       publications (hand-written)
misc.html           misc page (hand-written)
head.jpg            profile photo
favicon.png         browser-tab icon (64px, transparent background)
assets/             CV and paper PDFs linked from the pages

posts/*.md          blog sources (Markdown + YAML front matter)
templates/          Jinja2 templates for the blog
  post.html           one post
  blog_index.html     post list
  _katex.html         KaTeX include for math
build_blog.py       posts/*.md -> blog/*.html
blog/               generated; do not edit by hand
```

## Blog

Add `posts/YYYY-MM-DD-slug.md`:

```markdown
---
title: "Post title"
description: "One or two sentences for search results and link previews"
math: true   # load KaTeX
---

# First section
```

Use `#` for sections and `##` for subsections. The build numbers them
(1, 1.1, …; "References" is left unnumbered), shifts them to `<h3>`/`<h4>`
so they sit below the post title (`<h2>`), and gives tables borders and
padding through HTML attributes.

Build and preview:

```bash
uv run build_blog.py
python3 -m http.server 8000
```
