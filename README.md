# Blog System - Markdown to HTML Compiler

A simple Python-based static blog system that compiles Markdown files into HTML pages.

## Quick Start

### 1. Install Dependencies

**Using uv (recommended):**
```bash
# No installation needed! uv handles dependencies automatically
```

**Or using conda/pip:**
```bash
pip install markdown jinja2 pyyaml
```

**Dependencies (see pyproject.toml):**
- markdown>=3.5
- Jinja2>=3.1
- PyYAML>=6.0

### 2. Write a Blog Post

Create a new file in `posts/` directory with the naming format: `YYYY-MM-DD-title.md`

Example: `posts/2026-01-29-my-first-post.md`

```markdown
---
title: "My First Blog Post"
date: 2026-01-29
description: "A brief summary that appears on the blog list page"
math: true  # Set to true if you need KaTeX for math equations
---

# Introduction

This is my first blog post. Here's some **bold text** and *italic text*.

## Math Support (if math: true)

Inline math: $E = mc^2$

Display math:
$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

## Code Blocks

```python
def hello_world():
    print("Hello, World!")
```

## Lists

- Item 1
- Item 2
- Item 3
```

### 3. Compile Markdown to HTML

Run the build script:

**Using uv (recommended):**
```bash
uv run build_blog.py
```

**Or using python directly:**
```bash
python build_blog.py
```

**What this does:**
1. Reads all `.md` files from `posts/` directory
2. Parses YAML frontmatter (title, date, description, etc.)
3. Converts Markdown body to HTML using Python's `markdown` library
4. Applies Jinja2 templates (`templates/post.html` for individual posts)
5. Generates HTML files in `blog/` directory
6. Updates the blog index page (`blog/index.html`) with all posts sorted by date

**Output:**
```
Processing: posts/2026-01-29-my-first-post.md
  -> blog/2026-01-29-my-first-post.html
Built: blog/index.html

Done! Built 1 post(s).
```

## How the Compilation Works

### Process Flow

```
Markdown Files (posts/*.md)
         ↓
    [build_blog.py]
         ↓
   Parse Frontmatter (YAML) → Extract: title, date, description, math
         ↓
   Convert Markdown → HTML (using markdown library)
         ↓
   Apply Jinja2 Template (templates/post.html)
         ↓
   Generate HTML Files (blog/*.html)
         ↓
   Update Blog Index (blog/index.html)
```

### Supported Markdown Features

The Python `markdown` library with these extensions:
- **`fenced_code`** - Code blocks with \`\`\` syntax
- **`tables`** - Markdown tables
- **`footnotes`** - Academic footnotes `[^1]`
- **`attr_list`** - Custom attributes `{: .class}`
- **`toc`** - Table of contents generation

### Templates

**`templates/post.html`** - Individual blog post template
- Includes KaTeX scripts (if `math: true`)
- Navigation: Home | Blog
- Post header with title and date
- Content area
- Footer with "Back to all posts" link

**`templates/blog_index.html`** - Blog list page template
- Shows all posts sorted by date (newest first)
- Displays: date, title, description for each post

### File Structure

```
.
├── index.html              # Main homepage (Blog link added)
│
├── posts/                  # SOURCE: Your markdown files
│   ├── 2026-01-29-hello-world.md
│   └── 2026-02-15-another-post.md
│
├── blog/                   # OUTPUT: Generated HTML files
│   ├── index.html         # Blog list page (auto-generated)
│   ├── 2026-01-29-hello-world.html
│   └── 2026-02-15-another-post.html
│
├── templates/              # Jinja2 templates
│   ├── post.html          # Single post template
│   └── blog_index.html    # Blog list template
│
├── css/
│   └── blog.css           # Blog styles (matches main site)
│
├── build_blog.py          # Build script
└── requirements.txt       # Python dependencies
```

## Tips

### Re-compile After Changes

After editing any markdown file or template, re-run:
```bash
uv run build_blog.py
```

### Date Format

Dates in frontmatter must be `YYYY-MM-DD` format:
```yaml
date: 2026-01-29  # Correct
date: Jan 29, 2026  # Wrong
```

### Preview Locally

Open the generated HTML files directly in your browser:
```bash
open blog/index.html
```

Or use a simple HTTP server:
```bash
python -m http.server 8000
# Visit: http://localhost:8000
```

## Deployment

After building, commit and push the generated files:

```bash
uv run build_blog.py          # Compile
git add blog/ posts/ templates/ css/
git commit -m "Add new blog post"
git push
```

The site will be live at https://sunchangsheng.com
