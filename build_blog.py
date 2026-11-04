#!/usr/bin/env python3
"""
Blog build script for sunchangsheng.com
Converts markdown posts to HTML and generates blog index.

Usage:
    python build_blog.py
"""

import re
from datetime import datetime
from pathlib import Path

import yaml
import markdown
from jinja2 import Environment, FileSystemLoader


# Configuration
POSTS_DIR = Path("posts")
OUTPUT_DIR = Path("blog")
TEMPLATES_DIR = Path("templates")

# Markdown extensions
MD_EXTENSIONS = [
    'fenced_code',   # ```code blocks```
    'tables',        # Markdown tables
    'footnotes',     # Academic footnotes [^1]
    'attr_list',     # Custom attributes
]

# The toc extension only gives headings their id anchors; posts have no TOC
MD_EXTENSION_CONFIGS = {
    'toc': {
        'permalink': False,  # No permalink symbols
    }
}


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body from markdown.

    Args:
        content: Raw markdown file content

    Returns:
        Tuple of (metadata dict, body string)
    """
    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(pattern, content, re.DOTALL)
    if match:
        metadata = yaml.safe_load(match.group(1))
        body = match.group(2)
        return metadata, body
    return {}, content


def format_date(date_str: str) -> str:
    """Format date as '9 Feb 2026' (day, abbreviated month, year).

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        Formatted date string
    """
    if not date_str:
        return ''
    if isinstance(date_str, datetime):
        dt = date_str
    else:
        dt = datetime.strptime(str(date_str), '%Y-%m-%d')
    return f'{dt.day} {dt:%b %Y}'


def protect_math(text: str) -> tuple[str, list[str]]:
    """Replace math blocks with placeholders to prevent Markdown mangling.

    Protects both display ($$...$$) and inline ($...$) math from being
    interpreted as emphasis or having backslashes stripped.

    Args:
        text: Markdown source text

    Returns:
        Tuple of (text with placeholders, list of original math blocks)
    """
    placeholders = []

    def _replace(m):
        placeholders.append(m.group(0))
        return f'\x00MATH{len(placeholders) - 1}\x00'

    # Display math first ($$...$$), then inline ($...$)
    text = re.sub(r'\$\$.*?\$\$', _replace, text, flags=re.DOTALL)
    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', _replace, text)
    return text, placeholders


def restore_math(html: str, placeholders: list[str]) -> str:
    """Restore math blocks from placeholders after Markdown conversion."""
    for i, original in enumerate(placeholders):
        html = html.replace(f'\x00MATH{i}\x00', original)
    return html


def number_sections(text: str) -> str:
    """Prefix "#" and "##" headings with section numbers (1, 1.1, ...).

    Skips code blocks and "References" headings.
    """
    lines = []
    in_code = False
    section = subsection = 0
    for line in text.split('\n'):
        if line.lstrip().startswith('```'):
            in_code = not in_code
        m = None if in_code else re.match(r'(#{1,2}) (.+)', line)
        if m and m.group(2).strip() != 'References':
            if len(m.group(1)) == 1:
                section += 1
                subsection = 0
                number = f'{section}'
            else:
                subsection += 1
                number = f'{section}.{subsection}'
            line = f'{m.group(1)} {number} {m.group(2)}'
        lines.append(line)
    return '\n'.join(lines)


def convert_post(md_path: Path) -> dict:
    """Convert a single markdown file to HTML.

    Args:
        md_path: Path to the markdown file

    Returns:
        Dictionary with post metadata and HTML content
    """
    content = md_path.read_text(encoding='utf-8')
    metadata, body = parse_frontmatter(content)

    body = number_sections(body)

    # Protect math blocks from Markdown processing
    body, math_placeholders = protect_math(body)

    md = markdown.Markdown(extensions=MD_EXTENSIONS + ['toc'],
                           extension_configs=MD_EXTENSION_CONFIGS)
    html_body = md.convert(body)

    # Restore math blocks
    html_body = restore_math(html_body, math_placeholders)

    # Posts use "#" for sections, but the post title is an <h2> (24px, the
    # size of the name on the About page): shift headings down two levels
    # (h1 -> h3, h2 -> h4) so sections sit below the title, as on research.html
    html_body = re.sub(r'<(/?)h([1-4])\b',
                       lambda m: f'<{m.group(1)}h{int(m.group(2)) + 2}', html_body)

    # Tables without CSS: ruled cells, inner padding, 90% wide and centred;
    # column alignment as an HTML attribute instead of an inline style
    html_body = html_body.replace(
        '<table>',
        '<table border="1" cellpadding="8" cellspacing="0" width="90%" align="center">')
    html_body = re.sub(r'<(t[hd]) style="text-align: (\w+);">', r'<\1 align="\2">', html_body)
    html_body = html_body.replace('<tr>', '<tr valign="top">')

    # Extract date from filename (YYYY-MM-DD-title.md)
    date_match = re.match(r'(\d{4}-\d{2}-\d{2})', md_path.stem)
    if date_match:
        date_str = date_match.group(1)
        metadata.setdefault('date', date_str)

    # Generate output filename
    output_name = md_path.stem + '.html'

    # Plain-text description for <meta>: drop Markdown and math markers
    plain_description = re.sub(r'[*_`$]', '', metadata.get('description', ''))

    return {
        'title': metadata.get('title', 'Untitled'),
        'date': str(metadata.get('date', '')),
        'date_formatted': format_date(metadata.get('date')),
        'description': plain_description,
        'math': metadata.get('math', False),
        'content': html_body,
        'filename': output_name,
    }


def build_blog():
    """Main build function. Converts all posts and generates index."""
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    post_template = env.get_template('post.html')
    index_template = env.get_template('blog_index.html')

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find all markdown files (an empty or missing posts/ gives an empty index)
    md_files = sorted(POSTS_DIR.glob('*.md'), reverse=True)

    # Convert all posts
    posts = []
    for md_file in md_files:
        print(f"Processing: {md_file}")
        post = convert_post(md_file)
        posts.append(post)

        # Render and write post HTML
        html = post_template.render(post=post)
        output_path = OUTPUT_DIR / post['filename']
        output_path.write_text(html, encoding='utf-8')
        print(f"  -> {output_path}")

    # Sort posts by date (newest first)
    posts.sort(key=lambda p: p['date'], reverse=True)

    # Generate blog index
    index_html = index_template.render(posts=posts)
    (OUTPUT_DIR / 'index.html').write_text(index_html, encoding='utf-8')
    print(f"Built: {OUTPUT_DIR / 'index.html'}")

    print(f"\nDone! Built {len(posts)} post(s).")


if __name__ == '__main__':
    build_blog()
