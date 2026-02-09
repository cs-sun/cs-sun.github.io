#!/usr/bin/env python3
"""
Blog build script for sunchangsheng.com
Converts markdown posts to HTML and generates blog index.

Usage:
    python build_blog.py
"""

import os
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

# TOC extension with configuration
MD_EXTENSION_CONFIGS = {
    'toc': {
        'title': '',  # Will be replaced with post title
        'toc_depth': '1-3',  # Include h1-h3 in TOC
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
    """Format date as '29 January 2026' (DMY format).

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
    return dt.strftime('%d %B %Y')


def convert_post(md_path: Path) -> dict:
    """Convert a single markdown file to HTML.

    Args:
        md_path: Path to the markdown file

    Returns:
        Dictionary with post metadata and HTML content
    """
    content = md_path.read_text(encoding='utf-8')
    metadata, body = parse_frontmatter(content)
    
    # Get post title for TOC
    post_title = metadata.get('title', 'Untitled')

    # Initialize markdown converter with dynamic title
    md_config = MD_EXTENSION_CONFIGS.copy()
    md_config['toc'] = md_config['toc'].copy()
    md_config['toc']['title'] = post_title
    
    md = markdown.Markdown(extensions=MD_EXTENSIONS + ['toc'], extension_configs=md_config)
    html_body = md.convert(body)
    
    # Wrap TOC in collapsible details/summary
    if '<div class="toc">' in html_body:
        html_body = html_body.replace(
            '<div class="toc">',
            '<div class="toc"><details><summary>  Table of Contents</summary>'
        )
        html_body = html_body.replace(
            '</div>',
            '</details></div>',
            1  # Only replace the first closing div (the TOC's closing div)
        )

    # Extract date from filename (YYYY-MM-DD-title.md)
    date_match = re.match(r'(\d{4}-\d{2}-\d{2})', md_path.stem)
    if date_match:
        date_str = date_match.group(1)
        metadata.setdefault('date', date_str)

    # Generate output filename
    output_name = md_path.stem + '.html'

    return {
        'title': metadata.get('title', 'Untitled'),
        'date': str(metadata.get('date', '')),
        'date_formatted': format_date(metadata.get('date')),
        'author': metadata.get('author', 'Sun Changsheng'),
        'description': metadata.get('description', ''),
        'tags': metadata.get('tags', []),
        'math': metadata.get('math', False),
        'content': html_body,
        'filename': output_name,
        'url': f'/blog/{output_name}',
    }


def build_blog():
    """Main build function. Converts all posts and generates index."""
    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    post_template = env.get_template('post.html')
    index_template = env.get_template('blog_index.html')

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Check if posts directory exists
    if not POSTS_DIR.exists():
        print(f"Posts directory '{POSTS_DIR}' does not exist.")
        print("Creating empty blog index...")
        index_html = index_template.render(posts=[])
        (OUTPUT_DIR / 'index.html').write_text(index_html, encoding='utf-8')
        print(f"Built: {OUTPUT_DIR / 'index.html'}")
        return

    # Find all markdown files
    md_files = sorted(POSTS_DIR.glob('*.md'), reverse=True)

    if not md_files:
        print("No markdown files found in posts directory.")
        print("Creating empty blog index...")
        index_html = index_template.render(posts=[])
        (OUTPUT_DIR / 'index.html').write_text(index_html, encoding='utf-8')
        print(f"Built: {OUTPUT_DIR / 'index.html'}")
        return

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
