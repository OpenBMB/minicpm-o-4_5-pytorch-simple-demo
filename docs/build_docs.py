#!/usr/bin/env python3
"""
Markdown -> HTML 文档构建脚本（支持 2 层导航）

用法:  python docs/build_docs.py
输出:  docs/site/
"""

import os, sys, re, shutil
from pathlib import Path

try:
    import markdown
    from markdown.extensions.toc import TocExtension
except ImportError:
    print("缺少 markdown 包，正在安装...")
    os.system(f"{sys.executable} -m pip install markdown")
    import markdown
    from markdown.extensions.toc import TocExtension

DOCS_DIR = Path(__file__).parent
SITE_DIR = DOCS_DIR / "site"

# (md路径, 显示标题, 分组名或None)
PAGES = [
    ("index.md",                   "项目概述",      None),
    ("architecture.md",            "系统架构",      None),
    ("gateway.md",                 "Gateway 模块",  None),
    ("worker.md",                  "Worker 模块",   None),
    ("core.md",                    "Core 模块",     None),
    ("model.md",                   "模型模块",      None),
    ("compile.md",                 "torch.compile", None),
    ("frontend/index.md",          "前端概述",      "前端模块"),
    ("frontend/pages.md",          "页面与路由",    "前端模块"),
    ("frontend/audio.md",          "音频处理",      "前端模块"),
    ("frontend/duplex-session.md", "双工会话",      "前端模块"),
    ("frontend/components.md",     "UI 组件",       "前端模块"),
    ("api.md",                     "API 参考",      None),
    ("deployment.md",              "配置与部署",    None),
]


def convert_mermaid_blocks(md_text):
    def _r(m):
        return f'<pre class="mermaid">\n{m.group(1).strip()}\n</pre>'
    return re.sub(r'```mermaid\s*\n(.*?)```', _r, md_text, flags=re.DOTALL)


def _html_path(md):
    return md.replace(".md", ".html")


def build_nav(current_file):
    items = []
    cur_group = None
    children = []
    group_active = False
    depth = current_file.count("/")
    pfx = "../" * depth

    def flush():
        nonlocal cur_group, children, group_active
        if cur_group is None:
            return
        items.append(f'    <li class="nav-group">')
        items.append(f'      <span class="nav-group-header">{cur_group}</span>')
        items.append(f'      <ul class="nav-group-children">')
        items.extend(children)
        items.append(f'      </ul>')
        items.append(f'    </li>')
        cur_group = None
        children = []
        group_active = False

    for md, label, group in PAGES:
        href = f"{pfx}{_html_path(md)}"
        active = ' class="active"' if md == current_file else ""
        link = f'<a href="{href}"{active}>{label}</a>'
        if group is None:
            flush()
            items.append(f'    <li>{link}</li>')
        else:
            if group != cur_group:
                flush()
                cur_group = group
            if md == current_file:
                group_active = True
            children.append(f'        <li>{link}</li>')
    flush()
    return "\n".join(items)


def build_page(md_path, title):
    fp = DOCS_DIR / md_path
    if not fp.exists():
        print(f"  [跳过] {md_path}")
        return ""
    text = convert_mermaid_blocks(fp.read_text(encoding="utf-8"))
    body = markdown.markdown(text, extensions=[
        "tables", "fenced_code", "codehilite",
        TocExtension(permalink=False, toc_depth=3),
    ])
    nav = build_nav(md_path)
    return TEMPLATE.format(title=title, css=CSS, nav_items=nav, body=body)


CSS = r"""
* { margin:0; padding:0; box-sizing:border-box; }
:root {
  --sidebar-w: 260px;
  --bg: #fff; --bg-side: #f8f9fa; --bg-code: #f6f8fa;
  --c1: #24292f; --c2: #57606a;
  --border: #d0d7de; --accent: #0969da; --nav-active: #ddf4ff;
}
body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,sans-serif; color:var(--c1); line-height:1.6; background:var(--bg); }
.sidebar-toggle { position:fixed; top:12px; left:12px; z-index:1001; background:var(--bg-side); border:1px solid var(--border); border-radius:6px; padding:6px 10px; font-size:18px; cursor:pointer; display:none; }
.sidebar { position:fixed; top:0; left:0; width:var(--sidebar-w); height:100vh; overflow-y:auto; background:var(--bg-side); border-right:1px solid var(--border); padding:20px 0; z-index:1000; transition:transform .3s; }
.sidebar-header { padding:0 20px 16px; border-bottom:1px solid var(--border); margin-bottom:8px; }
.sidebar-header h2 { font-size:16px; font-weight:600; }
.sidebar-subtitle { font-size:12px; color:var(--c2); }

.nav-list { list-style:none; padding:0 8px; }
.nav-list > li > a { display:block; padding:7px 12px; color:var(--c2); text-decoration:none; font-size:14px; border-radius:6px; transition:background .15s,color .15s; }
.nav-list > li > a:hover { background:#e8e8e8; color:var(--c1); }
.nav-list > li > a.active { background:var(--nav-active); color:var(--accent); font-weight:600; }

.nav-group { margin-top:4px; }
.nav-group-header { display:flex; align-items:center; gap:6px; padding:8px 12px; color:var(--c1); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; cursor:pointer; border-radius:6px; user-select:none; transition:background .15s; }
.nav-group-header:hover { background:#eaeef2; }
.nav-group-header::before { content:""; display:inline-block; width:0; height:0; border-left:5px solid var(--c2); border-top:3.5px solid transparent; border-bottom:3.5px solid transparent; transition:transform .2s; transform:rotate(90deg); flex-shrink:0; }
.nav-group.collapsed .nav-group-header::before { transform:rotate(0); }
.nav-group-children { list-style:none; margin:0 0 4px 19px; padding:3px 0 3px 13px; border-left:2px solid #e1e4e8; overflow:hidden; max-height:500px; transition:max-height .25s ease,opacity .2s ease,padding .2s ease; opacity:1; }
.nav-group.collapsed .nav-group-children { max-height:0; opacity:0; padding:0 0 0 13px; }
.nav-group-children li a { display:block; padding:4px 10px; font-size:13px; color:var(--c2); text-decoration:none; border-radius:4px; transition:background .15s,color .15s; }
.nav-group-children li a:hover { background:#e8e8e8; color:var(--c1); }
.nav-group-children li a.active { background:var(--nav-active); color:var(--accent); font-weight:600; }

.content { margin-left:var(--sidebar-w); max-width:900px; padding:40px 48px; }
article h1 { font-size:28px; font-weight:600; padding-bottom:10px; border-bottom:1px solid var(--border); margin-bottom:20px; }
article h2 { font-size:22px; font-weight:600; margin-top:32px; margin-bottom:12px; padding-bottom:6px; border-bottom:1px solid #eaecef; }
article h3 { font-size:18px; font-weight:600; margin-top:24px; margin-bottom:10px; }
article h4 { font-size:15px; font-weight:600; margin-top:20px; margin-bottom:8px; }
article p { margin-bottom:14px; }
article ul,article ol { margin-bottom:14px; padding-left:24px; }
article li { margin-bottom:4px; }
article a { color:var(--accent); text-decoration:none; }
article a:hover { text-decoration:underline; }
article code { background:var(--bg-code); padding:2px 6px; border-radius:4px; font-size:13px; font-family:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace; }
article pre { background:var(--bg-code); border:1px solid var(--border); border-radius:6px; padding:16px; overflow-x:auto; margin-bottom:16px; line-height:1.5; }
article pre code { background:none; padding:0; font-size:13px; }
article table { width:100%; border-collapse:collapse; margin-bottom:16px; font-size:14px; }
article th,article td { border:1px solid var(--border); padding:8px 12px; text-align:left; }
article th { background:var(--bg-code); font-weight:600; }
article tr:nth-child(even) { background:#f8f9fa; }
article hr { border:none; border-top:1px solid var(--border); margin:28px 0; }
article blockquote { border-left:4px solid var(--accent); padding:8px 16px; margin:0 0 16px; color:var(--c2); background:#f8f9fa; border-radius:0 6px 6px 0; }
article .mermaid { text-align:center; margin:20px 0; }
footer { margin-top:60px; padding-top:16px; border-top:1px solid var(--border); color:var(--c2); font-size:13px; }
@media(max-width:768px) {
  .sidebar { transform:translateX(-100%); }
  .sidebar.open { transform:translateX(0); box-shadow:2px 0 8px rgba(0,0,0,.15); }
  .sidebar-toggle { display:block; }
  .content { margin-left:0; padding:50px 20px 40px; }
  .content.shifted { margin-left:var(--sidebar-w); }
}
"""

TEMPLATE = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{title} - MiniCPM-o 4.5 文档</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
<style>{css}</style>
</head>
<body>
<button class="sidebar-toggle" onclick="toggleSidebar()" aria-label="Toggle sidebar">&#9776;</button>
<nav class="sidebar" id="sidebar">
  <div class="sidebar-header">
    <h2>MiniCPM-o 4.5</h2>
    <span class="sidebar-subtitle">项目文档</span>
  </div>
  <ul class="nav-list">
{nav_items}
  </ul>
</nav>
<main class="content" id="content">
  <article>{body}</article>
  <footer><p>MiniCPM-o 4.5 PyTorch Simple Demo &mdash; 由 build_docs.py 自动生成</p></footer>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/bash.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/json.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/javascript.min.js"></script>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({{startOnLoad:true,theme:'default',securityLevel:'loose'}});
</script>
<script>
hljs.highlightAll();
function toggleSidebar(){{document.getElementById('sidebar').classList.toggle('open');document.getElementById('content').classList.toggle('shifted');}}
document.querySelectorAll('.nav-group-header').forEach(function(h){{h.addEventListener('click',function(){{this.parentElement.classList.toggle('collapsed');}});}});
document.addEventListener('click',function(e){{var s=document.getElementById('sidebar'),t=document.querySelector('.sidebar-toggle');if(window.innerWidth<=768&&s.classList.contains('open')&&!s.contains(e.target)&&!t.contains(e.target)){{s.classList.remove('open');document.getElementById('content').classList.remove('shifted');}}}});
</script>
</body>
</html>
"""


def main():
    if SITE_DIR.exists():
        shutil.rmtree(SITE_DIR)
    SITE_DIR.mkdir(parents=True)
    print(f"构建: {SITE_DIR}\n")
    n = 0
    for md, title, _g in PAGES:
        html = build_page(md, title)
        if not html:
            continue
        out = SITE_DIR / _html_path(md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        print(f"  [OK] {_html_path(md)}")
        n += 1
    print(f"\n完成! {n} 页 -> {SITE_DIR}")
    print(f"入口: file://{SITE_DIR.resolve()}/index.html")

if __name__ == "__main__":
    main()
