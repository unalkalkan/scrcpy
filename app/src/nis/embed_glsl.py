#!/usr/bin/env python3
"""Embeds a GLSL source file as a C string: embed_glsl.py INPUT OUTPUT NAME"""
import sys

src, dst, name = sys.argv[1:4]
with open(src, encoding='utf-8') as f:
    lines = f.read().splitlines()
with open(dst, 'w', encoding='utf-8') as f:
    f.write('// Generated from %s by embed_glsl.py, do not edit\n' % src.split('/')[-1])
    f.write('static const char %s[] =\n' % name)
    for line in lines:
        f.write('    "%s\\n"\n' % line.replace('\\', '\\\\').replace('"', '\\"'))
    f.write(';\n')
