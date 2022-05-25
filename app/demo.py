#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@File    :   demo.py
@Author  :   Daniel W 
@Version :   1.0
@Contact :   willhelmd@tamu.edu
'''
import os 
from pathlib import Path

import gradio as gr

# get basepath
basedir = Path(os.path.dirname(os.path.abspath(__file__)))


def demo_out(x1,x2):
    def f(x1,x2): 
        return x1 + x2
    
    return f(x1,x2)


demo = gr.Interface(
    fn=demo_out, 
    inputs=[gr.Number(label='x1'), gr.Number(label='x2')], 
    outputs=[gr.Number(label='y')]
)

