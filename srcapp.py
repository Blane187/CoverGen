import gradio as gr
import sys
import os
import logging

now_dir = os.getcwd()
sys.path.append(now_dir)

# Tabs
from tabs.inference.inference import inference_tab
from tabs.download.download import download_tab




with gr.Blocks(theme='Hev832/EasyAndCool', title="Cover") as Applio:
    gr.Markdown("# CoverGen")
    
    with gr.Tab(("Inference")):
        inference_tab()


    with gr.Tab(("Download")):
        download_tab()

  
if __name__ == "__main__":
    Applio.launch(
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=6969,
    )
