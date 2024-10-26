# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

pip install gradio>=3.50.2
"""
import argparse
from threading import Thread
from PIL import Image
import gradio as gr
import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)
from rag import Rag

from template import get_conv_template

from detect import detect
from cut import cut_image
from predict import predict

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='auto', type=str)
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--template_name', default="vicuna", type=str,
                    help="Prompt template name, eg: alpaca, vicuna, baichuan2, chatglm2 etc.")
parser.add_argument('--system_prompt', default="", type=str)
parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
parser.add_argument('--share', action='store_true', help='Share gradio')
parser.add_argument('--port', default=8080, type=int, help='Port of gradio demo')
parser.add_argument('--weights', nargs='+', type=str, default='./runs/train/skincancerdetect/weights/best.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.15, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
args = parser.parse_args()


MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

def image_predict(image)->list:
    # return ['actinic keratosis',0.97]
    ##TOBE FILLED BY DONGWENCHAO
    img_detect_result=detect(args,image)
    images=cut_image(img_detect_result)
    return predict(images)

def rag_message(rag,prompt,image_result)->str:
    rag_result=rag.query_index(f"What are the treatment options for {image_result[0]}")## TO BE FILLED BY ZHOUWENBO
    # prompt_rag_result=[rag.query_index(prompt) for name in image_result]
    if len(image_result) == 0:
        return [""
    ,"A medical report shows that the patient is very healthy, please answer patients problem: "+prompt]
    else :
        return [""
    ,f"""A medical report shows that the patient has a {image_result[1]*100}% chance of suffering from {image_result[0]}.
    you know that expertise on the first disease includes:{rag_result[0]}
    You should combine the above information, please answer patients problem: 
        """+prompt]
    # At the same time, tell the patient that I am just an AI assistant doctor, and this result is for reference only. 
    # Please do not fully adopt this result without the approval of a regular hospital or licensed physician.
def main():
    rag=Rag()
    load_type = 'auto'
    if torch.cuda.is_available() and not args.only_cpu:
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    base_model = model_class.from_pretrained(
        args.base_model,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
        offload_folder="./offload"
    )
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("loaded lora model")
    else:
        model = base_model
    if device == torch.device('cpu'):
        model.float()
    model.eval()
    prompt_template = get_conv_template(args.template_name)
    system_prompt = args.system_prompt
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str
    
    def predict(message,history, image):
        """Generate answer from prompt with GPT and stream the output"""
        if image is not None:
            image_result=image_predict(image=image)
            message=rag_message(rag,message,image_result)
            history_messages = history + [[message[1], ""]]
        else:
            history_messages = history + [[prompt, ""]]
        print(history_messages)
        prompt = prompt_template.get_prompt(messages=history_messages, system_prompt=system_prompt)
        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        input_ids = tokenizer(prompt).input_ids
        context_len = 4096
        max_new_tokens = 1024
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        generation_kwargs = dict(
            input_ids=torch.as_tensor([input_ids]).to(device),
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.0,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_message = ""
        for new_token in streamer:
            if new_token != stop_str:
                partial_message += new_token
                yield partial_message

    gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(),
        textbox=gr.Textbox(placeholder="Ask me question", lines=4, scale=9),
        additional_inputs=[gr.Image(label="upload your photos", type="pil")],
        title="MedicalGPT",
        description="A caring AI medical advice expert",
        theme="soft",
    ).queue().launch(share=args.share, inbrowser=True, server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    main()
