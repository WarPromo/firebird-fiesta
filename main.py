
import io
import requests
import base64
import json
import sys
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from ultralytics import YOLO
import cv2
import time
import random;
import threading;

import pyautogui

def noise_pixel(im, x, y):
    im.putpixel((x,y), (rand_pixel(), rand_pixel(), rand_pixel()))

def rand_pixel():
    return int(random.random()*255);

generated_image = np.zeros(921600);

stop_loop = False;


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-skip_diffusion', help="If Stable diffusion should be done at all")

parser.add_argument('-url', type=str, help='The URL for the host of stable diffusion')

parser.add_argument('-prompt', type=str, help='The prompt that will be used for stable diffusion')

parser.add_argument('-negative_prompt', type=str, help='The negative prompt that will be used for stable diffusion')

args = parser.parse_args()

print("NOTE! To exit program, simply move your cursor to the top left of your screen")

if(args.skip_diffusion == None):
    if(args.url == None):
        print("ARG ERROR: Provide host URL")
        quit();

    if(args.prompt == None):
        print("ARG ERROR: Provide prompt")
        quit();

    if(args.negative_prompt == None):
        args.negative_prompt = "";


skip_diffusion = args.skip_diffusion != None;

def image_generator():

    global generated_image;
    model = YOLO('yolov8n-face.pt')
    cam = cv2.VideoCapture(0)
    requestURL = args.url;

    while True:

        if stop_loop:
            break;

        ret, frame = cam.read()

        print("Getting face coordinates...")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        source_img = frame.convert("RGB")
        results = model.predict(source=source_img)

        paddingmask = 1.5;
        padding = 5;

        maxsize = 0;
        chosenindex = -1;
        chosenbox = [];

        index = 0;

        for box in results[0].boxes.data:
            #print(box)

            index += 1;

            if(box[4] < 0.6): continue;



            width = box[2] - box[0]
            height = box[3] - box[1];

            if width*height > maxsize:
                maxsize = width*height;
                chosenbox = box;
                chosenindex = index;



            #draw.rectangle(( topleft, topright ), fill=(255,255,255,255))

        if len(chosenbox) == 0:
            print("Face not found, restarting...");
            continue;

        box = chosenbox;
        width = box[2] - box[0]
        height = box[3] - box[1];

        width_1 = width*padding;
        height_1 = height*padding;

        width_2 = width*paddingmask;
        height_2 = height*paddingmask;

        centerx = (box[2] + box[0]) / 2
        centery = (box[3] + box[1]) / 2

        topleft = [ int(centerx - width_1/2), int(centery - height_1/2) ]
        topright = [ int(centerx + width_1/2) , int(centery + height_1/2) ]

        if(topleft[0] < 0): topleft[0] = 0;
        if(topright[0] >= source_img.size[0]): topright[0] = source_img.size[0]

        if(topleft[1] < 0): topleft[1] = 0;
        if(topright[1] >= source_img.size[1]): topright[1] = source_img.size[1]


        topleft_mask = [ int(centerx - width_2/2), int(centery - height_2/2) ]
        topright_mask = [ int(centerx + width_2/2) , int(centery + height_2/2) ]


        if(topleft_mask[0] < 0): topleft_mask[0] = 0;
        if(topright_mask[0] >= source_img.size[0]): topright_mask[0] = source_img.size[0]

        if(topleft_mask[1] < 0): topleft_mask[1] = 0;
        if(topright_mask[1] >= source_img.size[1]): topright_mask[1] = source_img.size[1]



        cropped = source_img.crop((topleft[0], topleft[1], topright[0], topright[1]))

        mask = Image.new('RGBA', (cropped.size[0], cropped.size[1]))

        draw = ImageDraw.Draw(mask)

        mask_coord_left = (topleft_mask[0] - topleft[0], topleft_mask[1] - topleft[1])
        mask_coord_right = (topright_mask[0] - topleft[0], topright_mask[1] - topleft[1])

        draw.rectangle(( mask_coord_left, mask_coord_right ), fill=(255,255,255,255))


        cropped.save("./r1.png")
        mask.save("./r2.png")

        cropped.paste(mask, (0,0), mask)

        encoded = base64.b64encode(open("./r1.png", "rb").read()) #change the directory and image name to suit your needs
        encodedString=str(encoded, encoding='utf-8')

        encoded2 = base64.b64encode(open("./r2.png", "rb").read()) #change the directory and image name to suit your needs
        encodedString2=str(encoded2, encoding='utf-8')


        if not skip_diffusion:

            GoodEncoded='data:image/png;base64,' + encodedString
            MaskEncoded='data:image/png;base64,' + encodedString2

            payload = {
              "init_images": [
                GoodEncoded
              ],
              "denoising_strength": 0.75,
              "mask": MaskEncoded,
              "mask_blur": 4,
              "inpainting_fill": 0,
              "inpaint_full_res_padding": 50,
              "inpaint_full_res": True,
              "prompt": args.prompt,
              "negative_prompt": args.negative_prompt,
              "seed": -1,
              "subseed": -1,
              "batch_size": 1,
              "steps": 20,
              "cfg_scale": 7,
              "width": cropped.size[0],
              "height": cropped.size[1],
              "override_settings": {},
              "override_settings_restore_afterwards": True,
              "script_args": [],
              "sampler_index": "Euler a",
              "include_init_images": False,
              "send_images": True,
              "save_images": False,
            }

            payloadJson = json.dumps(payload)

            print("Calling stable diffusion...")

            resp = requests.post(url=requestURL + "/sdapi/v1/img2img", data=payloadJson).json()

            if "images" in resp:
                img = Image.open(io.BytesIO(base64.b64decode(resp['images'][0])))

                source_img.paste(img, (topleft[0], topleft[1]))
                source_img
            else:
                print("Image generation failed, restarting...");
                continue;

        source_img_outlines = source_img.copy();

        draw = ImageDraw.Draw(source_img_outlines)

        index = 0;

        for box in results[0].boxes.data:
            #print(box)
            
            index += 1;

            if not skip_diffusion and index == chosenindex:
                continue;

            up = ((box[0], box[1]), (box[2], box[1]))
            down = ((box[0], box[3]), (box[2], box[3]))
            left = ((box[0], box[1]), (box[0], box[3]))
            right = ((box[2], box[1]), (box[2], box[3]))

            draw.line(up, fill =(0,0,255, 255), width = 5)
            draw.line(down, fill =(0,0,255, 255), width = 5)
            draw.line(left, fill =(0,0,255, 255), width = 5)
            draw.line(right, fill =(0,0,255, 255), width = 5)

            

        open_cv_image = np.array(source_img_outlines)
        generated_image = open_cv_image[:, :, ::-1].copy()

        print(generated_image.size);




def image_displayer():

    global stop_loop;

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', 512, 512)

    while True:

        pos = pyautogui.position();



        if len(generated_image) > 0:
            cv2.imshow('dst_rt',generated_image)
            #print("show frame.");

        if(pos[0] == 0 and pos[1] == 0):
            stop_loop = True;
            break;


        #print("RUNNING");

        cv2.waitKey(1)

t1 = threading.Thread(target = image_generator)
t2 = threading.Thread(target = image_displayer)

print("Beginning program...");

t2.start();
t1.start();
