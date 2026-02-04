from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.utils import logging as hf_logging
import json
import cv2
import os
from typing import Dict

class QwenVL3:
    def __init__(self, model_path = "Qwen/Qwen3-VL-32B-Instruct"):
        print("Loading qwen3 model...")
        hf_logging.disable_progress_bar()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, dtype="auto", device_map="auto", local_files_only=True)
        min_pixels = 256 * 28 * 28          
        max_pixels = 1280 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(model_path, 
                                                    min_pixels=min_pixels,
                                                    max_pixels=max_pixels,
                                                    local_files_only=True)
        # self.prompt_format = "Detect {target}. Report bbox coordinates in JSON format."
        self.prompt_format = """Detect {target}. You are a helpful assistant to detect objects in images. 
                When asked to detect elements based on a description, you return a valid JSON object containing \
                bounding boxes for all elements in the form `[{{"bbox_2d": [xmin, ymin, xmax, ymax], \
                "label": "placeholder"}}, ...]`. For example, a valid response could be: \
                `[{{"bbox_2d": [10, 30, 20, 60], "label": "placehoslder"}}, {{"bbox_2d": [40, 15, 52, 27], \
                "label": "placeholder"}}]`."""
    
    
    def parse_bbox(self, results, image_path):
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        new_results = []
        for item in results:
            bbox = item.get("bbox_2d")
            label = item.get("label")

            if bbox and label and len(bbox) == 4:
                # 边界框坐标：[xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = [int(coord)/1000 for coord in bbox]
                xmin = int(xmin*width)
                ymin = int(ymin*height)
                xmax = int(xmax*width)
                ymax = int(ymax*height)
                new_results.append({"bbox_2d":[xmin, ymin, xmax, ymax], "label": label})
        return new_results 
        

    def predict(self, target_prompt, image_path):
        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": self.prompt_format.format(target=target_prompt)},
                    ],
                }
            ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        try:
            json_str = output_text[0].split('```json')[1].split('```')[0]
            results = json.loads(json_str)
            if isinstance(results, Dict):
                results = [results]
        except Exception as e:
            print(f"JSON decode error for image {image_path}")
            results = []

        parse_results = self.parse_bbox(results, image_path)
        return parse_results

    def draw_bbox_on_image(self, image_path, bboxs, save_image_path=None):
        image = cv2.imread(image_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for bbox in bboxs:
            coord = bbox.get("bbox_2d")
            label = bbox.get("label")
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 3, 3)[0]
            cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), color=(0, 255, 0), thickness=3) 
            y_label = max(t_size[1], coord[1]-t_size[1])
            cv2.putText(image, label, (coord[0], y_label), font, 3, (0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        if save_image_path is None:
            image_path_list = os.path.splitext(image_path)
            save_image_path = image_path_list[0]+"_res"+image_path_list[1]
        cv2.imwrite(save_image_path, image)


        

if __name__=="__main__":
    model_path = "Qwen/Qwen3-VL-32B-Instruct"
    target_prompt = "robotic arm"
    test_image_path = "/mnt/data/ganrunhe/input_video/Robo_real_frames_png/000001.png"
    save_image_path = "./test_case/robo_arms_res.jpg"

    model = QwenVL3(model_path)
    bboxs = model.predict(target_prompt, test_image_path)
    print(bboxs)
    # 调试时
    model.draw_bbox_on_image(test_image_path, bboxs, save_image_path)
    
        
