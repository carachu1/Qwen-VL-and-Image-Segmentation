# Image Segmentation Experiment Combining Segmentation Models and QWEN-VL-Chat Model

This repository contains the source code, documentation, and materials associated with the research project "Image Segmentation Experiment Combining Segmentation Models and QWEN-VL-Chat Model" presented by Guo Chumeng from The Hong Kong University of Science and Technology (Guangzhou), Guangzhou, Guangdong province, China.

The whole experiment of this project mainly contains three parts.
-  **Reproducing Qwen-VL**
-  **Image Segmentation Using SAM**
-  **Image Segmentation Using GroundingDino+SAM**

## Reproducing Qwen-VL
Replication of the functionalities of the Qwen-VL and Qwen-VL-Chat models for tasks involving image descriptions and image-related conversational generation.
Reference link: [Qwen-LM/Qwen-VL](https://github.com/QwenLM/Qwen-VL).

Before reproducing the code, it is needed to configure the relevant python environment and dependencies according to this link.
The specific steps of my reproducing Qwen-VL and Qwen-VL-Chat are both in the `Qwen Reproducing` file.

## Image Segmentation Using [SAM](https://github.com/facebookresearch/segment-anything)
An innovative approach that combines the Qwen-VL-Chat model with the Segment-Anything project for object detection and segmentation tasks in images.

### Requirments
> Three pretrained weights of SAM: [default or vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), [vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) and [vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
Downloading directly from these links is possible.

> The code requires python>=3.8, as well as pytorch>=1.7 and torchvision>=0.8.

> The prerequisite is that the reproduction of Qwen-VL is successful, as this part is combined with Qwen-VL.

In `test_seg` file, I mainly added a function called "perform_segmentation()"

**Extract Bounding Box Coordinates:** The extract_bbox_coordinates function extracts bounding box coordinates from a text string using a regex pattern. The coordinates are expected to be in the format "(x1,y1),(x2,y2)".

**Extract and Display Image:** The extract_and_display_image function extracts an image URL from the provided text response, downloads and displays the image. If the URL is not valid, it attempts to open the image from a local file path.

**Show Mask and Box:** The show_mask and show_box functions display the segmentation mask and the bounding box on the image.

**Segmentation Process:** 
- Bounding box coordinates are converted to absolute values based on the image dimensions.
- The SAM model is loaded and configured.
- The image is set in the predictor.
- Segmentation is performed for each bounding box, displaying the masks and boxes on the image.

```
query = tokenizer.from_list_format([
        {'image': '/hpc2hdd/home/cguo847/train2017/000000174712.jpg'},  # Either a local path or a URL
        {'text': '这是什么?'},
    ])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# 这张图片展示了一架商用飞机在夜空中飞行，距离桥面很近，可能正在降落。飞机的前部有四个灯亮着，其中两个在机头上，两个在机翼下方。图片的地面显示有一条沿伸至远方的白线，可能是跑道或标志线。在图片的右侧，可以看到一个月亮和飞机一起照亮了暗蓝色的天空。
# Second round of dialogue
response, history = model.chat(tokenizer, '框出图片中飞机的位置', history=history)
print(response)
# <ref>飞机</ref><box>(343,119),(714,261)</box>
segmentation_result = perform_segmentation(response)
if segmentation_result is not None:
    print("Segmentation completed successfully.")
else:
    print("Segmentation failed.")
```
### SAM Result:
![image](https://github.com/carachu1/Qwen-VL-and-Image-Segmentation/assets/150044043/3979e6d6-143a-41e2-be76-a96a278b6d6f)

## Image Segmentation Using [GroundingDino+SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)
A method that allows segmentation of objects directly during the Q&A process with the multimodal model.

### Requirments
> groundingdino_swint_ogc.pth

> sam_vit_h_4b8939.pth

> note: Problems matching torch version and cuda version

### Preparation of Fine-tuning Datasets

> Processing and Alignment of the Fine-Tuning Dataset: Utilizes the [lava-plus-v1-117k-tool-merge.json](https://huggingface.co/datasets/LLaVA-VL/llava-plus-data/tree/main) dataset, extracting and modifying data to meet new structure requirements. Refer to `converted_data.ipynb` code file for specific data conversion.

> Image Dataset Preparation: Downloads the [coco-dataset-2017-Train-images](https://cocodataset.org/#download) set for local usage during fine-tuning.

### Fine-tuning Qwen-VL-Chat
Fine-tune Qwen-VL-Chat using the Lora single-card fine-tuning script provided by the Qwen-VL team, and save the fine-tuned model. 

### Automated Detection and Segmentation

**Extract Image Path:**
The extract_image_path function extracts the image URL from the query output using a regex pattern. The image URL is expected to be enclosed within <img> tags.

**Detect and Segment:**
- The detect_and_segment function detects the <gsam> tag in the response and extracts the text prompt for segmentation.

- The function then constructs and executes a command to run the grounded_sam_demo.py script with the specified parameters.

- The segmentation results are displayed using matplotlib.

```
query = tokenizer.from_list_format([
    {'image': '/hpc2hdd/home/cguo847/train2017/000000174712.jpg'}, # Either a local path or an url
    {'text': 'Can you segement the airplane in this image?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# <gsam>airplane</gsam>

detect_and_segment(response, query)
#Detected text prompt: airplane
```
<img width="419" alt="image" src="https://github.com/carachu1/Qwen-VL-and-Image-Segmentation/assets/150044043/1af4daae-6bc1-454b-812b-c05813c1ead1">

<img width="244" alt="image" src="https://github.com/carachu1/Qwen-VL-and-Image-Segmentation/assets/150044043/b6d0f570-e041-4883-bd85-710aa97740ec">

<img width="242" alt="image" src="https://github.com/carachu1/Qwen-VL-and-Image-Segmentation/assets/150044043/ec02eaab-1a91-4ee9-a175-75aac9e438c9">

## Key Contributions

1. **Integration of Qwen-VL and Segment-Anything Project**: The innovative approach of integrating the Qwen-VL-Chat model with the Segment-Anything project enables deep understanding and segmentation of objects within images.

2. **Development of an Integrated System**: A novel system that combines QwenVL’s conversational capabilities with Segment-Anything’s image segmentation functionality, achieving comprehensive image understanding and editing in real-time.

3. **Dynamic Response Mechanism**: Incorporation of a dynamic response mechanism within Qwen-VL to detect specific tags in the generated responses and dynamically invoke the segmentation model.

4. **Enhanced User Experience**: Advanced AI models integration to provide intelligent image understanding and editing capabilities, enhancing user interaction with the system.


## Future Work

- Comprehensive evaluation of the system's performance, usability, and robustness.
- Integration of other large language models to improve question answering and instruction understanding.
- Development of a more dynamic and interactive dialogue system for real-time feedback.
- In-depth research on multimodal fusion technology for effective integration of visual and language information.

## References

The project references several key papers and resources, detailed in the provided documentation. For more information, refer to the `References` section in the documentation.

## Contact

For inquiries, feedback, or collaboration, please contact Guo Chumeng at [cguo847@connect.hkust-gz.edu.cn](mailto:cguo847@connect.hkust-gz.edu.cn).
