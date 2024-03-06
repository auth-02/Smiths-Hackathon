import requests
import base64
import json

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
    return encoded_string

def generate_response(model, prompt, images=None):
    url = 'http://localhost:11434/api/generate'  
    data = {
        'model': model,
        'prompt': prompt
    }
    if images:
        data['images'] = images

    response = requests.post(url, json=data)
    response_content = response.content.decode("utf-8") 
    json_objects = [json.loads(obj) for obj in response_content.strip().split("\n")]
    print(json_objects)
    responses = [item.get('response', '') for item in json_objects]
    combined_response = ' '.join(responses)
    sentences = combined_response.split(". ")
    formatted_response = "".join([" " + sentence + "." for sentence in sentences])
    print(formatted_response)
    print('\n')

model_name = "llava:7b-v1.5-q2_K"
prompt_text = "Describe the image?"
image_paths = ["D:/CODESPACES/Hackathons/Smiths Hackathon/04 CodeBase/results/Object Detection.jpg"]  

base64_image_data = [encode_image_to_base64(image_path) for image_path in image_paths]

generate_response(model_name, prompt_text, images=base64_image_data)