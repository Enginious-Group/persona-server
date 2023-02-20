
import io
import os
from zipfile import ZipFile


from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse, FileResponse, HTMLResponse
from PIL import Image

from api import *
from my_image_processing import MyModel 


BASE_DIR = Path('D:\Persona\Maya\Input')

class Message(BaseModel):
    message: str



def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

state = {}

app_desc = " Persona API "

app = FastAPI(title='Persona API', description=app_desc, debug=True)


@app.on_event("startup")
async def load_model():
    state['model'] = MyModel()


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/process/obj_forward")


@app.post(
    "/process/obj_forward",

    responses={
        200: {
            "content": {"zip": {}},
            "description": "Returns processed image",
        },
        422: {"model": Message},
        500: {"model": Message},
    }
)
async def obj_forward_process(file: UploadFile = File(...)):


    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return HTMLResponse({"message": "Image must be jpg or png format!"}, status_code=422)

    image = read_imagefile(await file.read())

    try:
        res = state['model'].process_image(image)
        
        person_name = file.filename.split(".")[0]
        current_working_dir = os.path.join(BASE_DIR, person_name)
        if not os.path.exists(current_working_dir):
            os.makedirs(current_working_dir)

        current_concept_path = os.path.join(current_working_dir, file.filename)
       
        res.save(current_concept_path)

        process_file(Path(current_concept_path), Path(BASE_DIR))

        person_name_png = person_name + '.png'
        person_name_obj = person_name + '.obj'
        person_name_json = person_name + '.json'
       
        with ZipFile('response.zip', 'w') as response_zip:
            if os.path.exists(os.path.join(current_working_dir, 'texture_correction_deca', person_name_png)):
                response_zip.write(os.path.join(current_working_dir, 'texture_correction_deca', person_name_png), 't_' + person_name + '.png')
            if os.path.exists(os.path.join(current_working_dir, 'deca3dface', person_name_obj)):
                response_zip.write(os.path.join(current_working_dir, 'deca3dface', person_name_obj), person_name + '.obj')
            if os.path.exists(os.path.join(current_working_dir, 'correction_color_deca', person_name_json)):
                response_zip.write(os.path.join(current_working_dir, 'correction_color_deca', person_name_json), "color_correction.json")
            if os.path.exists(os.path.join(current_working_dir, 'age_estimator', person_name_json)):
                response_zip.write(os.path.join(current_working_dir, 'age_estimator', person_name_json), "age_estimator.json")
            if os.path.exists(os.path.join(current_working_dir, 'body_shape_classifier', person_name_json)):
                response_zip.write(os.path.join(current_working_dir, 'body_shape_classifier', person_name_json), "body_shape_classifier.json")
            if os.path.exists(os.path.join(current_working_dir, 'hairstyle_classifier', person_name_json)):
                response_zip.write(os.path.join(current_working_dir, 'hairstyle_classifier', person_name_json), "hairstyle.json")
            if os.path.exists(os.path.join(current_working_dir, 'hair_color', person_name_json)):
                response_zip.write(os.path.join(current_working_dir, 'hair_color', person_name_json), "haircolor.json")
        return FileResponse("response.zip", media_type="zip")
    except Exception as e:
        message = "Something went wrong during image processing"
        if app.debug:
            message += f"\n{e}"
        return HTMLResponse(message, status_code=500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        required=False,
        help='storage path'
    )
    args = parser.parse_args()
    if args.s:
        BASE_DIR = Path(args.s)

    uvicorn.run(app, debug=True, port=9092, host="0.0.0.0")
    

