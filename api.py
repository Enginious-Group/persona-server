"""
Example usage:
    python process_with_ml_apis.py --input_folder data/test_data --output_folder test_output --save_logs_as test_logs.log

The following things are hardcoded in the script:
    - ephemeral url of the VM serving the APIs (it may change when the VM is stopped and restarted)
    - ports on which the apis can be reached
    - structure of ML APIs to be used
    - list of available ML APIs. Currently it is:
        - body_shape_classifier
        - detailed_3d_face
        - face_cutter
        - gfpgan
        - ethnicity_predictor
        - encoder4editing
        - age_estimator
        - texture_correction
"""

import argparse
import logging
import requests
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from zipfile import ZipFile
import shutil


# EPHEM_URL = "104.198.4.217"
# EPHEM_URL = "34.83.110.169"
EPHEM_URL = "34.168.92.129"

PORTS = {
    "body_shape_classifier":   9094,
    "detailed_3d_face":        9095,
    "deca3dface":              9098,
    "face_cutter":             9090,
    "gfpgan":                  9091,
    "ethnicity_predictor":     9092,
    "encoder4editing":         9093,
    "age_estimator":           9096,
    "texture_correction":      9097,
    "texture_correction_deca": 9097,
    "correction_color":        9097,
    "correction_color_deca":   9097,
    "hairstyle_classifier":    9084,
    "hair_color":              9083,
}

def flatten(t:list):
    return [item for sublist in t for item in sublist]


class ApiError(Exception):
    pass


def call_ml_api(input_filepath:Path, output_dirpath:Path, api_name:str, output_extension:str):
    '''
    Function making a request call to a given ML API
        input_filepath: Path to a file to be sent to the API
        output_dirpath: Path to the folder in which results should be stored (API-specific subfolder will be created)
        api_name: should be recognised as key in PORTS
        output_extension: extension of the resulting file, needs to be adjusted to the api_name (e.g., for now we are using 'json', 'png' and 'obj')
    ''' 
    # Create url to access api
    url = f'http://{EPHEM_URL}:{PORTS[api_name]}/process/{api_name}'

    # Open input file, and process with api 
    with open(input_filepath, 'rb') as file:
        r = requests.post(url, files={'file': file})

    if r.status_code == 200:
        logging.info(f"{api_name} returned results!")

        # Create folder for output
        output_dir = output_dirpath / input_filepath.stem / api_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save output to file
        output_filepath = output_dir / (input_filepath.stem + "." + output_extension)
        with open(output_filepath, 'wb') as output:
            output.write(r.content)

        # Return
        return output_filepath

    else:
        raise ApiError(f"{api_name} failed! Status code: {r.status_code}")


def unzip_d3df_file(zip_filepath:Path):
    d3df_outputs = {
        "obj": "api_output/image/image_head.obj",
        "texture": "api_output/image/image.jpg",
    }
    with ZipFile(zip_filepath, 'r') as f:
        f.extractall(zip_filepath.parent)
    
    results = {}
    # Move the unzipped files
    for file_type in d3df_outputs.keys():
        src = zip_filepath.parent / d3df_outputs[file_type]
        dst = zip_filepath.parent / f"{zip_filepath.stem}{Path(d3df_outputs[file_type]).suffix}"
        shutil.move(src=src, dst=dst)
        results[file_type] = dst

    # Remove the unused folder
    shutil.rmtree(zip_filepath.parent / "api_output")

    logging.info(f"detailed_3d_face result was unzipped!")

    return results

def unzip_deca_file(zip_filepath:Path):
    with ZipFile(zip_filepath, 'r') as f:
        f.extractall(zip_filepath.parent)
    shutil.move(zip_filepath.parent / "head.png", zip_filepath.parent / f"{zip_filepath.stem}.png")
    shutil.move(zip_filepath.parent / "head.obj", zip_filepath.parent / f"{zip_filepath.stem}.obj")
    zip_filepath.unlink()
    
    
    deca_outputs = {
        "obj": zip_filepath.parent / f"{zip_filepath.stem}.obj",
        "texture": zip_filepath.parent / f"{zip_filepath.stem}.png",
    }

    logging.info(f"deca3dface result was unzipped!")
    return deca_outputs


def process_file(input_filepath:Path, output_dirpath:Path):
    """
    Pass an image through a branched sequence of ML APIs
    """

    logging.info(f"Processing file: {input_filepath} ...")
    
    try:
        #_ = call_ml_api(input_filepath, output_dirpath, "body_shape_classifier", "json")
        face = call_ml_api(input_filepath, output_dirpath, "face_cutter", "png")
        face_gfp = call_ml_api(face, output_dirpath, "gfpgan", "png")
        face_gfp_e4e = call_ml_api(face_gfp, output_dirpath, "encoder4editing", "png")
        _ = call_ml_api(face_gfp_e4e, output_dirpath, "age_estimator", "json")
        _ = call_ml_api(face_gfp_e4e, output_dirpath, "ethnicity_predictor", "json")
        _ = call_ml_api(face_gfp_e4e, output_dirpath, "hairstyle_classifier", "json")
        _ = call_ml_api(face_gfp_e4e, output_dirpath, "hair_color", "json")
        

        deca_zip = call_ml_api(face_gfp_e4e, output_dirpath, "deca3dface", "zip")
        deca_results = unzip_deca_file(deca_zip)
        _ = call_ml_api(deca_results["texture"], output_dirpath, "texture_correction_deca", "png")
        _ = call_ml_api(deca_results["texture"], output_dirpath, "correction_color_deca", "json")

        d3df_zip = call_ml_api(face_gfp_e4e, output_dirpath, "detailed_3d_face", "zip")
        d3df_results = unzip_d3df_file(d3df_zip)
        _ = call_ml_api(d3df_results["texture"], output_dirpath, "texture_correction", "png")
        _ = call_ml_api(d3df_results["texture"], output_dirpath, "correction_color", "json")

        logging.info("Done!")
        # mlobjpath = Path('D:\Persona\Maya\Input\ML_Head\deca3dface\ML_Head.obj')
        # mljpgpath = Path('D:\Persona\Maya\Input\ML_Head\deca3dface\ML_Head.png')
        # mlobjnew = Path('D:\Persona\Maya\Input\ML_Head.obj')
        # print('copycopy')
        # mljpgnew = Path('D:\Persona\Maya\Input\genericHead_texture_v001.png')
        # shutil.copyfile(mlobjpath, mlobjnew)
        # shutil.copyfile(mljpgpath, mljpgnew)

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed because {e}")
    except ApiError as e:
        logging.error(f"Failed because {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Process files in input_folder with ML APIs, get results in output_folder, find processing summary in record.txt') # TODO store record.txt
    parser.add_argument(
        '-i',
        '--input_folder',
        required=True,
        help='relative path to folder with input images'
    )
    parser.add_argument(
        '-o',
        '--output_folder',
        required=True,
        help='relative path to folder where outputs should be stored'
    )
    parser.add_argument(
        '--save_logs_as',
        default='processing_logs.log',
        help='save logs to file, provide filename'
    )

    args = parser.parse_args()
    
    # See if the logs should be stored
    if args.save_logs_as:
        logging.basicConfig(
            filename=args.save_logs_as,
            encoding='utf-8',
            level=logging.INFO, 
            format='%(asctime)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

    # Get data from input folder
    input_folderpath = Path(args.input_folder)
    input_filepaths = flatten([list(input_folderpath.rglob(f"*.{ext}")) for ext in ['png', 'jpg', 'jpeg', "PNG", "JPG", "JPEG", "tiff", "TIFF"]])
    logging.info(f"Found {len(input_filepaths)} images in the chosen folder.")

    # Setup output folder
    output_dirpath = Path(args.output_folder)
    
    # Process the files
    for input_filepath in tqdm(input_filepaths):
        process_file(input_filepath, output_dirpath)