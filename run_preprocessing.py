import os
from ConvertPlainTxt import ConvertPlainTxt
from ConvertToJson import ConvertToJson

def run_preprocessing():

    """
    Traverse the "legal resources" directory, convert each .docx file to plain text,
    then parse that text into JSON and save the result into a "json_files" subdirectory.
    """

    # Construct the path to the "legal resources" folder
    root_dir = os.path.join(os.getcwd(), "legal resources")

    # Create (if it doesn't exist) a subfolder called "json_files" under root_dir
    json_output_dir = os.path.join(root_dir, "json_files2")
    os.makedirs(json_output_dir, exist_ok=True)

    txt_converter = ConvertPlainTxt()

    # Walk through every folder, subfolder, and file under root_dir
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:

            if file.startswith("~$"):   # Skip temporary Office files that start with "~$"
                continue

            if file.lower().endswith(".docx"):
                docx_file_path = os.path.join(subdir, file)
                json_file_path = os.path.join(json_output_dir, os.path.splitext(file)[0] + ".json")

                json_parser = ConvertToJson(docx_file_path)
                plain_txt = txt_converter.docx_to_text(docx_file_path)
                json_parser.parse_and_save(plain_txt, json_file_path)
                print(f"Saved JSON to {json_file_path}\n")


if __name__ == "__main__":
    run_preprocessing()
