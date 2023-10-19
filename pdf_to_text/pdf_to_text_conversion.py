import glob
from pprint import pprint
import sys
from unicodedata import normalize

sys.path.append('/Users/afreenshaikh/Documents/CMU_projects/scipdf_parser')
sys.path.append('/Users/afreenshaikh/Documents/CMU_projects/nlp_from_scratch_assignment')

import spacy

from utils.preannotation import generate_pre_annotations


from utils.file_utils import write_json


import scipdf
# attention_dict = scipdf.parse_pdf_to_dict('/Users/afreenshaikh/Documents/CMU_projects/scipdf_parser/example_data/attention_paper.pdf', fulltext=True, as_list=True)
# pprint(attention_dict)

PARA_LENGTH_LIMIT = 10
nlp = spacy.load("en_core_web_sm")

def convert_pdf_to_dict_scipdf(pdf_path):
    pdf_dict = scipdf.parse_pdf_to_dict(pdf_path, fulltext=True, as_list=True)
    pdf_dict["pdf_path"] = pdf_path
    return pdf_dict

def write_pdf_to_dict_scipdf(pdf_path, output_path):
    pdf_dict = convert_pdf_to_dict_scipdf(pdf_path)
    write_json(pdf_dict, output_path)
    return pdf_dict

def convert_dict_to_annotation_input(pdf_dict):

    
    def get_tokenized_text(text):
        # text = normalize('NFKD', text).encode('ascii','ignore')
        text = text.encode("utf-8")
        text = text.decode("utf-8")
        # print(text)
        doc = nlp(text)
        tokens_text = [x.text for x in doc if x.is_alpha or x.is_digit or x.is_punct or x.is_currency or x.text.replace(".","").isdecimal()]
        tokens_text = " ".join(tokens_text)
        return tokens_text

    pdf_path = pdf_dict["pdf_path"]
    def get_metadata(section):
        return {
            "pdf_path": pdf_path,
            "section": section
        }
    annotation_data = []

    # include title
    annotation_data.append({
        "text" : get_tokenized_text(pdf_dict["title"]),
        "metadata": get_metadata("title")
    })

    # include abstract - all as one para
    annotation_data.append({
        "text" : get_tokenized_text(pdf_dict["abstract"]),
        "metadata": get_metadata("abstract")
    })

    #include sections
    for section in pdf_dict["sections"]:
        section_heading = section["heading"]
        # if section_heading:
        #     annotation_data.append({
        #     "text" : get_tokenized_text(section_heading),
        #     "metadata": get_metadata(f"section_heading - {section_heading}")
        #     })
        for paragraph in section["text"]:
            if len(paragraph)>PARA_LENGTH_LIMIT:
                annotation_data.append({
                "text" : get_tokenized_text(paragraph),
                "metadata": get_metadata(f"para - {section_heading}")
                })
        
    return add_annotations_to_data(annotation_data)

def add_annotations_to_data(annotated_data):
    final_data = []
    for dp in annotated_data:
        text = dp['text']
        doc = nlp(text)
        predictions = generate_pre_annotations(doc)
        final_data.append({
            "data":{"text":text,
                    "metadata":dp["metadata"]},
            "predictions":[
                {
                    "result":predictions
                }
            ]
        })

    return final_data



if __name__ == "__main__":
    pdfs_base_path = "/Users/afreenshaikh/Library/CloudStorage/GoogleDrive-afreens@andrew.cmu.edu/.shortcut-targets-by-id/1tZMZ1hVZ12FuHdPiu88Shf3zG1ZgrK47/A2/acl-2023-short"
    dict_output_path = "/Users/afreenshaikh/Library/CloudStorage/GoogleDrive-afreens@andrew.cmu.edu/.shortcut-targets-by-id/1tZMZ1hVZ12FuHdPiu88Shf3zG1ZgrK47/A2/pdf_dict"
    annotation_input = "/Users/afreenshaikh/Library/CloudStorage/GoogleDrive-afreens@andrew.cmu.edu/.shortcut-targets-by-id/1tZMZ1hVZ12FuHdPiu88Shf3zG1ZgrK47/A2/annotation_input1"

    all_pdf_files = glob.glob(pdfs_base_path+"/*")
    sorted_pdf_files = sorted(all_pdf_files, key=lambda x:int(x.split("/")[-1][:-4][3:]))
    # print(sorted_pdf_files)
    for pdf_file_path in sorted_pdf_files[2:30]:
        file_name = pdf_file_path.split("/")[-1][:-4]
        print(file_name, f"/{file_name}.json")
        pdf_dict = write_pdf_to_dict_scipdf(pdf_file_path, dict_output_path + f"/{file_name}.json")
        annotation_input_data = convert_dict_to_annotation_input(pdf_dict)
        write_json(annotation_input_data, annotation_input + f"/{file_name}.json")

        


