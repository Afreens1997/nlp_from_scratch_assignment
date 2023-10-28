import glob
import os
from pprint import pprint
import sys
from unicodedata import normalize
import argparse

sys.path.append('/path/to/scipdf_parser/code/checkout')
sys.path.append('/path/to/project/')

import spacy

from utils.preannotation import generate_pre_annotations


from utils.file_utils import read_json, write_json


import scipdf

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
        tokens_text = [x.text for x in doc if x.text.isprintable()]
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

    parser = argparse.ArgumentParser()
    parser.add_argument('pdf_base_path', type=str,
                        help='Path of folder containing pdfs')
    parser.add_argument('dict_output_path', type=str,
                        help='path to save scipdf parsed pdf files')
    parser.add_argument('annotation_input_path', type=str,
                        help='path to store annotation input jsons')
    
    args = parser.parse_args()


    
    pdfs_base_path = args.pdf_base_path
    dict_output_path = args.dict_output_path
    annotation_input = args.annotation_input_path

    
    all_pdf_files = glob.glob(pdfs_base_path+"/*")
    for pdf_file_path in all_pdf_files:
        
        file_name = pdf_file_path.split("/")[-1][:-4]
        print(file_name, f"/{file_name}.json")
        pdf_dict = write_pdf_to_dict_scipdf(pdf_file_path, dict_output_path + f"/{file_name}.json")
        pdf_dict = read_json(dict_output_path + f"/{file_name}.json")
        annotation_input_data = convert_dict_to_annotation_input(pdf_dict)
        write_json(annotation_input_data, annotation_input + f"/{file_name}.json")
