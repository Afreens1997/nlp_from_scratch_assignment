from collections import defaultdict
from pprint import pprint
import sys
import uuid

import spacy
from spacy.matcher import Matcher


sys.path.append('/Users/afreenshaikh/Documents/CMU_projects/scipdf_parser')
sys.path.append('/Users/afreenshaikh/Documents/CMU_projects/nlp_from_scratch_assignment')

from utils.preannotation import remove_overlapping_spans
from utils.file_utils import read_json, write_json


all_data = read_json("/Users/afreenshaikh/Library/CloudStorage/GoogleDrive-afreens@andrew.cmu.edu/.shortcut-targets-by-id/1tZMZ1hVZ12FuHdPiu88Shf3zG1ZgrK47/A2/dump/afreen/anlp_afreen_dataset_v1.json")
pprint(all_data[0])

file_data_map = defaultdict(list)

for dp in all_data:
    file_name = dp["file_upload"].split("-")[-1][:-5]
    file_data_map[file_name].append(dp)


label_map = {
   "metric_names_pattern":"MetricName",
    "task_names_pattern":"TaskName",
    "method_names_pattern":"MethodName",
    "dataset_names_pattern":"DatasetName"
}

metric_names = set()
task_names = set()

per_file_annotations = {}
for file_name,dps in file_data_map.items():
    method_names = set()
    dataset_names = set()
    for dp in dps:
        dp_annotations = sum([x['result'] for x in dp["annotations"]], [])
        dp["annotations"][0]["result"] = dp_annotations
        dp["annotations"] = [dp["annotations"][0]]
        for annot in dp_annotations:
            if "MethodName" in annot["value"]["labels"]:
                method_names.add(annot["value"]["text"])
            if "DatasetName" in annot["value"]["labels"]:
                dataset_names.add(annot["value"]["text"])
            if "MetricName" in annot["value"]["labels"]:
                metric_names.add(annot["value"]["text"])
            if "TaskName" in annot["value"]["labels"]:
                task_names.add(annot["value"]["text"])
    per_file_annotations[file_name] = {}
    per_file_annotations[file_name]["method_names"] = method_names
    per_file_annotations[file_name]["dataset_names"] = dataset_names

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

for file_name, per_file_annot in per_file_annotations.items():
    metric_names_pattern = [[{"TEXT": y} for y in x.split()] for x in metric_names]
    task_names_pattern  = [[{"TEXT": y} for y in x.split()] for x in task_names]
    method_names_pattern  = [[{"TEXT": y} for y in x.split()] for x in per_file_annot["method_names"]]
    dataset_names_pattern  = [[{"TEXT": y} for y in x.split()] for x in per_file_annot["dataset_names"]]

    matcher.add("metric_names_pattern", metric_names_pattern)
    matcher.add("task_names_pattern", task_names_pattern)
    matcher.add("method_names_pattern", method_names_pattern)
    matcher.add("dataset_names_pattern", dataset_names_pattern)

    for dp in file_data_map[file_name]:
        doc = nlp(dp["data"]["text"])
        matches = matcher(doc) 
        len1 = len(dp["annotations"][0]["result"])
        # print(len(dp["annotations"][0]["result"]))
        for match_id, start, end in matches:
            span = doc[start:end]  
            string_id = nlp.vocab.strings[match_id]
            span_label = label_map[string_id]
            # print(span, len(span), span_label)
            dp["annotations"][0]["result"].append(
                {
                    "id":str(uuid.uuid4()),
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                    "start": span.start_char,
                    "end": span.end_char,
                    "text": span.text,
                    "labels": [span_label]
                    }
                }
            )
        
        dp["annotations"][0]["result"] = remove_overlapping_spans(dp["annotations"][0]["result"])
        len2 = len(dp["annotations"][0]["result"])
        if len1!=len2:
            print(len1, len2, [x["value"]["text"] for x in dp["annotations"][0]["result"]])

all_data_new = []
for k,v in file_data_map.items():
    all_data_new.extend(v)

# write_json(all_data_new, "/Users/afreenshaikh/Library/CloudStorage/GoogleDrive-afreens@andrew.cmu.edu/.shortcut-targets-by-id/1tZMZ1hVZ12FuHdPiu88Shf3zG1ZgrK47/A2/dump/afreen/anlp_afreen_dataset_completed.json")
print(list(per_file_annotations.keys()))
    

