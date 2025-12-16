# entity_results=[]
# #实体名为短语（或存在空格如eid=53627）的话，只允许其最多拥有3个空格
# entity_phrase_max_whitespace=3


# print("loading datas")            
# with open('./mixed_entity_with_code_identifier.json', 'r', encoding='utf-8') as f:
#     entity_result=json.load(f)

#     for name,entity in tqdm(entity_result.items()):
        
#         #由于基于名字分类后同名实体有很多，只要拿第一个作为样本实体插入es就行 后续可以根据es找到的样本实体的名字，再到usual_entity_cls_by_name.json中找到其他的普通实体
#         #到code_identifier_cls_by_name.json找到其他同名代码标识符实体
#         sample_entity=entity[0]
#         #如果实体名内不含字母（如eid=13697),则跳过 
#         if re.sub(r'[^a-zA-Z]', '', sample_entity["name_en"])=='':
#             continue
        
#         if len(sample_entity["name_en"])>=50:
#             continue
        
#         entity_results.append(name)
        

# print(len(entity_results))   
# abbr_match_res=pair(entity_results)

# with open('./output/abbr_result/1211/match_res.json','w') as f:
#     json.dump(abbr_match_res,f,indent=4,ensure_ascii=False)

from test_abbr import pair
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import itertools
import re
import json
import os


from test_abbr import abbreviate 

CHECKPOINT_FILE = "./output/abbr_result/checkpoint.json"
CHUNK_OUTPUT_DIR = "./output/abbr_result/1211/chunk_outputs"
FINAL_OUTPUT_FILE = "./output/abbr_result/1211/final_result.json"
PROCESSED_CHUNK_ID=[]
CHUNK_LOG_DIR="./output/abbr_result/1211/log"
print(cpu_count())

# def normalize_phrase(p: str):
#     normalized = re.split(r'[\[\(<]', p)[0].strip()
#     normalized = re.sub(r'-', ' ', normalized).lower()
#     normalized = normalized.strip(".,/\\!@#$%^&*()-_=+`~\"';:<>?")
#     normalized = re.sub(r'\s+', ' ', normalized)
#     return normalized

def build_corpus_maps(corpus):
    normalized_to_original = {}
    for item in corpus:
        if len(item)>40 and '_' in item :
            continue
        normalized = re.sub(r'[/]', ' ', item)
        normalized = re.split(r'[\[\(<]', normalized)[0].strip()
        normalized = re.sub(r'-', ' ', normalized).lower()
        normalized = normalized.strip(".,/\\!@#$%^&*()-_=+`~\"';:<>?")
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized_to_original.setdefault(normalized, []).append(item)
        #特殊情况："High-Level Data Link Control"->HighLevel Data Link Control
        if '-' in item:
            normalized=item.replace('-','')
            normalized = re.split(r'[\[\(<]', normalized)[0].strip()
            normalized = normalized.strip(".,/\\!@#$%^&*()-_=+`~\"';:<>?")
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized_to_original.setdefault(normalized, []).append(item)

    normalized_corpus = frozenset(normalized_to_original.keys())
    compact_to_original = {}

    for nrm in normalized_corpus:
        compact = nrm.replace(' ', '')
        compact_to_original.setdefault(compact, []).extend(
            normalized_to_original[nrm]
        )

    compact_corpus = frozenset(compact_to_original.keys())
    return normalized_corpus, normalized_to_original, compact_to_original, compact_corpus


def pair_worker(chunk_id,chunk, compact_corpus_set, normalized_to_original, compact_to_original):
    local_results = {}
    log=open(f"{CHUNK_LOG_DIR}/chunk_{chunk_id}.log", "a")
    log.write(f"[START] chunk {chunk_id} | Time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.flush()
    for i,phrase in enumerate(chunk):
        log.write(f"[PROCESS] phrase {phrase} {i+1}/{len(chunk)} | Time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.flush()
        if len(phrase) > 50:
            continue
        abbrs = abbreviate(phrase, max_part_len=8, max_abbr_len=8)
        matches = abbrs & compact_corpus_set

        if matches:
            full_items = normalized_to_original[phrase]
            for abbr in matches:
                for full, ab in itertools.product(full_items, compact_to_original[abbr]):
                    if full != ab:
                        local_results.setdefault(full, []).append(ab)

    out_path = f"{CHUNK_OUTPUT_DIR}/chunk_{chunk_id}.json"
    with open(out_path, "w") as f:
        json.dump(local_results, f, ensure_ascii=False)
    
    log.write(f"[FINISH] chunk {chunk_id} | Time:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.flush()
    return chunk_id

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return {"processed_chunks": []}
    with open(CHECKPOINT_FILE) as f:
        return json.load(f)


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def parallel_pair(entity_results, n_process=None, chunk_size=5000):
    if n_process is None:
        n_process = cpu_count()

    #映射
    phrases = entity_results
    normalized_corpus, normalized_to_original, compact_to_original, compact_corpus = build_corpus_maps(phrases)

    chunks = [
        normalized_corpus[i:i+chunk_size]
        for i in range(0, len(normalized_corpus), chunk_size)
    ]

    print(f"Launching {n_process} processes, {len(chunks)} chunks...")
    cp = load_checkpoint()
    processed_chunks = set(cp.get("processed_chunks", []))

    todo = [(i, chunks[i]) for i in range(len(chunks)) if i not in PROCESSED_CHUNK_ID]
    print(f"Need to process {len(todo)} chunks")

    worker = partial(
        pair_worker,
        compact_corpus_set=compact_corpus,
        normalized_to_original=normalized_to_original,
        compact_to_original=compact_to_original
    )


    with Pool(processes=n_process) as pool:
        for chunk_id in tqdm(pool.starmap(worker, todo), total=len(todo)):
            processed_chunks.add(chunk_id)
            save_checkpoint({"processed_chunks": list(processed_chunks)})


    print("All chunks processed.")
    return processed_chunks


def merge_chunk_files():
    final_result = {}

    for fname in os.listdir(CHUNK_OUTPUT_DIR):
        if not fname.endswith(".json"):
            continue
        with open(f"{CHUNK_OUTPUT_DIR}/{fname}",'r') as f:
            data = json.load(f)
        for k, v in data.items():
            final_result.setdefault(k, []).extend(v)

    with open(FINAL_OUTPUT_FILE, "w") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    print(f"Final merged result saved to {FINAL_OUTPUT_FILE}")


def sort_result(detect_abbr_result_path):
    f_res=open('./output/abbr_result/1211/sort_result.json','a') 
    with open('./mixed_entity_with_code_identifier.json', 'r', encoding='utf-8') as f:
        name_to_entities_dict=json.load(f)

    with open(detect_abbr_result_path,'r')as f:
        abbr_result=json.load(f)
        #一个long_name可能有几个缩写形式，把每一个缩写形式拿出来和long_name拼成abbr+long_name
        for long_name,abbr_list in abbr_result.items():
            long_name_entity_list=[{"eid":entity["eid"],"source_table":entity["source_table"]} for entity in name_to_entities_dict[long_name]]

            for abbr in abbr_list:
                short_name_entity_list=[{"eid":entity["eid"],"source_table":entity["source_table"]} for entity in name_to_entities_dict[abbr]]
                # result_dict[f"{abbr}+{long_name}"]=[short_name_entity_list,long_name_entity_list]
                f_res.write(json.dumps({f"{abbr}+{long_name}":[short_name_entity_list,long_name_entity_list]})+'\n')
                f_res.flush()
        

# 827893

if __name__ == "__main__":
    # sort_result("/home/fdse/hzc/KernelConcept/output/abbr_result/1211/final_result.json")
    # entity_results=[]
    # merge_chunk_files()
    for root,dir,files in os.walk('/home/fdse/hzc/KernelConcept/output/abbr_result/1211/chunk_outputs'):
        for file in files:
            PROCESSED_CHUNK_ID.append(int(file.split('_')[1].split('.')[0]))
    entity_results=[]
    code_identifier_entity_results=set()
    usual_entity_results=set()
    with open('./mixed_entity_with_code_identifier.json', 'r', encoding='utf-8') as f:
        entity_result=json.load(f)

        for name,entity in tqdm(entity_result.items()):
            
            #由于基于名字分类后同名实体有很多，只要拿第一个作为样本实体插入es就行 后续可以根据es找到的样本实体的名字，再到usual_entity_cls_by_name.json中找到其他的普通实体
            #到code_identifier_cls_by_name.json找到其他同名代码标识符实体
            sample_entity=entity[0]
            # if sample_entity["name_en"].lower()=='_s_':
            #     a=1
            #如果实体名内不含字母（如eid=13697),则跳过 
            if re.sub(r'[^a-zA-Z]', '', name)=='':
                continue
            
            if len(name)>=50:
                continue
            
            if "identifier_type" in sample_entity.keys():
                if sample_entity["identifier_type"]=="member":
                    continue
            for e in entity:
                if e["source_table"]=="entity_extraction":
                    usual_entity_results.add({"name":name,"source_table":e["source_table"]})
                else:
                    code_identifier_entity_results.add({"name":name,"source_table":e["source_table"]})
            entity_results.append(name)

    print(len(entity_results))
    parallel_pair(entity_results, n_process=1, chunk_size=10000)

    # Merge
    merge_chunk_files()

    # with open("./output/abbr_result/1211/match_res.json", "w") as f:
    #     json.dump(result, f, indent=4, ensure_ascii=False)
