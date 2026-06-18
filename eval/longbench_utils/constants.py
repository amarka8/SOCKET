LONGBENCH_E_DATASET = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
                        "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]

LONGBENCH_DATASET = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa","2wikimqa",
                    "musique","gov_report","qmsum","multi_news","trec","triviaqa","samsum",
                    "passage_retrieval_en", "lcc","repobench-p", "passage_count"]

# RULER-HARD-32K task subsets (config == split == task name on
# xAlg-AI/att-hub-ruler-32k). Kept separate from LONGBENCH_DATASET: these use
# substring metrics (string_match), self-contained prompts, and per-row
# max_new_tokens, NOT the LongBench qa_f1/rouge path.
RULER_DATASET = ["qa_1", "qa_2", "fwe", "vt", "niah_multikey_2", "niah_multikey_3"]