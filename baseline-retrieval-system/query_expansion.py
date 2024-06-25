from tira.third_party_integrations import ensure_pyterrier_is_loaded, persist_and_normalize_run
from tira.rest_api_client import Client
import pyterrier as pt

# Create a REST client to the TIRA platform for retrieving the pre-indexed data.
ensure_pyterrier_is_loaded()
tira = Client()

# The dataset: the union of the IR Anthology and the ACL Anthology
# This line creates an IRDSDataset object and registers it under the name provided as an argument.
dataset = 'ir-acl-anthology-20240504-training'
pt_dataset = pt.get_dataset('irds:ir-lab-sose-2024/ir-acl-anthology-20240504-training')


# A (pre-built) PyTerrier index loaded from TIRA
index = tira.pt.index('ir-lab-sose-2024/tira-ir-starter/Index (tira-ir-starter-pyterrier)', pt_dataset)


bm25 = pt.BatchRetrieve(index, wmodel="BM25")


qe = pt.rewrite.Bo1QueryExpansion(index)
qe_2 = pt.rewrite.KLQueryExpansion(index)
qe_3 = pt.rewrite.RM3(index)
qe_4 = pt.rewrite.Bo1QueryExpansion(index, fb_terms=20)
gpt_cot = tira.pt.transform_queries('workshop-on-open-web-search/tu-dresden-03/qe-gpt3.5-cot', dataset, prefix='llm_expansion_')
gpt_sq_fs = tira.pt.transform_queries('workshop-on-open-web-search/tu-dresden-03/qe-gpt3.5-sq-fs', dataset, prefix='llm_expansion_')
gpt_sq_zs = tira.pt.transform_queries('ir-benchmarks/tu-dresden-03/qe-gpt3.5-sq-zs', dataset, prefix='llm_expansion_')

tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

def pt_tokenize(text):
    return ' '.join(tokeniser.getTokens(text))

def expand_query(topic):
  ret = ' '.join([topic['query'], topic['query'], topic['query'],  topic['query'],  topic['query'], topic['llm_expansion_query']])

  # apply the tokenization
  return pt_tokenize(ret)

pt_expand_query = pt.apply.query(expand_query)



qe_pipeline = bm25 >> qe >> bm25
qe_pipeline_2 = bm25 >> qe_2 >> bm25
qe_pipeline_3 = bm25 >> qe_3 >> bm25
qe_pipeline_4 = bm25 >> qe_4 >> bm25
pipeline_gpt_cot = (gpt_cot >> pt_expand_query) >> bm25
pipeline_gpt_sq_fs = (gpt_sq_fs >> pt_expand_query) >> bm25
pipeline_gpt_sq_zs = (gpt_sq_zs >> pt_expand_query) >> bm25


run_base = bm25(pt_dataset.get_topics('text'))
run_qe = qe_pipeline(pt_dataset.get_topics('text')) #Bo1QueryExpansion
run_qe_2 = qe_pipeline_2(pt_dataset.get_topics('text')) #KLQueryExpansion
run_qe_3 = qe_pipeline_3(pt_dataset.get_topics('text')) #RM3
run_qe_4 = qe_pipeline_4(pt_dataset.get_topics('text')) #Bo1QueryExpansion 2x more terms
run_pipeline_gpt_cot = pipeline_gpt_cot(pt_dataset.get_topics('text'))
run_pipeline_gpt_sq_fs = pipeline_gpt_sq_fs(pt_dataset.get_topics('text'))
run_pipeline_gpt_sq_zs = pipeline_gpt_sq_zs(pt_dataset.get_topics('text'))

persist_and_normalize_run(run_base, system_name='bm25-baseline', default_output='../runs/base')
persist_and_normalize_run(run_qe, system_name='query-expansion_Bo1QE', default_output='../runs/qe')
persist_and_normalize_run(run_qe_2, system_name='query-expansion_KLQE', default_output='../runs/qe_2')
persist_and_normalize_run(run_qe_3, system_name='query-expansion_RM3', default_output='../runs/qe_3')
persist_and_normalize_run(run_qe_4, system_name='query-expansion_Bo1QEx2', default_output='../runs/qe_4')
persist_and_normalize_run(run_pipeline_gpt_cot, system_name='llm-query-expansion-gpt-cot', default_output='../runs/qe_gpt_cot')
persist_and_normalize_run(run_pipeline_gpt_sq_fs, system_name='llm-query-expansion-gpt-sq-fs', default_output='../runs/qe_gpt_sq_fs')
persist_and_normalize_run(run_pipeline_gpt_sq_zs, system_name='llm-query-expansion-gpt-sq-fz', default_output='../runs/qe_gpt_sq_fz')