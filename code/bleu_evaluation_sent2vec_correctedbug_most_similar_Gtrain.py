import pickle
import subprocess
from joblib import Parallel, delayed
import random

with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/D_test_normalized.pickle", "rb") as handle:
    d_test_normalized = pickle.load(handle)

with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/S_test_normalized.pickle", "rb") as handle:
    S_test_normalized = pickle.load(handle)

with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models/original_data_discriminator_cnn_final.pickle", "rb") as handle:
	(generated_data_idx_train, original_data_idx_train, generated_data_idx_val, original_data_idx_val, generated_data_idx_test, original_data_idx_test) = pickle.load(handle)


def try_parse_float(s, val=None):
	try:
		return float(s)
	except ValueError:
		return val

def get_similar_human_sentences(machine_sentence):
	process = subprocess.Popen("/srv/disk01/ggarbace/EvaluationProj/sent2vec/fasttext nnSent /srv/disk01/ggarbace/EvaluationProj/sent2vec/wiki_unigrams.bin /srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/G_train/Gtrain.txt 15 <<< '" + machine_sentence.replace("'", "") + "'", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
	out, err = process.communicate()
	return out

def process_output(results, original_data_idx_test, i, line):
	"""

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn05_most_similar/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn05_most_similar/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn05_most_similar/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn05_most_similar/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn05_most_similar/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn05_most_similar/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn05_most_similar/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn05_most_similar/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn05_most_similar/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn05_most_similar/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn07_most_similar/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn07_most_similar/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn07_most_similar/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn07_most_similar/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn07_most_similar/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn07_most_similar/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn07_most_similar/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn07_most_similar/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn07_most_similar/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn07_most_similar/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn10_most_similar/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn10_most_similar/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn10_most_similar/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn10_most_similar/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn10_most_similar/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn10_most_similar/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn10_most_similar/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn10_most_similar/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn10_most_similar/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/word_rnn10_most_similar/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/attention_ac/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/attention_ac/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/attention_ac/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/attention_ac/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/attention_ac/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/attention_ac/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/attention_ac/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/attention_ac/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/attention_ac/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/attention_ac/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/no_attention_ac/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/no_attention_ac/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/no_attention_ac/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/no_attention_ac/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/no_attention_ac/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/no_attention_ac/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/no_attention_ac/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/no_attention_ac/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/no_attention_ac/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/no_attention_ac/ref10.txt", "a+")
	
	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/skip_conn/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/skip_conn/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/skip_conn/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/skip_conn/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/skip_conn/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/skip_conn/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/skip_conn/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/skip_conn/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/skip_conn/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/skip_conn/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/mle_seqgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/mle_seqgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/mle_seqgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/mle_seqgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/mle_seqgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/mle_seqgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/mle_seqgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/mle_seqgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/mle_seqgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/mle_seqgan/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/ss/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/ss/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/ss/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/ss/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/ss/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/ss/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/ss/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/ss/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/ss/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/ss/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/rankgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/rankgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/rankgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/rankgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/rankgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/rankgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/rankgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/rankgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/rankgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/rankgan/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/leakgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/leakgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/leakgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/leakgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/leakgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/leakgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/leakgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/leakgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/leakgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/leakgan/ref10.txt", "a+")
	
	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/seqgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/seqgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/seqgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/seqgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/seqgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/seqgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/seqgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/seqgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/seqgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/seqgan/ref10.txt", "a+")
	"""

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/google_lm/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/google_lm/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/google_lm/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/google_lm/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/google_lm/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/google_lm/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/google_lm/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/google_lm/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/google_lm/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_Gtrain/google_lm/ref10.txt", "a+")


	"""
	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn05_most_similar/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn05_most_similar/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn05_most_similar/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn05_most_similar/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn05_most_similar/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn05_most_similar/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn05_most_similar/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn05_most_similar/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn05_most_similar/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn05_most_similar/ref10.txt", "a+")
	
	
	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn07_most_similar/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn07_most_similar/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn07_most_similar/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn07_most_similar/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn07_most_similar/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn07_most_similar/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn07_most_similar/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn07_most_similar/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn07_most_similar/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn07_most_similar/ref10.txt", "a+")
	
	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn10_most_similar/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn10_most_similar/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn10_most_similar/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn10_most_similar/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn10_most_similar/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn10_most_similar/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn10_most_similar/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn10_most_similar/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn10_most_similar/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/word_rnn10_most_similar/ref10.txt", "a+")
	
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/skip_connections/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/skip_connections/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/skip_connections/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/skip_connections/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/skip_connections/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/skip_connections/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/skip_connections/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/skip_connections/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/skip_connections/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/skip_connections/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/mle_seqgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/mle_seqgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/mle_seqgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/mle_seqgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/mle_seqgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/mle_seqgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/mle_seqgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/mle_seqgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/mle_seqgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/mle_seqgan/ref10.txt", "a+")
	
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/ss/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/ss/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/ss/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/ss/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/ss/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/ss/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/ss/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/ss/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/ss/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/ss/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/rankgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/rankgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/rankgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/rankgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/rankgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/rankgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/rankgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/rankgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/rankgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/rankgan/ref10.txt", "a+")
	
	
	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/leakgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/leakgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/leakgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/leakgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/leakgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/leakgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/leakgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/leakgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/leakgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/leakgan/ref10.txt", "a+")

	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/seqgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/seqgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/seqgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/seqgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/seqgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/seqgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/seqgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/seqgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/seqgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/seqgan/ref10.txt", "a+")
	
	
	
	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/google_lm/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/google_lm/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/google_lm/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/google_lm/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/google_lm/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/google_lm/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/google_lm/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/google_lm/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/google_lm/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/google_lm/ref10.txt", "a+")
	
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/d_test/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/d_test/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/d_test/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/d_test/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/d_test/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/d_test/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/d_test/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/d_test/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/d_test/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/d_test/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/s_test/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/s_test/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/s_test/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/s_test/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/s_test/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/s_test/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/s_test/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/s_test/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/s_test/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/s_test/ref10.txt", "a+")
	
	
	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn07_most_similar/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn07_most_similar/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn07_most_similar/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn07_most_similar/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn07_most_similar/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn07_most_similar/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn07_most_similar/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn07_most_similar/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn07_most_similar/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn07_most_similar/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn10_most_similar/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn10_most_similar/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn10_most_similar/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn10_most_similar/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn10_most_similar/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn10_most_similar/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn10_most_similar/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn10_most_similar/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn10_most_similar/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/word_rnn10_most_similar/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/attention_ac/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/attention_ac/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/attention_ac/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/attention_ac/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/attention_ac/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/attention_ac/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/attention_ac/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/attention_ac/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/attention_ac/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/attention_ac/ref10.txt", "a+")

	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/no_attention_ac/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/no_attention_ac/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/no_attention_ac/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/no_attention_ac/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/no_attention_ac/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/no_attention_ac/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/no_attention_ac/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/no_attention_ac/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/no_attention_ac/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug/no_attention_ac/ref10.txt", "a+")
	
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_mle_seqgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_mle_seqgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_mle_seqgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_mle_seqgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_mle_seqgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_mle_seqgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_mle_seqgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_mle_seqgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_mle_seqgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_mle_seqgan/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_ss/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_ss/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_ss/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_ss/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_ss/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_ss/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_ss/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_ss/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_ss/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_ss/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_rankgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_rankgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_rankgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_rankgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_rankgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_rankgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_rankgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_rankgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_rankgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_rankgan/ref10.txt", "a+")


	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_leakgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_leakgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_leakgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_leakgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_leakgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_leakgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_leakgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_leakgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_leakgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_leakgan/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_seqgan/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_seqgan/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_seqgan/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_seqgan/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_seqgan/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_seqgan/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_seqgan/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_seqgan/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_seqgan/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_seqgan/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_skip_connections/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_skip_connections/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_skip_connections/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_skip_connections/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_skip_connections/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_skip_connections/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_skip_connections/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_skip_connections/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_skip_connections/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_skip_connections/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_googlelm/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_googlelm/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_googlelm/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_googlelm/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_googlelm/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_googlelm/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_googlelm/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_googlelm/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_googlelm/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/most_similar_googlelm/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_human_scores/most_similar_references/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_human_scores/most_similar_references/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_human_scores/most_similar_references/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_human_scores/most_similar_references/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_human_scores/most_similar_references/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_human_scores/most_similar_references/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_human_scores/most_similar_references/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_human_scores/most_similar_references/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_human_scores/most_similar_references/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_human_scores/most_similar_references/ref10.txt", "a+")
	

	h1 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_machine_scores/most_similar_references/ref1.txt", "a+")
	h2 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_machine_scores/most_similar_references/ref2.txt", "a+")
	h3 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_machine_scores/most_similar_references/ref3.txt", "a+")
	h4 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_machine_scores/most_similar_references/ref4.txt", "a+")
	h5 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_machine_scores/most_similar_references/ref5.txt", "a+")
	h6 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_machine_scores/most_similar_references/ref6.txt", "a+")
	h7 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_machine_scores/most_similar_references/ref7.txt", "a+")
	h8 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_machine_scores/most_similar_references/ref8.txt", "a+")
	h9 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_machine_scores/most_similar_references/ref9.txt", "a+")
	h10 = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/bleu_evaluation/upper_bound_machine_scores/most_similar_references/ref10.txt", "a+")
	"""

	## parse the output to retrieve the returned sentences
	out_lines = results.split("\n")
	out_lines = [ line for line in out_lines if line != "" ]
	
	if len(out_lines) == 0:
		print "NOT FOUND!!!"
		print "i=", i
		print "Original line", line

	#print "OUT_LINES", out_lines
	out_lines = out_lines[ : 11]
	for i in range(1, len(out_lines)):
		## parse the content of the line
		original_line = out_lines[i] 
		splitted_line = original_line.split(" ")
		#print "splitted_line[0]", splitted_line[0]
		#print "splitted_line[1]", splitted_line[1]
		if (try_parse_float(splitted_line[0]) is None or try_parse_float(splitted_line[1]) is None):
			continue

		current_line = " ".join(splitted_line[2:])
		#print "ORIGINAL LINE", original_line
		#print "SPLITTED LINE", current_line
		if (i == 1):
			h1.write(current_line + "\n")
		elif (i == 2):
			h2.write(current_line + "\n")
		elif (i == 3):
			h3.write(current_line + "\n")
		elif (i==4):
			h4.write(current_line + "\n")
		elif (i==5):
			h5.write(current_line + "\n")
		elif (i==6):
			h6.write(current_line + "\n")
		elif (i==7):
			h7.write(current_line + "\n")
		elif (i==8):
			h8.write(current_line + "\n")
		elif (i==9):
			h9.write(current_line + "\n")
		elif (i==10):
			h10.write(current_line + "\n")

	if len(out_lines) < 10:
		diff = 10 - len(out_lines)
		#randomly select human sentences
		random_human_examples = random.sample(original_data_idx_test, diff + 1)
		print "RANDOM HUMAN EXAMPLES", len(random_human_examples)
		for j in range(len(random_human_examples)):
			current_line = random_human_examples[j][0]
			if (j + len(out_lines) == 1):
				h1.write(current_line + "\n")
			elif (j + len(out_lines) == 2):
				h2.write(current_line + "\n")
			elif (j + len(out_lines) == 3):
				h3.write(current_line + "\n")
			elif (j + len(out_lines) == 4):
				h4.write(current_line + "\n")
			elif (j + len(out_lines) == 5):
				h5.write(current_line + "\n")
			elif (j + len(out_lines) == 6):
				h6.write(current_line + "\n")
			elif (j + len(out_lines) == 7):
				h7.write(current_line + "\n")
			elif (j + len(out_lines) == 8):
				h8.write(current_line + "\n")
			elif (j + len(out_lines) == 9):
				h9.write(current_line + "\n")
			elif (j + len(out_lines) == 10):
				h10.write(current_line + "\n")

	h1.flush()
	h2.flush()
	h3.flush()
	h4.flush()
	h5.flush()
	h6.flush()
	h7.flush()
	h8.flush()
	h9.flush()
	h10.flush()

	h1.close()
	h2.close()
	h3.close()
	h4.close()
	h5.close()
	h6.close()
	h7.close()
	h8.close()
	h9.close()
	h10.close()



### for each machine generated sentence retrieve the top-n most similar sentences and write each to file ###
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_WordRNN05.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_WordRNN07.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_WordRNN10.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_AttentionAC.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_NoAttentionAC.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_SkipConnectionsAC.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_MLESeqGAN.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_SS.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_RankGAN.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_LeakGAN.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_SeqGAN.txt", "rb") as f:
with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/student_hypothesis/student_GoogleLM.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/human_examples_Dtest_all_normalized.txt", "rb") as f:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/machine_examples_Stest_all_normalized.txt", "rb") as f:
	lines = f.readlines()

results = Parallel(n_jobs=48)(delayed(get_similar_human_sentences)(line) for line in lines)
#print "RESULTS", results

Parallel(n_jobs=30)(delayed(process_output)(results[i], original_data_idx_test, i, lines[i]) for i in range(len(results)))

