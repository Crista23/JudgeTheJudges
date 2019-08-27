from __future__ import division
import csv
from collections import defaultdict
import collections
from datetime import datetime
from dateutil.parser import parse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

def days_hours_minutes(td):
	return td.days, td.seconds//3600, (td.seconds//60)%60

## Determine which workers participated in the pilot study ##
with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_eval_pilot/workers_pilot_study.pickle", "rb") as handle:
	worker_ids_pilot_study = pickle.load(handle)

print "*****Workers ids pilot study", worker_ids_pilot_study

## Determine which workers completed more than 1 HIT ##
with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/workers_duplicate.pickle", "rb") as handle:
	worker_ids_duplicate = pickle.load(handle)
duplicate_workers = [ worker_id for (worker_id, count) in worker_ids_duplicate ]
worker_ids_duplicate = duplicate_workers
print "******DUPLICATE WORKERS", worker_ids_duplicate

## Determine which workers did not answer all questions ##
with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/workers_incomplete_hits.pickle", "rb") as handle:
	worker_ids_incomplete = pickle.load(handle)
print "******Worker ids incomplete", worker_ids_incomplete

## Determine which workers failed to answer the gotcha question correctly ##
with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/workers_failed_gotcha.pickle", "rb") as handle:
	worker_ids_failed_gotcha = pickle.load(handle)
print "******Worker ids failed gotcha", worker_ids_failed_gotcha

workers_to_avoid = worker_ids_pilot_study.union(worker_ids_duplicate).union(worker_ids_incomplete).union(worker_ids_failed_gotcha)
print "Workers to avoid", workers_to_avoid
print "LEN(workers_to_avoid)", len(set(workers_to_avoid))


with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/Batch_3205459_batch_results_8th.csv", "rb") as csvfile:
 	mturk_pilot_annotations = csv.DictReader(csvfile, delimiter=',')

 	hit_ids = defaultdict(int)
 	worker_ids = list()
 	worker_time = defaultdict(int)
 	banned_workers_failed_gotcha = list()
 	worker_answers = defaultdict(dict)
 	workers_incomplete_hits = list()
 	accuracy_per_student = defaultdict(dict)

 	tp_all = list()
 	tn_all = list()
 	fn_all = list()
 	fp_all = list()

	student_WordRNN10 = defaultdict(int)
	overall_annotations_per_student_WordRNN10 = defaultdict(lambda: defaultdict(int))

	student_WordRNN07 = defaultdict(int)
	overall_annotations_per_student_WordRNN07 = defaultdict(lambda: defaultdict(int))

	student_WordRNN05 = defaultdict(int)
	overall_annotations_per_student_WordRNN05 = defaultdict(lambda: defaultdict(int))

	student_GoogleLM = defaultdict(int)
	overall_annotations_per_student_GoogleLM = defaultdict(lambda: defaultdict(int))

	student_AttentionAC = defaultdict(int)
	overall_annotations_per_student_AttentionAC = defaultdict(lambda: defaultdict(int))

	student_NoAttentionAC = defaultdict(int)
	overall_annotations_per_student_NoAttentionAC = defaultdict(lambda: defaultdict(int)) 	

	student_SkipConnections = defaultdict(int)
	overall_annotations_per_student_SkipConnections = defaultdict(lambda: defaultdict(int))

	student_MLESeqGAN = defaultdict(int)
	overall_annotations_per_student_MLESeqGAN = defaultdict(lambda: defaultdict(int))

	student_SS = defaultdict(int)
	overall_annotations_per_student_SS = defaultdict(lambda: defaultdict(int))

	student_SeqGAN = defaultdict(int)
	overall_annotations_per_student_SeqGAN = defaultdict(lambda: defaultdict(int))

	student_RankGAN = defaultdict(int)
	overall_annotations_per_student_RankGAN = defaultdict(lambda: defaultdict(int))

	student_LeakGAN = defaultdict(int)
	overall_annotations_per_student_LeakGAN = defaultdict(lambda: defaultdict(int))

	student_real = defaultdict(int)
	overall_annotations_per_student_REAL = defaultdict(lambda: defaultdict(int))

	student_fake = defaultdict(int)
	overall_annotations_per_student_FAKE = defaultdict(lambda: defaultdict(int))


	count_WordRNN10_all = 0
	count_WordRNN07_all = 0
	count_WordRNN05_all = 0
	count_GoogleLM_all = 0
	count_AttentionAC_all = 0
	count_NoAttentionAC_all = 0
	count_SkipConnections_all = 0
	count_MLESeqGAN_all = 0
	count_SS_all = 0
	count_SeqGAN_all = 0
	count_RankGAN_all = 0
	count_LeakGAN_all = 0

	count_real = 0
	count_fake = 0
	count_WordRNN10 = 0
	count_WordRNN07 = 0
	count_WordRNN05 = 0
	count_GoogleLM = 0
	count_AttentionAC = 0
	count_NoAttentionAC = 0
	count_SkipConnections = 0
	count_MLESeqGAN = 0
	count_SS = 0
	count_SeqGAN = 0
	count_RankGAN = 0
	count_LeakGAN = 0


	j = 0
	for row in mturk_pilot_annotations:
		#print "j=", j
		j += 1
		#print "ROW", row
		current_hit = row["HITId"]
		worker_id = row["WorkerId"]
		#print "Worker id", worker_id
		if (worker_id in workers_to_avoid):
			continue

		hit_ids[current_hit] += 1
		worker_ids.append(worker_id)

		#print "Original Start Time", row["AcceptTime"]
		#print "Original Start Time", row["SubmitTime"]

		start_time = row["AcceptTime"].split(" ") [0] + ", " + " ".join(row["AcceptTime"].split(" ")[1:4])
		end_time = row["SubmitTime"].split(" ") [0] + ", " + " ".join(row["SubmitTime"].split(" ")[1:4])
		
		worker_start_time = parse(start_time)
		worker_end_time = parse(end_time)

		#print "Worker start time", worker_start_time
		#print "Worker end time", worker_end_time

		minutes_spent_annotating = days_hours_minutes(worker_end_time - worker_start_time)[2]
		#print "TIME SPENT ANNOTATING (MINUTES)", minutes_spent_annotating
		worker_time[worker_id] = minutes_spent_annotating

		## Determine if the worker has correctly responded to the security question
		gotcha_loc = row["Input.gotcha_locs"]
		gotcha_question_answer = row["Answer.review_" + str(gotcha_loc) + "_judgment"]
		#print "GOTCHA ANSWER", gotcha_question_answer
		if gotcha_question_answer != "fake":
			banned_workers_failed_gotcha.append(worker_id)
			continue

		tp = 0
		tn = 0
		fn = 0
		fp = 0

		has_incomplete_answers = 0
		for i in range(0,21):
			# do not count the security question and its answer
			#print "i=", i
			if i == int(gotcha_loc):
				continue

			question_answer_worker = row["Answer.review_" + str(i) + "_judgment"]
			if (question_answer_worker.strip() == ""):
				workers_incomplete_hits.append(worker_id)
				has_incomplete_answers = 1
				break

			question_answer_gold_standard = row["Answer.{review_" + str(i) + "_student_model}"]
			question_text = row["Input.review_" + str(i) + "_text"]

			if question_answer_gold_standard == "Real":
				count_real += 1
				overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["text"] = question_text
				if question_answer_worker == "real":
					tp += 1

					student_real["correct"] += 1
					overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["correct"] += 1
					if (overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["correct"] == 5):
						overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["mistaken"] = 0
				elif question_answer_worker == "fake":
					fn += 1
					student_real["mistaken"] += 1
					overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["mistaken"] += 1
					if (overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
						overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["correct"] = 0
			else:
				## all negative gold standard answers
				count_fake += 1
				overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["text"] = question_text
				if question_answer_worker == "real":
					fp += 1
					student_fake["mistaken"] += 1
					overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["mistaken"] += 1
					if (overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
						overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["correct"] = 0

				elif question_answer_worker == "fake":
					student_fake["correct"] += 1
					tn += 1
					overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["correct"] += 1
					if (overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["correct"] == 5):
						overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				# mistaken means annotated as real, while correct means annotated as fake
				if (question_answer_gold_standard == "WordRNN10"):
					count_WordRNN10 += 1
					overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_WordRNN10["mistaken"] += 1
						overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_WordRNN10["correct"] += 1
						overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "WordRNN07"):
						count_WordRNN07 += 1
						overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["text"] = question_text
						if question_answer_worker == "real":
							student_WordRNN07["mistaken"] += 1
							overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["mistaken"] += 1
							if (overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
								overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["correct"] = 0
						elif question_answer_worker == "fake":
							student_WordRNN07["correct"] += 1
							overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["correct"] += 1
							if (overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["correct"] == 5):
								overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "WordRNN05"):
					count_WordRNN05 += 1
					overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_WordRNN05["mistaken"] += 1
						overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_WordRNN05["correct"] += 1
						overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "GoogleLM"):
					count_GoogleLM += 1
					overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_GoogleLM["mistaken"] += 1
						overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_GoogleLM["correct"] += 1
						overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "AttentionAC"):
					count_AttentionAC += 1
					overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_AttentionAC["mistaken"] += 1
						overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_AttentionAC["correct"] += 1
						overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "NoAttentionAC"):
					count_NoAttentionAC += 1
					overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_NoAttentionAC["mistaken"] += 1
						overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_NoAttentionAC["correct"] += 1
						overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "SkipConnectionsAC"): 
					count_SkipConnections += 1
					overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_SkipConnections["mistaken"] += 1
						overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_SkipConnections["correct"] += 1
						overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "MLESeqGAN"):
					count_MLESeqGAN += 1
					overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_MLESeqGAN["mistaken"] += 1
						overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_MLESeqGAN["correct"] += 1
						overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "SS"):
					count_SS += 1
					overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_SS["mistaken"] += 1
						overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_SS["correct"] += 1
						overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "SeqGAN"):
					count_SeqGAN += 1
					overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_SeqGAN["mistaken"] += 1
						overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_SeqGAN["correct"] += 1
						overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "RankGAN"):
					count_RankGAN += 1
					overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_RankGAN["mistaken"] += 1
						overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_RankGAN["correct"] += 1
						overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "LeakGAN"):
					count_LeakGAN += 1
					overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_LeakGAN["mistaken"] += 1
						overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_LeakGAN["correct"] += 1
						overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["mistaken"] = 0


		#### STATISTICS PER EXAMPLE INCLUDING ALL 5 WORKERS: each dictionary is of the form "HitID_QuestionID: {"correct": 3, "mistaken": 2}" ####
		print "WordRNN10 overall_annotations_per_student", len(overall_annotations_per_student_WordRNN10)
		print "WordRNN07 overall_annotations_per_student", len(overall_annotations_per_student_WordRNN07)
		print "WordRNN05 overall_annotations_per_student", len(overall_annotations_per_student_WordRNN05)
		print "GoogleLM overall_annotations_per_student", len(overall_annotations_per_student_GoogleLM)
		print "AttentionAC overall_annotations_per_student", len(overall_annotations_per_student_AttentionAC)
		print "No AttentionAC overall_annotations_per_student", len(overall_annotations_per_student_NoAttentionAC)
		print "Skip Connections overall_annotations_per_student", len(overall_annotations_per_student_SkipConnections)
		print "MLESeqGAN overall_annotations_per_student", len(overall_annotations_per_student_MLESeqGAN)
		print "SS overall_annotations_per_student", len(overall_annotations_per_student_SS)
		print "SeqGAN overall_annotations_per_student", len(overall_annotations_per_student_SeqGAN)
		print "RankGAN overall_annotations_per_student", len(overall_annotations_per_student_RankGAN)
		print "LeakGAN overall_annotations_per_student", len(overall_annotations_per_student_LeakGAN)
		print "REAL overall_annotations_per_student", len(overall_annotations_per_student_REAL)
		print "FAKE overall_annotations_per_student", len(overall_annotations_per_student_FAKE)


		## COUNT HOW MANY SUM UP TO 5
		total_complete_items_wordrnn10 = 0
		for hitID_questionID in overall_annotations_per_student_WordRNN10:
			if overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"] == 5:
				total_complete_items_wordrnn10 += 1
		print "TOTAL ITEMS WORDRNN10 5 ANNOTATORS:", total_complete_items_wordrnn10
		print "overall_annotations_per_student_WordRNN10", overall_annotations_per_student_WordRNN10
		print "student_WordRNN10", student_WordRNN10

		total_complete_items_wordrnn07 = 0
		for hitID_questionID in overall_annotations_per_student_WordRNN07:
			if overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"] == 5:
				total_complete_items_wordrnn07 += 1
		print "TOTAL ITEMS WORDRNN07 5 ANNOTATORS:", total_complete_items_wordrnn07

		total_complete_items_wordrnn05 = 0
		for hitID_questionID in overall_annotations_per_student_WordRNN05:
			if overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"] == 5:
				total_complete_items_wordrnn05 += 1
		print "TOTAL ITEMS WORDRNN05 5 ANNOTATORS:", total_complete_items_wordrnn05

		total_complete_items_googlelm = 0
		for hitID_questionID in overall_annotations_per_student_GoogleLM:
			if overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"] + overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"] == 5:
				total_complete_items_googlelm += 1
		print "TOTAL ITEMS GOOGLELM 5 ANNOTATORS:", total_complete_items_googlelm

		total_complete_items_attentionac = 0
		for hitID_questionID in overall_annotations_per_student_AttentionAC:
			if overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"] == 5:
				total_complete_items_attentionac += 1
		print "TOTAL ITEMS ATTENTION AC 5 ANNOTATORS:", total_complete_items_attentionac

		total_complete_items_noattentionac = 0
		for hitID_questionID in overall_annotations_per_student_NoAttentionAC:
			if overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"] == 5:
				total_complete_items_noattentionac += 1
		print "TOTAL ITEMS NO ATTENTION AC 5 ANNOTATORS:", total_complete_items_noattentionac

		total_complete_items_skipconnections = 0
		for hitID_questionID in overall_annotations_per_student_SkipConnections:
			if overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"] + overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"] == 5:
				total_complete_items_skipconnections += 1
		print "TOTAL ITEMS SKIP CONNECTIONS 5 ANNOTATORS:", total_complete_items_skipconnections

		total_complete_items_mleseqgan = 0
		for hitID_questionID in overall_annotations_per_student_MLESeqGAN:
			if overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"] == 5:
				total_complete_items_mleseqgan += 1
		print "TOTAL ITEMS MLE SEQGAN 5 ANNOTATORS:", total_complete_items_mleseqgan

		total_complete_items_ss = 0
		for hitID_questionID in overall_annotations_per_student_SS:
			if overall_annotations_per_student_SS[hitID_questionID]["correct"] + overall_annotations_per_student_SS[hitID_questionID]["mistaken"] == 5:
				total_complete_items_ss += 1
		print "TOTAL ITEMS SS 5 ANNOTATORS:", total_complete_items_ss

		total_complete_items_seqgan = 0
		for hitID_questionID in overall_annotations_per_student_SeqGAN:
			if overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"] == 5:
				total_complete_items_seqgan += 1
		print "TOTAL ITEMS SeqGAN 5 ANNOTATORS:", total_complete_items_seqgan

		total_complete_items_rankgan = 0
		for hitID_questionID in overall_annotations_per_student_RankGAN:
			if overall_annotations_per_student_RankGAN[hitID_questionID]["correct"] + overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"] == 5:
				total_complete_items_rankgan += 1
		print "TOTAL ITEMS RANKGAN 5 ANNOTATORS:", total_complete_items_rankgan

		total_complete_items_leakgan = 0
		for hitID_questionID in overall_annotations_per_student_LeakGAN:
			if overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"] + overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"] == 5:
				total_complete_items_leakgan += 1
		print "TOTAL ITEMS LEAKGAN 5 ANNOTATORS:", total_complete_items_leakgan

		total_complete_items_real = 0
		for hitID_questionID in overall_annotations_per_student_REAL:
			if overall_annotations_per_student_REAL[hitID_questionID]["correct"] + overall_annotations_per_student_REAL[hitID_questionID]["mistaken"] == 5:
				total_complete_items_real += 1
		print "TOTAL ITEMS REAL 5 ANNOTATORS:", total_complete_items_real

		total_complete_items_fake = 0
		for hitID_questionID in overall_annotations_per_student_FAKE:
			if overall_annotations_per_student_FAKE[hitID_questionID]["correct"] + overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"] == 5:
				total_complete_items_fake += 1
		print "TOTAL ITEMS REAL 5 ANNOTATORS:", total_complete_items_fake


		#### COMPUTE INTER_ANNOTATOR AGREEMENT WHEN EACH EXAMPLE IS ANNOTATED BY 5 WORKERS ####
		
		fleis_kappa_matrix_REAL = []
		for hitID_questionID in overall_annotations_per_student_REAL:
			#print "********Text", overall_annotations_per_student_REAL[hitID_questionID]["text"]
			if overall_annotations_per_student_REAL[hitID_questionID]["correct"] + overall_annotations_per_student_REAL[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_REAL[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_REAL[hitID_questionID]["mistaken"] == 3:
					print "REAL", overall_annotations_per_student_REAL[hitID_questionID]["text"], overall_annotations_per_student_REAL[hitID_questionID]["correct"], overall_annotations_per_student_REAL[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_REAL.append([overall_annotations_per_student_REAL[hitID_questionID]["mistaken"], overall_annotations_per_student_REAL[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix REAL", len(fleis_kappa_matrix_REAL)

		fleis_kappa_matrix_FAKE = []
		for hitID_questionID in overall_annotations_per_student_FAKE:
			if overall_annotations_per_student_FAKE[hitID_questionID]["correct"] + overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_FAKE[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"] == 3:
					print "FAKE", overall_annotations_per_student_FAKE[hitID_questionID]["text"], overall_annotations_per_student_FAKE[hitID_questionID]["correct"], overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_FAKE.append([overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"], overall_annotations_per_student_FAKE[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix FAKE", len(fleis_kappa_matrix_FAKE)

		fleis_kappa_matrix_ALL = fleis_kappa_matrix_FAKE + fleis_kappa_matrix_REAL
		print "Fleis Kappa matrix ALL", len(fleis_kappa_matrix_ALL)

		fleis_kappa_matrix_WordRNN10 = []
		for hitID_questionID in overall_annotations_per_student_WordRNN10:
			if overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"] == 3:
					print "WordRNN10", overall_annotations_per_student_WordRNN10[hitID_questionID]["text"], overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"], overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_WordRNN10.append([overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"], overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"]])
		print "Fleis Kappa matrix WordRNN10", len(fleis_kappa_matrix_WordRNN10)

		fleis_kappa_matrix_WordRNN07 = []
		for hitID_questionID in overall_annotations_per_student_WordRNN07:
			if overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"] == 3:
					print "WordRNN07", overall_annotations_per_student_WordRNN07[hitID_questionID]["text"], overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"], overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_WordRNN07.append([overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"], overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix WordRNN07", len(fleis_kappa_matrix_WordRNN07)

		fleis_kappa_matrix_WordRNN05 = []
		for hitID_questionID in overall_annotations_per_student_WordRNN05:
			if overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"] == 3:
					print "WordRNN05", overall_annotations_per_student_WordRNN05[hitID_questionID]["text"], overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"], overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_WordRNN05.append([overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"], overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix WordRNN05", len(fleis_kappa_matrix_WordRNN05)

		fleis_kappa_matrix_GoogleLM = []
		for hitID_questionID in overall_annotations_per_student_GoogleLM:
			if overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"] + overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"] == 3:
					print "GoogleLM", overall_annotations_per_student_GoogleLM[hitID_questionID]["text"], overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"], overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_GoogleLM.append([overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"], overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix GoogleLM", len(fleis_kappa_matrix_GoogleLM)

		fleis_kappa_matrix_AttentionAC = []
		for hitID_questionID in overall_annotations_per_student_AttentionAC:
			if overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"] == 3:
					print "AttentionAC", overall_annotations_per_student_AttentionAC[hitID_questionID]["text"], overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"], overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_AttentionAC.append([overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"], overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix AttentionAC", len(fleis_kappa_matrix_AttentionAC)

		fleis_kappa_matrix_NoAttentionAC = []
		for hitID_questionID in overall_annotations_per_student_NoAttentionAC:
			if overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"] == 3:
					print "NoAttentionAC", overall_annotations_per_student_NoAttentionAC[hitID_questionID]["text"], overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"], overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_NoAttentionAC.append([overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"], overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix NoAttentionAC", len(fleis_kappa_matrix_NoAttentionAC)	

		fleis_kappa_matrix_SkipConnections = []
		for hitID_questionID in overall_annotations_per_student_SkipConnections:
			if overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"] + overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"] == 3:
					print "SkipConnections", overall_annotations_per_student_SkipConnections[hitID_questionID]["text"], overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"], overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_SkipConnections.append([overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"], overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix SkipConnections", len(fleis_kappa_matrix_SkipConnections)

		fleis_kappa_matrix_MLESeqGAN = []
		for hitID_questionID in overall_annotations_per_student_MLESeqGAN:
			if overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"] == 3:
					print "MLESeqgan", overall_annotations_per_student_MLESeqGAN[hitID_questionID]["text"], overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"], overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_MLESeqGAN.append([overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"], overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix MLESeqgan", len(fleis_kappa_matrix_MLESeqGAN)

		fleis_kappa_matrix_SS = []
		for hitID_questionID in overall_annotations_per_student_SS:
			if overall_annotations_per_student_SS[hitID_questionID]["correct"] + overall_annotations_per_student_SS[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_SS[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_SS[hitID_questionID]["mistaken"] == 3:
					print "SS", overall_annotations_per_student_SS[hitID_questionID]["text"], overall_annotations_per_student_SS[hitID_questionID]["correct"], overall_annotations_per_student_SS[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_SS.append([overall_annotations_per_student_SS[hitID_questionID]["mistaken"], overall_annotations_per_student_SS[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix SS", len(fleis_kappa_matrix_SS)

		fleis_kappa_matrix_SeqGAN = []
		for hitID_questionID in overall_annotations_per_student_SeqGAN:
			if overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"] == 3:
					print "SeqGAN", overall_annotations_per_student_SeqGAN[hitID_questionID]["text"], overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"], overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_SeqGAN.append([overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"], overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix SeqGAN", len(fleis_kappa_matrix_SeqGAN)

		fleis_kappa_matrix_RankGAN = []
		for hitID_questionID in overall_annotations_per_student_RankGAN:
			if overall_annotations_per_student_RankGAN[hitID_questionID]["correct"] + overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_RankGAN[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"] == 3:
					print "RankGAN", overall_annotations_per_student_RankGAN[hitID_questionID]["text"], overall_annotations_per_student_RankGAN[hitID_questionID]["correct"], overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_RankGAN.append([overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"], overall_annotations_per_student_RankGAN[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix RankGAN", len(fleis_kappa_matrix_RankGAN)

		fleis_kappa_matrix_LeakGAN = []
		for hitID_questionID in overall_annotations_per_student_LeakGAN:
			if overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"] + overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"] == 5:
				if overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"] == 3 or overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"] == 3:
					print "LeakGAN", overall_annotations_per_student_LeakGAN[hitID_questionID]["text"], overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"], overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"]
				fleis_kappa_matrix_LeakGAN.append([overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"], overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"]])
		#print "Fleis Kappa matrix LeakGAN", len(fleis_kappa_matrix_LeakGAN)

