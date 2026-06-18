import pyarrow.parquet as pq
from pathlib import Path

QUESTION_TYPE = "relative_distance"
IMG_ID = 0


table = pq.read_table(f"data/{QUESTION_TYPE}/test-00000.parquet")
qa = {column: table[column][IMG_ID].as_py() for column in table.column_names}

if qa["user_1_question"] is None:
    answerer_image = qa["user_2_image"]
    helper_image = qa["user_1_image"]
else:
    answerer_image = qa["user_1_image"]
    helper_image = qa["user_2_image"]

output_dir = Path(".")
answerer_path = output_dir / f"{QUESTION_TYPE}_sample_{IMG_ID}_answerer.png"
helper_path = output_dir / f"{QUESTION_TYPE}_sample_{IMG_ID}_helper.png"

answerer_path.write_bytes(answerer_image["bytes"])
helper_path.write_bytes(helper_image["bytes"])

print(f"Answerer image saved to: {answerer_path}")
print(f"Helper image saved to: {helper_path}")


'''

python main.py   --tasks_qa_file data/relative_distance/test-00000.parquet   --experiment_variant two_agent+parallel   --answerer_client_name ali   --helper_client_name ali   --answerer_api_base https://dashscope.aliyuncs.com/compatible-mode/v1   --helper_api_base https://dashscope.aliyuncs.com/compatible-mode/v1   --answerer_model_name qwen3-vl-8b-instruct   --helper_model_name qwen3-vl-8b-instruct   --max_questions 3   --max_num_turns 5   --terminate

'''
