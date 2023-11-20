from peft import AutoPeftModelForSeq2SeqLM
import os
output_dir = "./results/checkpoint-7000"

model = AutoPeftModelForSeq2SeqLM.from_pretrained(output_dir)
model = model.merge_and_unload()
output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)

# save tokenizer
tokenizer = model.get_tokenizer()
tokenizer.save_pretrained(output_merged_dir)
