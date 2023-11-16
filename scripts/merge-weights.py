from peft import AutoPeftModelForCausalLM
import os
output_dir = "./results/sft-eval/checkpoint-7600"

model = AutoPeftModelForCausalLM.from_pretrained(output_dir)
model = model.merge_and_unload()
output_merged_dir = os.path.join(output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)