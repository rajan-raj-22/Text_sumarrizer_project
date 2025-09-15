from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch
import time
from datetime import datetime, timedelta

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
    
    def train(self):
        start_time = time.time()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
        
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, 
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,  # Changed from evaluation_strategy to eval_strategy
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )

        # Calculate estimated time
        num_epochs = self.config.num_train_epochs
        num_training_steps = len(dataset_samsum_pt["train"]) // self.config.per_device_train_batch_size * num_epochs
        print(f"\nTraining Config:")
        print(f"Number of epochs: {num_epochs}")
        print(f"Batch size: {self.config.per_device_train_batch_size}")
        print(f"Training samples: {len(dataset_samsum_pt['train'])}")
        print(f"Total training steps: {num_training_steps}")
        
        trainer = Trainer(
            model=model_pegasus, 
            args=trainer_args,
            tokenizer=tokenizer, 
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"], 
            eval_dataset=dataset_samsum_pt["validation"]
        )
        
        print("\nStarting training...")
        trainer.train()
        
        end_time = time.time()
        training_time = end_time - start_time
        training_time_str = str(timedelta(seconds=int(training_time)))
        
        print(f"\nTraining completed!")
        print(f"Total training time: {training_time_str}")
        
        # Save model and tokenizer
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))
        
        return {
            "training_time": training_time_str,
            "device_used": device,
            "num_epochs": num_epochs,
            "total_steps": num_training_steps
        }