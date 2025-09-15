from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.conponents.model_trainer import ModelTrainer
from textSummarizer.logging import logger


class modelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        
        # Get training stats
        training_stats = model_trainer.train()
        
        print("\nTraining Summary:")
        print(f"Device used: {training_stats['device_used']}")
        print(f"Total epochs: {training_stats['num_epochs']}")
        print(f"Total steps: {training_stats['total_steps']}")
        print(f"Total time taken: {training_stats['training_time']}")