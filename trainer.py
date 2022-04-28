import torch
import os
import logging

logger = logging.getLogger(__name__)

class Trainer:

    def __init__(
        self, 
        model,
        device,
        optimizer,
        criteria,
        train_dataloader,
        dev_dataloader,
        batch_size, 
        learning_rate, 
        max_epoch, 
        max_update, 
        validate_interval_updates, 
        log_interval,
        checkpoint_dir
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criteria = criteria

        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader 

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.max_update = max_update
        self.validate_interval_updates = validate_interval_updates
        self.log_interval = log_interval
        self.checkpoint_dir = checkpoint_dir

        self.best_accuracy = 0

    def train_step(self, batch):
        self.model.train()
        input, target = batch[0], batch[1]

        self.model.train()
        output = self.model(input)
        loss = self.criteria(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def valid_step(self, batch):
        self.model.eval()

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        counter = 0
        with torch.no_grad():
            for batch in self.dev_dataloader:
                input, target = batch[0], batch[1]      

                output = self.model(input)
                loss = self.criteria(output, target).item()

                total_loss += float(loss)
                correct_predictions += (output.argmax(1) == target).type(torch.float).sum().item()
                total_predictions += len(target)
                counter += 1

        valid_loss = total_loss / counter
        valid_accuracy = correct_predictions / total_predictions
        return valid_loss, valid_accuracy

    def train(self):
        total_step = -1
        for epoch in range(self.max_epoch):
            for step, batch in enumerate(self.train_dataloader):
                total_step += 1
                train_loss = self.train_step(batch)

                if (total_step % self.log_interval) == 0:
                    logger.info(f"[TRAIN] Epoch={epoch} Step={total_step} Training_loss={train_loss:.5f}")

                if (total_step % self.validate_interval_updates) == 0:
                    logger.info("Begin validation ...")
                    valid_loss, valid_accuracy = self.valid_step(batch)
                    logger.info(f"[VALID] Total_step={total_step} Valid_loss={valid_loss:.5f} Valid_accuracy={valid_accuracy:.4f}")

                    if not os.path.exists(self.checkpoint_dir):
                        os.makedirs(self.checkpoint_dir)
                        
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"checkpoint_{total_step}.pt"))
                    if valid_accuracy > self.best_accuracy:
                        self.best_accuracy = valid_accuracy
                        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "checkpoint_best.pt"))