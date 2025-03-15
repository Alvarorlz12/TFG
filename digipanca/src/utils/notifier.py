import os
import requests
from datetime import datetime

class Notifier:
    """Telegram notifier."""
    def __init__(self, experiment):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.token or not self.chat_id:
            raise ValueError("Telegram token or chat ID not provided")

        self.url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        self.experiment = experiment

    def _escape_markdown(self, text):
        """Escapes special characters for Telegram MarkdownV2."""
        escape_chars = "_*[]()~`>#+-=|{}.!\\"
        return "".join(f"\\{char}" if char in escape_chars else char for char in text)

    def _generate_progress_bar(self, current_epoch, total_epochs, length=10):
        """Generates a sleek ASCII progress bar using ▰ and ▱."""
        progress = int((current_epoch / total_epochs) * length)
        filled = "█" * progress
        empty = "░" * (length - progress)
        percent = (current_epoch / total_epochs) * 100
        return f"{filled}{empty} {percent:.0f}%"

    def send_message(self, message):
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "MarkdownV2"
        }
        # requests.post(self.url, json=payload)
        response = requests.post(self.url, json=payload)
        if response.status_code != 200:
            print(f"Error sending message: {response.text}")

    def send_start_message(self, summary):
        """Send a message to Telegram when training starts."""
        message = (
            f"🚀 *TRAINING STARTED* 🚀\n"
            f"📌 *Experiment:* `{summary['experiment']}`\n"
            f"📝 *Description:* `{summary['description']}`\n"
            f"⚙️ *Configuration:*\n"
            f"   🔹 *Model:* `{summary['model_type']}`\n"
            f"   🔹 *Epochs:* `{summary['epochs']}`\n"
            f"   🔹 *Batch Size:* `{summary.get('batch_size', 'N/A')}`\n"
            f"   🔹 *Learning Rate:* `{summary.get('learning_rate', 'N/A')}`\n"
            f"   🔹 *Optimizer:* `{summary.get('optimizer', 'N/A')}`\n"
            f"   🔹 *Loss Function:* `{summary.get('loss_function', 'N/A')}`\n"
            f"📅 *Start Time:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
        )
        self.send_message(message)

    def send_progress_message(self, current_epoch, total_epochs, train_loss, 
                              val_loss, train_dice, val_dice):
        """Sends a progress update message to Telegram."""
        progress_bar = self._generate_progress_bar(current_epoch, total_epochs)

        message = (
            f"📢 *TRAINING UPDATE* 📢\n"
            f"📌 *Experiment:* `{self.experiment}`\n"
            f"📊 {progress_bar} \\[`{current_epoch}/{total_epochs}`\\]\n\n"
            f"📉 *Loss:* `{train_loss:.4f}` \\(Train\\) \\| `{val_loss:.4f}` \\(Val\\)\n"
            f"🎯 *Dice Score:* `{train_dice:.4f}` \\(Train\\) \\| `{val_dice:.4f}` \\(Val\\)\n"
        )

        self.send_message(message)


    def send_end_message(self, summary):
        """Send a message to Telegram when training ends."""
        total_epochs = summary["epochs"]
        completed_epochs = summary.get("completed_epochs", total_epochs)
        early_stopping = f"\\(EARLY STOPPED\\)" if completed_epochs < total_epochs else ""

        total_time = summary["training_time"]
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)

        message = (
            f"✅ *TRAINING COMPLETED* ✅ {early_stopping}\n"
            f"📌 *Experiment:* `{summary['experiment']}`\n"
            f"📅 *End Time:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n"
            f"⏳ *Duration:* `{int(hours):0>2}:{int(minutes):0>2}:{seconds:05.2f}`\n"
            f"📉 *Final Validation Loss:* `{summary['val_loss']:.4f}`\n"
            f"📊 *Final Metrics:*\n"
            f"   🔹 *IoU Score:* `{summary['metrics'].get('iou', 'N/A'):.4f}`\n"
            f"   🔹 *Dice Coefficient:* `{summary['metrics'].get('dice', 'N/A'):.4f}`\n"
            f"📈 *Best Model:*\n"
            f"   🔹 *Best Validation Loss:* `{summary['best_model']['loss']:.4f}`\n"
            f"   🔹 *Best Epoch:* `{summary['best_model']['epoch']}/{total_epochs}`\n"
            f"   🔹 *Best Metrics:*\n"
            f"      🔸 *IoU Score:* `{summary['best_model']['metrics'].get('iou', 'N/A'):.4f}`\n"
            f"      🔸 *Dice Coefficient:* `{summary['best_model']['metrics'].get('dice', 'N/A'):.4f}`\n"
            f"          🟢 *Pancreas:* `{summary['best_model']['metrics'].get('dice_class_1', 'N/A'):.4f}`\n"
            f"          🟣 *Tumor:* `{summary['best_model']['metrics'].get('dice_class_2', 'N/A'):.4f}`\n"
            f"      🔸 *Precision:* `{summary['best_model']['metrics'].get('precision', 'N/A'):.4f}`\n"
            f"      🔸 *Recall:* `{summary['best_model']['metrics'].get('recall', 'N/A'):.4f}`\n"
            f"🏁 *Epochs Completed:* `{completed_epochs}/{total_epochs}`\n"
        )
        self.send_message(message)

    def send_error_message(self, error_msg, epoch=None):
        """Send an error message to Telegram."""
        error_header = f"❌ *TRAINING ERROR* ❌\n📌 *Experiment:* `{self.experiment}`\n"
        if epoch is not None:
            message = f"{error_header}\n⚠️ *Error at epoch* `{epoch}`:\n{self._escape_markdown(error_msg)}"
        else:
            message = f"{error_header}\n⚠️ {self._escape_markdown(error_msg)}"
        
        self.send_message(message)