import os
import requests
from datetime import datetime

class Notifier:
    """Telegram notifier."""
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        if not self.token or not self.chat_id:
            raise ValueError("Telegram token or chat ID not provided")

        self.url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, message):
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "MarkdownV2"
        }
        requests.post(self.url, json=payload)

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

    def send_end_message(self, summary):
        """Send a message to Telegram when training ends."""
        total_epochs = summary["epochs"]
        completed_epochs = summary.get("completed_epochs", total_epochs)
        early_stopping = "(EARLY STOPPED)" if completed_epochs < total_epochs else ""

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
            f"      🔸 *Precision:* `{summary['best_model']['metrics'].get('precision', 'N/A'):.4f}`\n"
            f"      🔸 *Recall:* `{summary['best_model']['metrics'].get('recall', 'N/A'):.4f}`\n"
            f"🏁 *Epochs Completed:* `{completed_epochs}/{total_epochs}`\n"
        )
        self.send_message(message)