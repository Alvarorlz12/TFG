import os
import requests
import json
import gspread

from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta

class Notifier:
    """Telegram notifier."""
    def __init__(self, experiment, config_path="configs/bot_config.json",
                 only_save=False):
        """
        Initialize the Telegram notifier.
        
        Parameters
        ----------
        experiment : str
            The name of the experiment.
        config_path : str, optional
            Path to the Telegram bot configuration file.
        only_save : bool, optional
            If True, only connect to Google Sheets and skip Telegram.
        """
        self.experiment = experiment
        self.config_path = config_path
        self.only_save = only_save
        self._load_config()
        self._connect_to_google_sheets()

        self.url = f"https://api.telegram.org/bot{self.token}"

        if not only_save:
            # Get or create the topic for the experiment
            self.topic_id = self._get_topic_id(self.experiment)
            if self.topic_id is None:
                self.topic_id = self._create_topic(experiment)
                if self.topic_id is not None:
                    self._save_topic_id(self.experiment, self.topic_id)
                else:
                    raise ValueError(f"Error creating topic {self.experiment}.")

    def _load_config(self):
        """Load the Telegram bot configuration."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            config = json.load(f)

        self.token = config.get("TOKEN")
        self.chat_id = config.get("CHAT_ID")
        self.sheet_id = config.get("SHEET_ID")

        if self.token is None or self.chat_id is None or self.sheet_id is None:
            raise ValueError("TELEGRAM_TOKEN, CHAT_ID, or SHEET_ID is missing in the config file.")
        
    def _connect_to_google_sheets(self):
        """Connect to Google Sheets."""
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name("configs/credentials.json", scope)
        client = gspread.authorize(credentials)
        self.sheet = client.open_by_key(self.sheet_id)
        self.topics_sheet = self.sheet.worksheet("Topics")
        self.results_sheet = self.sheet.worksheet("Results")

    def _get_topic_id(self, experiment):
        """Get the topic ID from the Google Sheet."""
        topics = self.topics_sheet.get_all_records()
        for row in topics:
            if row["name"] == experiment:
                return row["thread_id"]
        return None
    
    def _save_topic_id(self, experiment, thread_id):
        """Save the topic ID to the Google Sheet."""
        self.topics_sheet.append_row([experiment, thread_id])

    def _create_topic(self, topic_name):
        """Create a new topic for the experiment."""
        if topic_name is None:
            raise ValueError("Topic name is missing.")
        payload = {
            "chat_id": self.chat_id,
            "name": topic_name
        }
        response = requests.post(f"{self.url}/createForumTopic", json=payload)
        if response.status_code == 200:
            return response.json()["result"]["message_thread_id"]
        else:
            print(f"Error creating topic {topic_name}: {response.text}")
            return None

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
        """Send a message to Telegram."""
        if self.only_save:
            return
        if self.topic_id is None:
            raise ValueError("Topic ID is missing.")
        
        payload = {
            "chat_id": self.chat_id,
            "message_thread_id": self.topic_id,
            "text": message,
            "parse_mode": "MarkdownV2"
        }
        response = requests.post(f"{self.url}/sendMessage", json=payload)
        if response.status_code != 200:
            print(f"Error sending message: {response.text}")

    def send_start_message(self, summary):
        """Send a message to Telegram when training starts."""
        if self.only_save:
            return
        config_file = os.path.basename(summary["config_file"]).replace(".yaml", "")
        message = (
            f"🚀 *TRAINING STARTED* 🚀\n"
            f"📌 *Experiment:* `{summary['experiment']}`\n"
            f"📝 *Description:* `{summary['description']}`\n"
            f"⚙️ *Configuration:* `{config_file}`\n"
            f"   🔹 *Model:* `{summary['model_type']}`\n"
            f"   🔹 *Epochs:* `{summary['epochs']}`\n"
            f"   🔹 *Batch Size:* `{summary.get('batch_size', 'N/A')}`\n"
            f"   🔹 *Learning Rate:* `{summary.get('learning_rate', 'N/A')}`\n"
            f"   🔹 *Optimizer:* `{summary.get('optimizer', 'N/A')}`\n"
            f"   🔹 *Loss Function:* `{summary.get('loss_function', 'N/A')}`\n"
            f"📅 *Start Time:* `{summary['start_time']}`\n"
        )
        self.send_message(message)

    def send_progress_message(self, current_epoch, total_epochs, train_loss, 
                              val_loss, train_dice, val_dice):
        """Sends a progress update message to Telegram."""
        if self.only_save:
            return
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
        if self.only_save:
            return
        total_epochs = summary["epochs"]
        completed_epochs = summary.get("completed_epochs", total_epochs)
        early_stopping = f"\\(EARLY STOPPED\\)" if completed_epochs < total_epochs else ""

        total_time = summary["training_time"]
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)

        message = (
            f"✅ *TRAINING COMPLETED* ✅ {early_stopping}\n"
            f"📌 *Experiment:* `{summary['experiment']}`\n"
            f"📅 *End Time:* `{summary['end_time']}`\n"
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
        if self.only_save:
            return
        error_header = f"❌ *TRAINING ERROR* ❌\n📌 *Experiment:* `{self.experiment}`\n"
        if epoch is not None:
            message = f"{error_header}\n⚠️ *Error at epoch* `{epoch}`:\n{self._escape_markdown(error_msg)}"
        else:
            message = f"{error_header}\n⚠️ {self._escape_markdown(error_msg)}"
        
        self.send_message(message)

    def delete_topic(self):
        """Deletes the topic for the experiment."""
        if self.topic_id is None:
            raise ValueError("Topic ID is missing.")
        topic_row = self.topics_sheet.find(self.experiment).row
        if topic_row is None:
            raise ValueError(f"Topic {self.experiment} not found in the Google Sheet.")
        
        url = f"{self.url}/deleteForumTopic"
        payload = {
            "chat_id": self.chat_id,
            "message_thread_id": self.topic_id
        }
        response = requests.post(url=url, json=payload)
        if response.status_code != 200:
            print(f"Error deleting topic {self.experiment}: {response.text}")
            return
        else:
            print(f"Topic {self.experiment} deleted successfully.")
            # Remove the topic from Google Sheets
            self.topics_sheet.delete_rows(topic_row)

    def save_results(self, summary):
        """Save the results to Google Sheets."""
        row = [
            summary['experiment'],
            os.path.basename(summary['config_file']).replace(".yaml", ""),
            summary['model_type'],
            summary.get('batch_size', 'N/A'),
            summary.get('learning_rate', 'N/A'),
            summary.get('optimizer', 'N/A'),
            summary.get('loss_function', 'N/A'),
            summary.get('completed_epochs', summary['epochs']),
            summary['epochs'],
            summary['best_model']['epoch'],
            *[f"{value:.5f}".replace(".", ",") if isinstance(value, float) else value
                for value in [
                    summary['train_loss'], summary['val_loss'], summary['train_metrics']['dice'],
                    summary['metrics']['dice'], summary['best_model']['loss'], 
                    summary['best_model']['metrics'].get('dice', 'N/A'),
                    summary['best_model']['metrics'].get('dice_class_0', 'N/A'),
                    summary['best_model']['metrics'].get('dice_class_1', 'N/A'),
                    summary['best_model']['metrics'].get('dice_class_2', 'N/A'),
                    summary['best_model']['metrics'].get('dice_class_3', 'N/A'),
                    summary['best_model']['metrics'].get('dice_class_4', 'N/A')
                ]
            ],
            str(timedelta(seconds=summary['training_time']))[:-3],
            summary['start_time'],
            summary['end_time'],
            summary['experiment_dir']
        ]
        self.results_sheet.append_row(row)